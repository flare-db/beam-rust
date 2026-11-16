/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_io::{ArrowRecordBatch, ArrowRecordBatchPCollectionExt};
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use beam::coders::DefaultCoder;
use beam::internals::PipelineInternals;
use beam::pipeline::{Pipeline, PipelineError};
use beam::transforms::{
    CoGbkResult, CoGroupByKey, DoFn, KeyedPCollectionTuple, PTransform, ParDo, ProcessContext,
};
use beam::values::{IsBounded, PCollection, PCollectionList};
use beam::{Error, Result};
use datafusion::common::NullEquality;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::logical_expr::JoinType;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr_common::metrics;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::collect;
use datafusion::physical_plan::joins::utils::build_join_schema;
use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion::physical_plan::repartition::BatchPartitioner;
use datafusion::prelude::SessionContext;

pub(crate) struct Join {
    left_expressions: Vec<Arc<dyn PhysicalExpr>>,
    right_expressions: Vec<Arc<dyn PhysicalExpr>>,
    join_type: JoinType,
    num_partitions: usize,
}

impl Join {
    const DEFAULT_NUM_PARTITIONS: usize = 2;

    /// Creates an equi-join on the left and right key expressions.
    pub(crate) fn new(
        left_expressions: Vec<Arc<dyn PhysicalExpr>>,
        right_expressions: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self {
            left_expressions,
            right_expressions,
            join_type: JoinType::Inner,
            num_partitions: Self::DEFAULT_NUM_PARTITIONS,
        }
    }

    /// Sets the join type. The default is [`JoinType::Inner`].
    pub(crate) fn with_join_type(mut self, join_type: JoinType) -> Self {
        self.join_type = join_type;
        self
    }

    /// Sets the number of hash partitions that each side is shuffled into.
    pub(crate) fn with_num_partitions(mut self, num_partitions: usize) -> Self {
        self.num_partitions = num_partitions;
        self
    }
}

impl PTransform<PCollectionList<ArrowRecordBatch>> for Join {
    type Output = PCollection<ArrowRecordBatch>;

    fn expand(&self, input: &PCollectionList<ArrowRecordBatch>) -> Self::Output {
        let (Some(left_rows), Some(right_rows)) = (input.get(0), input.get(1)) else {
            return join_error_output(input.pipeline(), "Join requires two input PCollections");
        };
        let (Some(left_schema), Some(right_schema)) =
            (left_rows.arrow_schema(), right_rows.arrow_schema())
        else {
            return join_error_output(
                input.pipeline(),
                "Join inputs need an Arrow schema; call with_arrow_schema first",
            );
        };

        let (output_schema, _) =
            build_join_schema(left_schema.as_ref(), right_schema.as_ref(), &self.join_type);

        let left_partition = left_rows.apply(
            "Partition left side",
            ParDo::new(PartitionFn::new(
                self.num_partitions,
                self.left_expressions.clone(),
            )),
        );

        let right_partition = right_rows.apply(
            "Partition right side",
            ParDo::new(PartitionFn::new(
                self.num_partitions,
                self.right_expressions.clone(),
            )),
        );

        let tuple =
            KeyedPCollectionTuple::of("left", &left_partition).and("right", &right_partition);

        let grouped = tuple.apply("CoGroupByPartition", CoGroupByKey::new());

        let joined = grouped.apply(
            "HashJoin",
            ParDo::new(HashJoinFn::new(
                self.left_expressions.clone(),
                self.right_expressions.clone(),
                left_schema,
                right_schema,
                self.join_type,
            )),
        );

        joined.with_arrow_schema(&Arc::new(output_schema))
    }
}

/// Performs the join on the shuffled batches of one hash partition.
#[derive(Clone)]
struct HashJoinFn {
    left_expressions: Vec<Arc<dyn PhysicalExpr>>,
    right_expressions: Vec<Arc<dyn PhysicalExpr>>,
    left_schema: SchemaRef,
    right_schema: SchemaRef,
    join_type: JoinType,
    context: SessionContext,
}

impl HashJoinFn {
    /// Creates a hash-join function for one shuffled partition.
    fn new(
        left_expressions: Vec<Arc<dyn PhysicalExpr>>,
        right_expressions: Vec<Arc<dyn PhysicalExpr>>,
        left_schema: SchemaRef,
        right_schema: SchemaRef,
        join_type: JoinType,
    ) -> Self {
        Self {
            left_expressions,
            right_expressions,
            left_schema,
            right_schema,
            join_type,
            context: SessionContext::new(),
        }
    }
}

impl DoFn for HashJoinFn {
    type In = (i32, CoGbkResult);
    type Out = ArrowRecordBatch;

    fn process_element(
        &mut self,
        (_partition, grouped): Self::In,
        ctx: &mut ProcessContext<'_, Self::Out>,
    ) -> Result {
        let left_batches = grouped
            .get_vec::<ArrowRecordBatch>("left")?
            .into_iter()
            .map(ArrowRecordBatch::into_inner)
            .collect::<Vec<_>>();
        let right_batches = grouped
            .get_vec::<ArrowRecordBatch>("right")?
            .into_iter()
            .map(ArrowRecordBatch::into_inner)
            .collect::<Vec<_>>();

        let left =
            MemorySourceConfig::try_new_exec(&[left_batches], Arc::clone(&self.left_schema), None)?;
        let right = MemorySourceConfig::try_new_exec(
            &[right_batches],
            Arc::clone(&self.right_schema),
            None,
        )?;

        let on = self
            .left_expressions
            .iter()
            .cloned()
            .zip(self.right_expressions.iter().cloned())
            .collect::<Vec<_>>();

        let join = HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &self.join_type,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?;

        let plan: Arc<dyn ExecutionPlan> = Arc::new(join);
        let batches = futures::executor::block_on(collect(plan, self.context.task_ctx()))?;
        for batch in batches {
            ctx.emit(ArrowRecordBatch::new(batch))?;
        }
        Ok(())
    }
}

/// Records a construction error and returns a placeholder output for a bad `Join` input.
fn join_error_output(pipeline: &Pipeline, reason: &str) -> PCollection<ArrowRecordBatch> {
    let name = pipeline.scope_name().unwrap_or_default();
    pipeline.record_construction_error(PipelineError::InvalidConstruction {
        transform_id: name.clone(),
        reason: reason.to_string(),
    });
    let coder_id = ArrowRecordBatch::register_coder(pipeline);
    pipeline
        .add_pcollection(&format!("{name}.out"), &coder_id, IsBounded::Bounded)
        .with_arrow_schema(&ArrowSchema::empty())
}

/// Hash-partitions every Arrow RecordBatch by its join keys.
struct PartitionFn {
    num_partitions: usize,
    hash_exprs: Vec<Arc<dyn PhysicalExpr>>,
    partitioner: Option<BatchPartitioner>,
}

impl PartitionFn {
    fn new(num_partitions: usize, hash_exprs: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        Self {
            num_partitions,
            hash_exprs,
            partitioner: None,
        }
    }

    /// Partitions an Arrow RecordBatch by hashing its join keys.
    ///
    /// Each output contains a partition ID and the subset of rows
    /// belonging to that hash partition.
    fn partition_batch(&mut self, batch: RecordBatch) -> Result<Vec<(i32, RecordBatch)>> {
        let partitioner = self
            .partitioner
            .as_mut()
            .ok_or_else(|| Error::msg("PartitionFn::setup was not called"))?;

        partitioner
            .partition_iter(batch)
            .map_err(Error::from)?
            .map(|piece| {
                piece
                    .map(|(id, batch)| (id as i32, batch))
                    .map_err(Error::from)
            })
            .collect()
    }
}

impl Clone for PartitionFn {
    /// Clones the configuration; the new instance builds its own partitioner in `setup`.
    fn clone(&self) -> Self {
        Self {
            num_partitions: self.num_partitions,
            hash_exprs: self.hash_exprs.clone(),
            partitioner: None,
        }
    }
}

impl DoFn for PartitionFn {
    type In = ArrowRecordBatch;
    type Out = (i32, ArrowRecordBatch);

    fn setup(&mut self) -> Result {
        self.partitioner = Some(
            BatchPartitioner::new_hash_partitioner(
                self.hash_exprs.clone(),
                self.num_partitions,
                metrics::Time::new(),
            )
            .map_err(Error::from)?,
        );
        Ok(())
    }

    fn process_element(
        &mut self,
        batch: Self::In,
        ctx: &mut ProcessContext<'_, Self::Out>,
    ) -> Result {
        for (id, piece) in self.partition_batch(batch.into_inner())? {
            ctx.emit((id, ArrowRecordBatch::new(piece)))?;
        }
        Ok(())
    }
}
