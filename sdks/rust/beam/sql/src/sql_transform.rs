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
use std::collections::{BTreeMap, BTreeSet};

use arrow_io::{ArrowRecordBatch, ArrowRecordBatchPCollectionExt};
use arrow_schema::Schema as ArrowSchema;
use beam::Error;
use beam::coders::DefaultCoder;
use beam::internals::PipelineInternals;
use beam::pipeline::Pipeline;
use beam::pipeline::PipelineError;
use beam::transforms::PTransform;
use beam::values::{IsBounded, PCollection, PCollectionTuple};
use datafusion::error::DataFusionError;

use crate::planner::DataFusionQueryPlanner;
use crate::rel::{BeamPCollectionTable, BeamSqlRelUtils, SqlRelContext};

const PCOLLECTION_NAME: &str = "PCOLLECTION";
const DEFAULT_PARTITION_SIZE: usize = 2;

/// Beam SQL transform.
#[derive(Clone, Debug)]
pub struct SqlTransform {
    query: String,
    partition_size: usize,
}

impl SqlTransform {
    /// Creates a builder for a Beam SQL transform.
    fn builder(query: impl Into<String>) -> SqlTransformBuilder {
        SqlTransformBuilder {
            query: query.into(),
            partition_size: DEFAULT_PARTITION_SIZE,
        }
    }

    /// Creates a transform for `query` with the default options.
    pub fn query(query: impl Into<String>) -> Self {
        Self::builder(query).build()
    }

    /// Sets how many post-shuffle key groups one DataFusion join execution receives.
    pub fn with_partition_size(self, size: usize) -> Self {
        self.into_builder().with_partition_size(size).build()
    }

    fn into_builder(self) -> SqlTransformBuilder {
        SqlTransformBuilder {
            query: self.query,
            partition_size: self.partition_size,
        }
    }
}

/// Builder for an immutable [`SqlTransform`].
#[derive(Clone, Debug)]
struct SqlTransformBuilder {
    query: String,
    partition_size: usize,
}

impl SqlTransformBuilder {
    fn with_partition_size(mut self, size: usize) -> Self {
        self.partition_size = size;
        self
    }

    fn build(self) -> SqlTransform {
        SqlTransform {
            query: self.query,
            partition_size: self.partition_size,
        }
    }
}

impl SqlTransform {
    fn execute(
        &self,
        pipeline: &Pipeline,
        tables: Result<BTreeMap<String, BeamPCollectionTable>, SqlTransformError>,
    ) -> PCollection<ArrowRecordBatch> {
        let result = tables.and_then(|tables| {
            if self.partition_size == 0 {
                return Err(SqlTransformError::message(
                    "partition_size must be greater than zero",
                ));
            }

            if tables.is_empty() {
                return Err(SqlTransformError::message(
                    "SqlTransform requires at least one input table",
                ));
            }

            let planner = DataFusionQueryPlanner::new(&tables, self.partition_size)?;
            let node = futures::executor::block_on(planner.build_plan(&self.query))?;

            let context = SqlRelContext::new(&tables);
            BeamSqlRelUtils::to_pcollection(node.as_ref(), &context)
        });

        match result {
            Ok(output) => output,
            Err(error) => {
                let name = pipeline.primitive_name("SqlTransformError");

                pipeline.record_construction_error(PipelineError::InvalidConstruction {
                    transform_id: name.clone(),
                    reason: error.to_string(),
                });

                let coder_id = ArrowRecordBatch::register_coder(pipeline);

                pipeline
                    .add_pcollection(&format!("{name}.out"), &coder_id, IsBounded::Bounded)
                    .with_arrow_schema(&ArrowSchema::empty())
            }
        }
    }
}

impl PTransform<PCollection<ArrowRecordBatch>> for SqlTransform {
    type Output = PCollection<ArrowRecordBatch>;

    fn expand(&self, input: &PCollection<ArrowRecordBatch>) -> Self::Output {
        self.execute(input.pipeline(), resolve_single_input_table(input))
    }
}

impl PTransform<PCollectionTuple> for SqlTransform {
    type Output = PCollection<ArrowRecordBatch>;

    fn expand(&self, input: &PCollectionTuple) -> Self::Output {
        self.execute(input.pipeline(), resolve_input_tables(input))
    }
}

fn resolve_input_tables(
    tuple: &PCollectionTuple,
) -> Result<BTreeMap<String, BeamPCollectionTable>, SqlTransformError> {
    tuple
        .iter()
        .try_fold(
            (BTreeSet::new(), BTreeMap::new()),
            |(mut names, mut inputs), (name, collection)| {
                if !names.insert(name.to_ascii_lowercase()) {
                    return Err(SqlTransformError::message(format!(
                        "duplicate SQL table name '{name}'"
                    )));
                }
                let collection =
                    collection
                        .downcast_ref::<ArrowRecordBatch>()
                        .ok_or_else(|| {
                            SqlTransformError::message(format!(
                                "SQL table '{name}' has element type {}, expected \
                                 ArrowRecordBatch",
                                collection.element_type_name()
                            ))
                        })?;
                let schema = collection.arrow_schema().ok_or_else(|| {
                    SqlTransformError::message(format!(
                        "SQL table '{name}' has no Arrow schema; call with_arrow_schema first"
                    ))
                })?;
                inputs.insert(
                    name.to_string(),
                    BeamPCollectionTable {
                        upstream: collection.clone(),
                        schema,
                    },
                );
                Ok((names, inputs))
            },
        )
        .map(|(_, inputs)| inputs)
}

fn resolve_single_input_table(
    input: &PCollection<ArrowRecordBatch>,
) -> Result<BTreeMap<String, BeamPCollectionTable>, SqlTransformError> {
    let schema = input.arrow_schema().ok_or_else(|| {
        SqlTransformError::message(
            "PCOLLECTION has no Arrow schema; call with_arrow_schema before SqlTransform",
        )
    })?;
    Ok(BTreeMap::from([(
        PCOLLECTION_NAME.to_string(),
        BeamPCollectionTable {
            upstream: input.clone(),
            schema,
        },
    )]))
}

/// Error returned while SQL parsing, planning, translation, or expansion fails.
#[derive(Debug, thiserror::Error)]
pub(crate) enum SqlTransformError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    DataFusion(#[from] DataFusionError),
    #[error("{0}")]
    Beam(Error),
}

impl SqlTransformError {
    pub(crate) fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl From<Error> for SqlTransformError {
    fn from(error: Error) -> Self {
        Self::Beam(error)
    }
}
