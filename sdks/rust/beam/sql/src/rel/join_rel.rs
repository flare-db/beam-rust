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

use arrow_io::{ArrowRecordBatch, ArrowRecordBatchPCollectionExt};
use arrow_schema::SchemaRef;
use beam::values::{IsBounded, PCollection, PCollectionList};
use beam::{Error, Result};
use datafusion::common::{DFSchemaRef, NullEquality};
use datafusion::logical_expr::{Expr, JoinType};

use crate::rel::{BeamRelNode, BeamSqlRelUtils, SqlRelContext};
use crate::transform::join::Join;

/// An equality join of two Beam SQL inputs.
pub(crate) struct BeamJoinRel {
    left: Box<dyn BeamRelNode>,
    right: Box<dyn BeamRelNode>,
    left_schema: DFSchemaRef,
    right_schema: DFSchemaRef,
    on: Vec<(Expr, Expr)>,
    join_type: JoinType,
    output_schema: SchemaRef,
    partition_size: usize,
}

impl BeamJoinRel {
    #[expect(
        clippy::too_many_arguments,
        reason = "the constructor maps one-to-one to DataFusion Join metadata"
    )]
    pub(crate) fn new(
        left: Box<dyn BeamRelNode>,
        right: Box<dyn BeamRelNode>,
        left_schema: DFSchemaRef,
        right_schema: DFSchemaRef,
        on: Vec<(Expr, Expr)>,
        filter: Option<Expr>,
        join_type: JoinType,
        output_schema: SchemaRef,
        null_equality: NullEquality,
        partition_size: usize,
    ) -> Result<Self> {
        match join_type {
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {}
            other => {
                return Err(Error::msg(format!(
                    "BeamJoinRel does not support {other:?} joins"
                )));
            }
        }
        if on.is_empty() {
            return Err(Error::msg(
                "BeamJoinRel requires at least one equality join key",
            ));
        }
        if filter.is_some() {
            return Err(Error::msg("BeamJoinRel supports equality ON clauses only"));
        }
        if null_equality != NullEquality::NullEqualsNothing {
            return Err(Error::msg(
                "BeamJoinRel supports SQL NULLs as non-matching join keys only",
            ));
        }
        Ok(Self {
            left,
            right,
            left_schema,
            right_schema,
            on,
            join_type,
            output_schema,
            partition_size,
        })
    }
}

impl BeamRelNode for BeamJoinRel {
    fn build_ptransform(
        &self,
        context: &SqlRelContext<'_>,
    ) -> Result<PCollection<ArrowRecordBatch>> {
        if self.partition_size == 0 {
            return Err(Error::msg("partition_size must be greater than zero"));
        }

        let left = self.left.build_ptransform(context)?;
        let right = self.right.build_ptransform(context)?;
        if left.is_bounded() != IsBounded::Bounded || right.is_bounded() != IsBounded::Bounded {
            return Err(Error::msg(
                "BeamJoinRel currently supports bounded inputs only",
            ));
        }
        if left.windowing_strategy_id() != right.windowing_strategy_id() {
            return Err(Error::msg(
                "BeamJoinRel requires compatible windowing strategies on both inputs",
            ));
        }

        let left_expressions = self
            .on
            .iter()
            .map(|(left, _)| BeamSqlRelUtils::compile_expr(left, &self.left_schema))
            .collect::<Result<Vec<_>>>()?;
        let right_expressions = self
            .on
            .iter()
            .map(|(_, right)| BeamSqlRelUtils::compile_expr(right, &self.right_schema))
            .collect::<Result<Vec<_>>>()?;

        let input = PCollectionList::of(left).and(right);
        let joined = input.apply(
            context.transform_name("HashJoin"),
            Join::new(left_expressions, right_expressions)
                .with_join_type(self.join_type)
                .with_num_partitions(self.partition_size),
        );
        Ok(joined.with_arrow_schema(&self.output_schema))
    }
}
