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

//! Filter and projection translation for Beam SQL.

use std::sync::Arc;

use arrow_array::{Array, BooleanArray, RecordBatch, RecordBatchOptions};
use arrow_io::{ArrowRecordBatch, ArrowRecordBatchPCollectionExt};
use arrow_schema::SchemaRef;
use arrow_select::filter::filter_record_batch;
use beam::transforms::{DoFn, ParDo, ProcessContext};
use beam::values::PCollection;
use beam::{Error, Result};
use datafusion::common::DFSchemaRef;
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::PhysicalExpr;

use crate::rel::{BeamRelNode, BeamSqlRelUtils, SqlRelContext, indent_plan};

/// Filter and projection node for Beam SQL.
///
/// A single Beam DoFn applies an optional predicate and the output projection to one Arrow Batch.
pub(crate) struct BeamFilterRel {
    input: Box<dyn BeamRelNode>,
    input_schema: DFSchemaRef,
    predicate: Option<Expr>,
    projection: Option<Vec<Expr>>,
    output_schema: SchemaRef,
}

impl BeamFilterRel {
    /// Creates a filter with an optional output projection.
    pub(crate) fn new(
        input: Box<dyn BeamRelNode>,
        input_schema: DFSchemaRef,
        predicate: Option<Expr>,
        projection: Option<Vec<Expr>>,
        output_schema: SchemaRef,
    ) -> Self {
        Self {
            input,
            input_schema,
            predicate,
            projection,
            output_schema,
        }
    }
}

impl BeamRelNode for BeamFilterRel {
    /// Compiles the expressions and expands one filter transform.
    fn build_ptransform(
        &self,
        context: &SqlRelContext<'_>,
    ) -> Result<PCollection<ArrowRecordBatch>> {
        let input = self.input.build_ptransform(context)?;
        let predicate = self
            .predicate
            .as_ref()
            .map(|predicate| BeamSqlRelUtils::compile_expr(predicate, &self.input_schema))
            .transpose()?;
        let projection = self
            .projection
            .as_ref()
            .map(|expressions| {
                expressions
                    .iter()
                    .map(|expression| BeamSqlRelUtils::compile_expr(expression, &self.input_schema))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;

        let output = input.apply(
            context.transform_name("Filter"),
            ParDo::new(FilterFn {
                predicate,
                projection,
                output_schema: Arc::clone(&self.output_schema),
            }),
        );
        Ok(output.with_arrow_schema(&self.output_schema))
    }

    /// Formats the predicate, projection, and child plan for logging.
    fn explain(&self) -> String {
        let predicate = self
            .predicate
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default();
        let projection = self
            .projection
            .as_ref()
            .map(|expressions| {
                expressions
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        format!(
            "BeamFilterRel(predicate=[{predicate}], projection=[{projection}])\n{}",
            indent_plan(&self.input.explain())
        )
    }
}

/// Applies the compiled predicate and projection to each Arrow batch.
#[derive(Clone)]
struct FilterFn {
    predicate: Option<Arc<dyn PhysicalExpr>>,
    projection: Option<Vec<Arc<dyn PhysicalExpr>>>,
    output_schema: SchemaRef,
}

impl DoFn for FilterFn {
    type In = ArrowRecordBatch;
    type Out = ArrowRecordBatch;

    /// Filters one batch, then emits the projected rows when any remain.
    fn process_element(
        &mut self,
        batch: Self::In,
        ctx: &mut ProcessContext<'_, Self::Out>,
    ) -> Result {
        let filtered = if let Some(predicate) = &self.predicate {
            let values = BeamSqlRelUtils::evaluate_array(predicate, &batch)?;
            let values = values
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    Error::msg(format!(
                        "BeamFilterRel predicate returned {}, expected Boolean",
                        values.data_type()
                    ))
                })?;
            filter_record_batch(&batch, values).map_err(Error::from)?
        } else {
            batch.into_inner()
        };
        if filtered.num_rows() == 0 {
            return Ok(());
        }

        let output = if let Some(projection) = &self.projection {
            let columns = projection
                .iter()
                .map(|expression| BeamSqlRelUtils::evaluate_array(expression, &filtered))
                .collect::<Result<Vec<_>>>()?;
            try_record_batch(
                Arc::clone(&self.output_schema),
                columns,
                filtered.num_rows(),
            )?
        } else {
            filtered
        };
        ctx.emit(ArrowRecordBatch::new(output))
    }
}

/// Builds a projected batch with an explicit row count.
fn try_record_batch(
    output_schema: SchemaRef,
    columns: Vec<arrow_array::ArrayRef>,
    row_count: usize,
) -> Result<RecordBatch> {
    let options = RecordBatchOptions::new().with_row_count(Some(row_count));
    RecordBatch::try_new_with_options(output_schema, columns, &options).map_err(Error::from)
}
