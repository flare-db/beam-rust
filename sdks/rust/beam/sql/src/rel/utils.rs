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

//! Utilities for Beam relational nodes.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_io::ArrowRecordBatch;
use beam::values::PCollection;
use beam::{Error, Result};
use datafusion::common::DFSchemaRef;
use datafusion::logical_expr::Expr;
use datafusion::logical_expr_common::columnar_value::ColumnarValue;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::prelude::SessionContext;

use crate::rel::{BeamRelNode, SqlRelContext};
use crate::sql_transform::SqlTransformError;

/// Utilities shared by Beam relational nodes.
pub(crate) struct BeamSqlRelUtils;

impl BeamSqlRelUtils {
    /// Expands a translated rel tree through its common PCollection boundary.
    pub(crate) fn to_pcollection(
        node: &dyn BeamRelNode,
        context: &SqlRelContext<'_>,
    ) -> Result<PCollection<ArrowRecordBatch>, SqlTransformError> {
        node.build_ptransform(context)
            .map_err(SqlTransformError::from)
    }

    /// Compiles a DataFusion expression against the input schema.
    pub(crate) fn compile_expr(
        expression: &Expr,
        input_schema: &DFSchemaRef,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        SessionContext::new()
            .create_physical_expr(expression.clone(), input_schema)
            .map_err(Error::from)
    }

    /// Evaluates an expression as an array with the batch row count.
    pub(crate) fn evaluate_array(
        expression: &Arc<dyn PhysicalExpr>,
        batch: &RecordBatch,
    ) -> Result<ArrayRef> {
        match expression.evaluate(batch).map_err(Error::from)? {
            ColumnarValue::Array(array) => Ok(array),
            ColumnarValue::Scalar(scalar) => scalar
                .to_array_of_size(batch.num_rows())
                .map_err(Error::from),
        }
    }
}
