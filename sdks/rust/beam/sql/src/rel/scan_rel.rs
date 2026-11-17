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

//! Table scan translation for Beam SQL.

use arrow_io::{ArrowRecordBatch, ArrowRecordBatchPCollectionExt};
use arrow_schema::SchemaRef;
use beam::transforms::{DoFn, ParDo, ProcessContext};
use beam::values::PCollection;
use beam::{Error, Result};

use crate::rel::{BeamRelNode, SqlRelContext};

/// Reads one registered Arrow table and applies the scan projection.
pub(crate) struct BeamTableScanRel {
    table: String,
    projection: Option<Vec<usize>>,
    projected_schema: SchemaRef,
}

impl BeamTableScanRel {
    /// Creates a scan for a registered table with an optional column projection.
    pub(crate) fn new(
        table: String,
        projection: Option<Vec<usize>>,
        projected_schema: SchemaRef,
    ) -> Self {
        Self {
            table,
            projection,
            projected_schema,
        }
    }
}

impl BeamRelNode for BeamTableScanRel {
    /// Expands the table scan into a projection transform when needed.
    fn build_ptransform(
        &self,
        context: &SqlRelContext<'_>,
    ) -> Result<PCollection<ArrowRecordBatch>> {
        let table = context
            .tables
            .get(&self.table)
            .or_else(|| {
                context
                    .tables
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(&self.table))
                    .map(|(_, table)| table)
            })
            .ok_or_else(|| Error::msg(format!("SQL table '{}' is not registered", self.table)))?;
        let input = table.upstream.clone().with_arrow_schema(&table.schema);
        let output = if let Some(projection) = &self.projection {
            input.apply(
                context.transform_name("Project"),
                ParDo::new(ProjectFn {
                    projection: projection.clone(),
                }),
            )
        } else {
            input
        };
        Ok(output.with_arrow_schema(&self.projected_schema))
    }

    /// Formats the table name and projected fields for logging.
    fn explain(&self) -> String {
        let projection = self
            .projection
            .as_ref()
            .map(|_| {
                let fields = self
                    .projected_schema
                    .fields()
                    .iter()
                    .map(|field| field.name().as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(", projection=[{fields}]")
            })
            .unwrap_or_default();
        format!("BeamTableScanRel(table=[[{}]]{projection})", self.table)
    }
}

/// Selects the DataFusion scan-projected columns from each Arrow batch.
#[derive(Clone)]
struct ProjectFn {
    projection: Vec<usize>,
}

impl DoFn for ProjectFn {
    type In = ArrowRecordBatch;
    type Out = ArrowRecordBatch;

    /// Selects the configured columns from one Arrow batch.
    fn process_element(
        &mut self,
        batch: Self::In,
        ctx: &mut ProcessContext<'_, Self::Out>,
    ) -> Result {
        let projected = batch.into_inner().project(&self.projection)?;
        ctx.emit(ArrowRecordBatch::new(projected))
    }
}
