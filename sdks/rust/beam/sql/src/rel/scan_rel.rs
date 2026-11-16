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
use beam::values::PCollection;
use beam::{Error, Result};

use crate::rel::{BeamRelNode, SqlRelContext};

/// Reads one registered Arrow table.
pub(crate) struct BeamTableScanRel {
    table: String,
}

impl BeamTableScanRel {
    pub(crate) fn new(table: String) -> Self {
        Self { table }
    }
}

impl BeamRelNode for BeamTableScanRel {
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
        Ok(table.upstream.clone().with_arrow_schema(&table.schema))
    }
}
