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

//! Beam relational nodes translated from DataFusion's logical plan.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_io::ArrowRecordBatch;
use arrow_schema::SchemaRef;
use beam::Result;
use beam::values::PCollection;

mod filter_rel;
mod join_rel;
mod scan_rel;
mod utils;

pub(crate) use filter_rel::BeamFilterRel;
pub(crate) use join_rel::BeamJoinRel;
pub(crate) use scan_rel::BeamTableScanRel;
pub(crate) use utils::BeamSqlRelUtils;

/// A PCollection-backed SQL table: one Arrow PCollection and its schema.
///
/// The table name is the key of the map that holds it.
#[derive(Clone)]
pub(crate) struct BeamPCollectionTable {
    pub upstream: PCollection<ArrowRecordBatch>,
    pub schema: SchemaRef,
}

/// Inputs available while expanding a translated SQL plan.
pub(crate) struct SqlRelContext<'a> {
    pub tables: &'a BTreeMap<String, BeamPCollectionTable>,
    next_transform_id: AtomicUsize,
}

impl<'a> SqlRelContext<'a> {
    pub(crate) fn new(tables: &'a BTreeMap<String, BeamPCollectionTable>) -> Self {
        Self {
            tables,
            next_transform_id: AtomicUsize::new(1),
        }
    }

    pub(crate) fn transform_name(&self, kind: &str) -> String {
        let id = self.next_transform_id.fetch_add(1, Ordering::Relaxed);
        format!("{kind}_{id}")
    }
}

/// A relational node that can emit a Beam transform graph.
pub(crate) trait BeamRelNode: Send + Sync {
    /// Expands this node and its inputs into the Beam graph.
    fn build_ptransform(
        &self,
        context: &SqlRelContext<'_>,
    ) -> Result<PCollection<ArrowRecordBatch>>;

    /// Formats this node and its inputs as a tree for plan logging.
    fn explain(&self) -> String;
}

pub(crate) fn indent_plan(plan: &str) -> String {
    plan.lines()
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}
