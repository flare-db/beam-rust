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
//! DataFusion query planning for Beam SQL.
//!
//! The planner registers input table schemas as empty DataFusion tables. It uses those schemas to
//! resolve names and plan each query. It then translates supported logical nodes into Beam
//! relational nodes.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::rel::{BeamFilterRel, BeamJoinRel, BeamTableScanRel};
use datafusion::catalog::empty::EmptyTable;
use datafusion::common::TableReference;
use datafusion::logical_expr::{BinaryExpr, Expr, LogicalPlan, Operator};
use datafusion::prelude::SessionContext;

use crate::rel::{BeamPCollectionTable, BeamRelNode};
use crate::sql_transform::SqlTransformError;

/// Plans SQL queries against registered Arrow table schemas.
#[derive(Clone)]
pub struct DataFusionQueryPlanner {
    ctx: SessionContext,
    partition_size: usize,
}

impl DataFusionQueryPlanner {
    /// Registers the table names and schemas for query planning.
    pub fn new(
        tables: &BTreeMap<String, BeamPCollectionTable>,
        partition_size: usize,
    ) -> Result<Self, SqlTransformError> {
        let ctx = SessionContext::new();
        for (name, table) in tables {
            ctx.register_table(
                name.as_str(),
                Arc::new(EmptyTable::new(Arc::clone(&table.schema))),
            )?;
        }
        Ok(Self {
            ctx,
            partition_size,
        })
    }

    /// Parses and optimizes a query, then translates its logical plan for BeamRelNode.
    pub async fn build_plan(
        &self,
        sql_statement: &str,
    ) -> Result<Box<dyn BeamRelNode>, SqlTransformError> {
        tracing::info!("SQL:\n{sql_statement}");
        let state = self.ctx.state();
        let plan = state.create_logical_plan(sql_statement).await?;
        let plan = state.optimize(&plan)?;
        tracing::info!("SQLPlan>\n{plan}");

        let beam_plan = self.translate_plan(&plan)?;
        tracing::info!("BEAMPlan>\n{}", beam_plan.explain());
        Ok(beam_plan)
    }

    /// Translates supported DataFusion logical nodes into Beam relational nodes.
    fn translate_plan(
        &self,
        plan: &LogicalPlan,
    ) -> Result<Box<dyn BeamRelNode>, SqlTransformError> {
        match plan {
            LogicalPlan::TableScan(scan) => Ok(Box::new(BeamTableScanRel::new(
                Self::table_name(&scan.table_name),
                scan.projection.clone(),
                Arc::new(scan.projected_schema.as_arrow().clone()),
            ))),
            LogicalPlan::SubqueryAlias(alias) => self.translate_plan(&alias.input),
            LogicalPlan::Filter(filter) => {
                let input = self.translate_plan(&filter.input)?;
                Ok(Box::new(BeamFilterRel::new(
                    input,
                    Arc::clone(filter.input.schema()),
                    Some(filter.predicate.clone()),
                    None,
                    Arc::new(filter.input.schema().as_arrow().clone()),
                )))
            }
            LogicalPlan::Projection(projection) => {
                let (base, predicate) = match projection.input.as_ref() {
                    LogicalPlan::Filter(filter) => {
                        (filter.input.as_ref(), Some(filter.predicate.clone()))
                    }
                    input => (input, None),
                };
                let input = self.translate_plan(base)?;
                Ok(Box::new(BeamFilterRel::new(
                    input,
                    Arc::clone(base.schema()),
                    predicate,
                    Some(projection.expr.clone()),
                    Arc::new(projection.schema.as_arrow().clone()),
                )))
            }
            LogicalPlan::Join(join) => {
                let left = self.translate_plan(&join.left)?;
                let right = self.translate_plan(&join.right)?;
                let (on, filter) = if join.on.is_empty() {
                    (
                        join.filter
                            .as_ref()
                            .map(Self::extract_equality_join_keys)
                            .transpose()?
                            .unwrap_or_default(),
                        None,
                    )
                } else {
                    (join.on.clone(), join.filter.clone())
                };
                let join = BeamJoinRel::new(
                    left,
                    right,
                    Arc::clone(join.left.schema()),
                    Arc::clone(join.right.schema()),
                    on,
                    filter,
                    join.join_type,
                    Arc::new(join.schema.as_arrow().clone()),
                    join.null_equality,
                    self.partition_size,
                )?;
                Ok(Box::new(join))
            }
            other => Err(SqlTransformError::message(format!(
                "Beam SQL does not support logical plan node {}",
                other.display()
            ))),
        }
    }

    /// Returns the unqualified table name from a DataFusion table reference.
    fn table_name(table: &TableReference) -> String {
        table.table().to_string()
    }

    /// Splits an equality condition into left and right key expressions.
    fn extract_equality_join_keys(
        expression: &Expr,
    ) -> Result<Vec<(Expr, Expr)>, SqlTransformError> {
        if let Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::And,
            right,
        }) = expression
        {
            let mut keys = Self::extract_equality_join_keys(left)?;
            keys.extend(Self::extract_equality_join_keys(right)?);
            return Ok(keys);
        }
        if let Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Eq,
            right,
        }) = expression
        {
            return Ok(vec![((**left).clone(), (**right).clone())]);
        }
        Err(SqlTransformError::message(format!(
            "BeamJoinRel supports equality join conditions only; found {expression}"
        )))
    }
}
