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

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::rel::{BeamFilterRel, BeamJoinRel, BeamTableScanRel};
use datafusion::catalog::empty::EmptyTable;
use datafusion::common::TableReference;
use datafusion::logical_expr::{BinaryExpr, Expr, LogicalPlan, Operator};
#[cfg(test)]
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;

use crate::rel::{BeamPCollectionTable, BeamRelNode};
use crate::sql_transform::SqlTransformError;

#[derive(Clone)]
pub struct DataFusionQueryPlanner {
    ctx: SessionContext,
    partition_size: usize,
}

impl DataFusionQueryPlanner {
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

    fn translate_plan(
        &self,
        plan: &LogicalPlan,
    ) -> Result<Box<dyn BeamRelNode>, SqlTransformError> {
        match plan {
            LogicalPlan::TableScan(scan) => Ok(Box::new(BeamTableScanRel::new(Self::table_name(
                &scan.table_name,
            )))),
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

    fn table_name(table: &TableReference) -> String {
        table.table().to_string()
    }

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

/// Plans `query` against schema-only providers for the Beam input tables.
#[cfg(test)]
pub(crate) fn plan_query(
    query: &str,
    tables: &BTreeMap<String, BeamPCollectionTable>,
) -> Result<LogicalPlan, SqlTransformError> {
    let context = SessionContext::new_with_config(SessionConfig::new());
    for (name, table) in tables {
        context.register_table(
            name.as_str(),
            Arc::new(EmptyTable::new(Arc::clone(&table.schema))),
        )?;
    }
    let state = context.state();
    let plan = futures::executor::block_on(state.create_logical_plan(query))?;
    state.optimize(&plan).map_err(SqlTransformError::from)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_io::{ArrowRecordBatch, ArrowRecordBatchPCollectionExt};
    use arrow_schema::{DataType, Field, Schema};
    use beam::pipeline::Pipeline;
    use beam::transforms::Create;

    use super::*;

    #[test]
    fn optimizer_pushes_join_filters_below_the_join() {
        const QUERY: &str = "SELECT services.service, trips.trip_miles
             FROM trips
             JOIN services ON trips.hvfhs_license_num = services.license
             WHERE services.service = 'Uber' AND trips.trip_miles > 10";

        let pipeline = Pipeline::new();
        let trips_schema = Arc::new(Schema::new(vec![
            Field::new("hvfhs_license_num", DataType::Utf8, true),
            Field::new("trip_miles", DataType::Float64, true),
            Field::new("base_passenger_fare", DataType::Float64, true),
        ]));
        let services_schema = Arc::new(Schema::new(vec![
            Field::new("license", DataType::Utf8, false),
            Field::new("service", DataType::Utf8, false),
        ]));
        let trips = pipeline
            .apply("CreateTrips", Create::new(Vec::<ArrowRecordBatch>::new()))
            .with_arrow_schema(&trips_schema);
        let services = pipeline
            .apply(
                "CreateServices",
                Create::new(Vec::<ArrowRecordBatch>::new()),
            )
            .with_arrow_schema(&services_schema);
        let tables = BTreeMap::from([
            (
                "trips".to_string(),
                BeamPCollectionTable {
                    upstream: trips,
                    schema: trips_schema,
                },
            ),
            (
                "services".to_string(),
                BeamPCollectionTable {
                    upstream: services,
                    schema: services_schema,
                },
            ),
        ]);

        let plan = plan_query(QUERY, &tables).expect("optimized plan");

        let LogicalPlan::Projection(projection) = plan else {
            panic!("expected projection, got {plan}");
        };
        let LogicalPlan::Join(join) = projection.input.as_ref() else {
            panic!("expected join below projection, got {}", projection.input);
        };
        assert!(
            matches!(join.left.as_ref(), LogicalPlan::Filter(_)),
            "trip filter should be below the join: {}",
            projection.input
        );
        assert!(
            matches!(join.right.as_ref(), LogicalPlan::Filter(_)),
            "service filter should be below the join: {}",
            projection.input
        );

        let planner = DataFusionQueryPlanner::new(&tables, 2).expect("planner");
        let beam_plan = futures::executor::block_on(planner.build_plan(QUERY)).expect("Beam plan");
        let explain = beam_plan.explain();
        assert!(
            explain.starts_with("BeamFilterRel("),
            "projection should be rendered at the root: {explain}"
        );
        assert!(
            explain.contains("BeamJoinRel(condition=["),
            "join should be rendered below the projection: {explain}"
        );
        assert!(
            explain.contains("BeamTableScanRel(table=[[trips]])")
                && explain.contains("BeamTableScanRel(table=[[services]])"),
            "both table scans should be rendered: {explain}"
        );
    }
}
