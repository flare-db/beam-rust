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

use arrow_io::{ArrowRecordBatch, ArrowRecordBatchPCollectionExt};
use arrow_schema::{DataType, Field, Schema};
use beam::pipeline::Pipeline;
use beam::transforms::Create;
use beam::values::PCollectionTuple;
use sql::SqlTransform;

#[test]
fn planner_pushes_join_filters_and_scan_projections_into_beam_graph() {
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
    let input = PCollectionTuple::of("trips", &trips).and("services", &services);

    let output = input.apply("Sql", SqlTransform::query(QUERY));
    let output_schema = output.arrow_schema().expect("SQL output schema");
    assert_eq!(
        output_schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        ["service", "trip_miles"]
    );

    let proto = pipeline.to_proto();
    let components = proto.components.expect("components");
    let names = components
        .transforms
        .values()
        .map(|transform| transform.unique_name.as_str())
        .collect::<Vec<_>>();

    assert!(
        names
            .iter()
            .filter(|name| name.contains("/Filter_"))
            .count()
            >= 2,
        "optimizer should push one filter to each join input: {names:?}"
    );
    assert_eq!(
        names
            .iter()
            .filter(|name| name.contains("/Project_"))
            .count(),
        2,
        "each table scan should materialize its projected columns: {names:?}"
    );
    assert!(
        names.iter().any(|name| name.ends_with("/HashJoin")),
        "join should be translated to a Beam hash join: {names:?}"
    );
    assert!(
        names.iter().all(|name| !name.contains("/Projection_")),
        "the root projection should remain fused with its relational node: {names:?}"
    );
}
