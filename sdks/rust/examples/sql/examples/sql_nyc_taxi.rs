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

//! Joins real NYC TLC trip Parquet data with a license-to-service table and filters it.
//!
//! This reads the same public source as the `nyc_taxi` example, converts the rows to Arrow,
//! and runs a SQL join and filter. It writes the matching trips to a text file.
//!
//! ```text
//! cargo run -p sql --example sql_nyc_taxi
//! cat /tmp/sql_nyc_taxi.txt
//! ```

use std::sync::Arc;

use arrow_array::{Array, Float64Array, RecordBatch, StringArray};
use arrow_io::{
    ArrowRecordBatch, ArrowRecordBatchPCollectionExt, beam_rows_to_record_batch,
    beam_to_arrow_schema,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use beam::io::parquet::parquetio;
use beam::prelude::*;
use sql::SqlTransform;

const INPUT: &str = "gs://apache-beam-samples/nyc_trip/parquet/fhvhv_tripdata_2023-02.parquet";
const OUTPUT: &str = "/tmp/sql_nyc_taxi.txt";
const BATCH_SIZE: usize = 4096;

const QUERY: &str = r#"
    SELECT services.service, trips.hvfhs_license_num, trips.trip_miles,
           trips.base_passenger_fare
    FROM trips
    JOIN services ON trips.hvfhs_license_num = services.license
    WHERE services.service = 'Uber' AND trips.trip_miles > 10
"#;

/// Columns of the TLC schema that this example reads.
#[derive(Clone, Debug, BeamRow)]
struct Trip {
    hvfhs_license_num: Option<String>,
    trip_miles: Option<f64>,
    base_passenger_fare: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    beam::logging::init_logging();
    let pipeline = Pipeline::new();

    // The Arrow schema is known from the derived Beam schema of `Trip`.
    let trip_schema: SchemaRef =
        Arc::new(beam_to_arrow_schema(Trip::beam_schema()).expect("trip Arrow schema"));

    let trips = pipeline
        .apply("ReadTrips", parquetio::Read::<Trip>::new(INPUT))
        .apply("BatchTrips", BatchElements::new(1, BATCH_SIZE))
        .apply(
            "TripsToArrow",
            Map::new(|rows: Vec<Trip>| {
                ArrowRecordBatch::new(
                    beam_rows_to_record_batch(&rows).expect("trip rows match the Arrow schema"),
                )
            }),
        )
        .with_arrow_schema(&trip_schema);

    let (services_schema, services_batch) = services();
    let services = pipeline
        .apply("CreateServices", Create::new(vec![services_batch]))
        .with_arrow_schema(&services_schema);

    let input = PCollectionTuple::of("trips", &trips).and("services", &services);

    input
        .apply("SqlJoinFilter", SqlTransform::query(QUERY))
        .apply(
            "FormatRows",
            FlatMap::new(|batch: ArrowRecordBatch| {
                let service = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("service");
                let license = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("license");
                let miles = batch
                    .column(2)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("miles");
                let fares = batch
                    .column(3)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("fare");
                (0..batch.num_rows())
                    .map(|index| {
                        let fare = if fares.is_null(index) {
                            "null".to_string()
                        } else {
                            format!("{:.2}", fares.value(index))
                        };
                        format!(
                            "{}: license {}, {:.2} miles, fare {}",
                            service.value(index),
                            license.value(index),
                            miles.value(index),
                            fare
                        )
                    })
                    .collect::<Vec<_>>()
            }),
        )
        .apply("WriteLines", textio::Write::new(OUTPUT).without_sharding());

    pipeline.run().await?;
    println!("Wrote {OUTPUT}");
    Ok(())
}

/// License-to-service lookup: `(license, service)`.
fn services() -> (SchemaRef, ArrowRecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("license", DataType::Utf8, false),
        Field::new("service", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![
                "HV0002", "HV0003", "HV0004", "HV0005",
            ])),
            Arc::new(StringArray::from(vec!["Juno", "Uber", "Via", "Lyft"])),
        ],
    )
    .expect("services batch");
    (schema, ArrowRecordBatch::new(batch))
}
