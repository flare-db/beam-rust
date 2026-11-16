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

//! Filters and projects an in-memory Arrow table with `SqlTransform`.
//!
//! This ports Case 1 of Java's `BeamSqlExample`: `SELECT ... WHERE c1 > 1`.
//!
//! ```text
//! cargo run -p sql --example sql_filter
//! cat /tmp/sql_filter.txt
//! ```

use arrow_array::{Array, Float64Array, Int64Array, StringArray};
use arrow_io::{ArrowRecordBatch, ArrowRecordBatchPCollectionExt};
use beam::prelude::*;
use sql::SqlTransform;
use sql_examples::numbers;

const OUTPUT: &str = "/tmp/sql_filter.txt";

const QUERY: &str = "SELECT c1, c2, c3 FROM PCOLLECTION WHERE c1 > 1";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    beam::logging::init_logging();
    let pipeline = Pipeline::new();

    let (schema, batch) = numbers();
    let input = pipeline
        .apply("CreateNumbers", Create::new(vec![batch]))
        .with_arrow_schema(&schema);

    input
        .apply("SqlFilter", SqlTransform::query(QUERY))
        .apply(
            "FormatRows",
            FlatMap::new(|batch: ArrowRecordBatch| {
                let c1 = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("c1");
                let c2 = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("c2");
                let c3 = batch
                    .column(2)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("c3");
                (0..batch.num_rows())
                    .map(|index| {
                        format!(
                            "c1: {}, c2: {}, c3: {}",
                            c1.value(index),
                            c2.value(index),
                            c3.value(index)
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
