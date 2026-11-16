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

//! Joins two in-memory Arrow tables with `SqlTransform` and writes the result to a file.
//!
//! This ports the join of Java's `BeamSqlPojoExample`: customers buy orders, and the query
//! lists the orders of one customer.
//!
//! ```text
//! cargo run -p sql --example sql_join
//! cat /tmp/sql_join.txt
//! ```

use arrow_array::{Array, Int64Array, StringArray};
use arrow_io::{ArrowRecordBatch, ArrowRecordBatchPCollectionExt};
use beam::prelude::*;
use sql::SqlTransform;
use sql_examples::{customers, orders};

const OUTPUT: &str = "/tmp/sql_join.txt";

const QUERY: &str = r#"
    SELECT customers.name, orders.id
    FROM orders
    JOIN customers ON orders.customer_id = customers.id
    WHERE customers.name = 'Grault'
"#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let pipeline = Pipeline::new();

    let (customers_schema, customers_batch) = customers();
    let (orders_schema, orders_batch) = orders();

    let customers = pipeline
        .apply("CreateCustomers", Create::new(vec![customers_batch]))
        .with_arrow_schema(&customers_schema);
    let orders = pipeline
        .apply("CreateOrders", Create::new(vec![orders_batch]))
        .with_arrow_schema(&orders_schema);

    // SQL names the tables from the tuple tags.
    let input = PCollectionTuple::of("customers", &customers).and("orders", &orders);

    input
        .apply("SqlJoin", SqlTransform::query(QUERY))
        .apply(
            "FormatRows",
            FlatMap::new(|batch: ArrowRecordBatch| {
                let names = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("customer name");
                let order_ids = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("order id");
                (0..batch.num_rows())
                    .map(|index| {
                        format!(
                            "{}: order id:{}",
                            names.value(index),
                            order_ids.value(index)
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
