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

//! In-memory Arrow tables shared by the SQL examples.

use std::sync::Arc;

use arrow_array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_io::ArrowRecordBatch;
use arrow_schema::{DataType, Field, Schema, SchemaRef};

/// Customers: `(id, name, country)`.
pub fn customers() -> (SchemaRef, ArrowRecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("country", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                "Foo", "Bar", "Baz", "Grault", "Qux",
            ])),
            Arc::new(StringArray::from(vec![
                "Wonderland",
                "Super Kingdom",
                "Wonderland",
                "Wonderland",
                "Super Kingdom",
            ])),
        ],
    )
    .expect("customers batch");
    (schema, ArrowRecordBatch::new(batch))
}

/// Orders: `(id, customer_id)`.
pub fn orders() -> (SchemaRef, ArrowRecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("customer_id", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9])),
            Arc::new(Int64Array::from(vec![5, 2, 1, 3, 1, 5, 4, 4, 1])),
        ],
    )
    .expect("orders batch");
    (schema, ArrowRecordBatch::new(batch))
}

/// Numbers: `(c1, c2, c3)`, as in the Java `BeamSqlExample`.
pub fn numbers() -> (SchemaRef, ArrowRecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("c1", DataType::Int64, false),
        Field::new("c2", DataType::Utf8, false),
        Field::new("c3", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["row", "row", "row"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        ],
    )
    .expect("numbers batch");
    (schema, ArrowRecordBatch::new(batch))
}
