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

use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_io::{ArrowRecordBatch, ArrowRecordBatchPCollectionExt};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use beam::transforms::{Create, FlatMap};
use beam::values::PCollectionTuple;
use sql::SqlTransform;
use testing::{TestPipeline, passert};

fn left_batch() -> (SchemaRef, ArrowRecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 1])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .expect("left batch");
    (schema, ArrowRecordBatch::new(batch))
}

fn right_batch() -> (SchemaRef, ArrowRecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 3, 1])),
            Arc::new(Int64Array::from(vec![10, 30, 11])),
        ],
    )
    .expect("right batch");
    (schema, ArrowRecordBatch::new(batch))
}

#[tokio::test]
async fn join_transform_matches_rows_across_partitions() {
    let pipeline = TestPipeline::new();
    let (left_schema, left_batch) = left_batch();
    let (right_schema, right_batch) = right_batch();
    let left = pipeline
        .apply("CreateLeft", Create::new(vec![left_batch]))
        .with_arrow_schema(&left_schema);
    let right = pipeline
        .apply("CreateRight", Create::new(vec![right_batch]))
        .with_arrow_schema(&right_schema);

    let output = PCollectionTuple::of("left", &left)
        .and("right", &right)
        .apply(
            "Join",
            SqlTransform::query(
                "SELECT l.id, l.name, r.amount \
                 FROM left l JOIN right r ON l.id = r.id",
            )
            .with_partition_size(2),
        );
    let rows = output.apply(
        "Collect",
        FlatMap::new(|batch: ArrowRecordBatch| {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("left id");
            let names = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("name");
            let amounts = batch
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("amount");
            (0..batch.num_rows())
                .map(|index| {
                    (
                        ids.value(index),
                        names.value(index).to_string(),
                        amounts.value(index),
                    )
                })
                .collect::<Vec<_>>()
        }),
    );
    passert::that("AssertJoinRows", &rows).contains_in_any_order([
        (1, "a".to_string(), 10),
        (1, "a".to_string(), 11),
        (1, "c".to_string(), 10),
        (1, "c".to_string(), 11),
    ]);

    pipeline.run().await.expect("pipeline and assertions");
}
