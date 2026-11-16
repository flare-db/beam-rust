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
use beam::coders::DefaultCoder;
use beam::pipeline::Pipeline;
use beam::transforms::{Create, FlatMap};
use beam::values::{IsBounded, PCollectionTuple};
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
async fn sql_inner_join_uses_arrow_partitioning_and_datafusion() {
    let pipeline = TestPipeline::new();
    let (left_schema, left_batch) = left_batch();
    let (right_schema, right_batch) = right_batch();
    let left = pipeline
        .apply("CreateLeft", Create::new(vec![left_batch]))
        .with_arrow_schema(&left_schema);
    let right = pipeline
        .apply("CreateRight", Create::new(vec![right_batch]))
        .with_arrow_schema(&right_schema);

    let input = PCollectionTuple::of("left", &left).and("right", &right);
    let output = input.apply(
        "Sql",
        SqlTransform::query(
            "SELECT l.id, l.name, r.amount \
             FROM left l JOIN right r ON l.id = r.id",
        )
        .with_partition_size(2),
    );
    let output_schema = output.arrow_schema().expect("SQL output schema");
    assert_eq!(
        output_schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        ["id", "name", "amount"]
    );
    let rows = output.apply(
        "Collect",
        FlatMap::new(|batch: ArrowRecordBatch| {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id");
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
    passert::that("AssertSqlRows", &rows).contains_in_any_order([
        (1, "a".to_string(), 10),
        (1, "a".to_string(), 11),
        (1, "c".to_string(), 10),
        (1, "c".to_string(), 11),
    ]);

    pipeline.run().await.expect("pipeline and assertions");
}

fn nullable_left_batch() -> (SchemaRef, ArrowRecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![Some(1), Some(2), None])),
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                Some("null-left"),
            ])),
        ],
    )
    .expect("nullable left batch");
    (schema, ArrowRecordBatch::new(batch))
}

fn nullable_right_batch() -> (SchemaRef, ArrowRecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("amount", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![Some(1), Some(3), None])),
            Arc::new(Int64Array::from(vec![Some(10), Some(30), Some(99)])),
        ],
    )
    .expect("nullable right batch");
    (schema, ArrowRecordBatch::new(batch))
}

#[tokio::test]
async fn sql_full_join_preserves_outer_and_null_rows() {
    let pipeline = TestPipeline::new();
    let (left_schema, left_batch) = nullable_left_batch();
    let (right_schema, right_batch) = nullable_right_batch();
    let left = pipeline
        .apply("CreateLeft", Create::new(vec![left_batch]))
        .with_arrow_schema(&left_schema);
    let right = pipeline
        .apply("CreateRight", Create::new(vec![right_batch]))
        .with_arrow_schema(&right_schema);
    let input = PCollectionTuple::of("left", &left).and("right", &right);
    let output = input.apply(
        "Sql",
        SqlTransform::query(
            "SELECT l.id AS lid, r.id AS rid, l.name, r.amount \
             FROM left l FULL JOIN right r ON l.id = r.id",
        ),
    );
    let rows = output.apply(
        "Collect",
        FlatMap::new(|batch: ArrowRecordBatch| {
            let lids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("lid");
            let rids = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("rid");
            let names = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("name");
            let amounts = batch
                .column(3)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("amount");
            (0..batch.num_rows())
                .map(|index| {
                    (
                        lids.is_valid(index).then(|| lids.value(index)),
                        rids.is_valid(index).then(|| rids.value(index)),
                        names
                            .is_valid(index)
                            .then(|| names.value(index).to_string()),
                        amounts.is_valid(index).then(|| amounts.value(index)),
                    )
                })
                .collect::<Vec<_>>()
        }),
    );
    passert::that("AssertSqlRows", &rows).contains_in_any_order([
        (Some(1), Some(1), Some("a".to_string()), Some(10)),
        (Some(2), None, Some("b".to_string()), None),
        (None, Some(3), None, Some(30)),
        (None, None, Some("null-left".to_string()), None),
        (None, None, None, Some(99)),
    ]);

    pipeline.run().await.expect("pipeline and assertions");
}

#[tokio::test]
async fn sql_where_filter_and_projection_are_vectorized() {
    let pipeline = TestPipeline::new();
    let (left_schema, left_batch) = left_batch();
    let (right_schema, right_batch) = right_batch();
    let left = pipeline
        .apply("CreateLeft", Create::new(vec![left_batch]))
        .with_arrow_schema(&left_schema);
    let right = pipeline
        .apply("CreateRight", Create::new(vec![right_batch]))
        .with_arrow_schema(&right_schema);
    let input = PCollectionTuple::of("left", &left).and("right", &right);
    let output = input.apply(
        "Sql",
        SqlTransform::query(
            "SELECT l.name AS label, r.amount + 1 AS adjusted \
             FROM left l JOIN right r ON l.id = r.id \
             WHERE r.amount > 10",
        ),
    );
    let rows = output.apply(
        "Collect",
        FlatMap::new(|batch: ArrowRecordBatch| {
            let labels = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("label");
            let adjusted = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("adjusted");
            (0..batch.num_rows())
                .map(|index| (labels.value(index).to_string(), adjusted.value(index)))
                .collect::<Vec<_>>()
        }),
    );
    passert::that("AssertSqlRows", &rows)
        .contains_in_any_order([("a".to_string(), 12), ("c".to_string(), 12)]);

    pipeline.run().await.expect("pipeline and assertions");
}

#[tokio::test]
async fn sql_left_and_right_joins_preserve_their_outer_side() {
    let pipeline = TestPipeline::new();
    let (left_schema, left_batch) = left_batch();
    let (right_schema, right_batch) = right_batch();
    let left = pipeline
        .apply("CreateLeft", Create::new(vec![left_batch]))
        .with_arrow_schema(&left_schema);
    let right = pipeline
        .apply("CreateRight", Create::new(vec![right_batch]))
        .with_arrow_schema(&right_schema);
    let input = PCollectionTuple::of("left", &left).and("right", &right);

    let left_output = input.apply(
        "SqlLeft",
        SqlTransform::query(
            "SELECT l.id, l.name, r.amount \
             FROM left l LEFT JOIN right r ON l.id = r.id",
        ),
    );
    let left_rows = left_output.apply(
        "CollectLeft",
        FlatMap::new(|batch: ArrowRecordBatch| {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id");
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
                        amounts.is_valid(index).then(|| amounts.value(index)),
                    )
                })
                .collect::<Vec<_>>()
        }),
    );
    passert::that("AssertLeftRows", &left_rows).contains_in_any_order([
        (1, "a".to_string(), Some(10)),
        (1, "a".to_string(), Some(11)),
        (2, "b".to_string(), None),
        (1, "c".to_string(), Some(10)),
        (1, "c".to_string(), Some(11)),
    ]);

    let right_output = input.apply(
        "SqlRight",
        SqlTransform::query(
            "SELECT r.id, l.name, r.amount \
             FROM left l RIGHT JOIN right r ON l.id = r.id",
        ),
    );
    let right_rows = right_output.apply(
        "CollectRight",
        FlatMap::new(|batch: ArrowRecordBatch| {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id");
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
                        names
                            .is_valid(index)
                            .then(|| names.value(index).to_string()),
                        amounts.value(index),
                    )
                })
                .collect::<Vec<_>>()
        }),
    );
    passert::that("AssertRightRows", &right_rows).contains_in_any_order([
        (1, Some("a".to_string()), 10),
        (1, Some("a".to_string()), 11),
        (1, Some("c".to_string()), 10),
        (1, Some("c".to_string()), 11),
        (3, None, 30),
    ]);

    pipeline.run().await.expect("pipeline and assertions");
}

#[test]
fn sql_expansion_partitions_shuffles_and_hash_joins() {
    let pipeline = Pipeline::new();
    let (left_schema, left_batch) = left_batch();
    let (right_schema, right_batch) = right_batch();
    let left = pipeline
        .apply("CreateLeft", Create::new(vec![left_batch]))
        .with_arrow_schema(&left_schema);
    let right = pipeline
        .apply("CreateRight", Create::new(vec![right_batch]))
        .with_arrow_schema(&right_schema);
    let input = PCollectionTuple::of("left", &left).and("right", &right);
    let _output = input.apply(
        "Sql",
        SqlTransform::query(
            "SELECT l.name, r.amount + 1 AS adjusted \
             FROM left l JOIN right r ON l.id = r.id \
             WHERE r.amount > 10",
        ),
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
            .any(|name| name.contains("/Partition left side")),
        "missing left hash partition: {names:?}"
    );
    assert!(
        names
            .iter()
            .any(|name| name.contains("/Partition right side")),
        "missing right hash partition: {names:?}"
    );
    assert!(
        names
            .iter()
            .any(|name| name.contains("/CoGroupByPartition/GroupByKey")),
        "missing shuffle GroupByKey: {names:?}"
    );
    assert!(
        names.iter().any(|name| name.ends_with("/HashJoin")),
        "missing DataFusion hash join DoFn: {names:?}"
    );
    assert!(
        names.iter().all(|name| !name.contains("ExtractLeftJoinKey")
            && !name.contains("ExtractRightJoinKey")
            && !name.contains("CoGBKJoin")),
        "old CoGBK join path is still present: {names:?}"
    );
    assert_eq!(
        names
            .iter()
            .filter(|name| name.contains("/Filter_"))
            .count(),
        2,
        "optimizer should push one filter to each join input: {names:?}"
    );
    assert!(
        names.iter().all(|name| !name.contains("/Projection_")),
        "projection was left as a separate graph node: {names:?}"
    );
}

#[tokio::test]
async fn sql_multiple_joins_have_unique_transform_names() {
    let pipeline = TestPipeline::new();
    let (left_schema, left_batch) = left_batch();
    let (right_schema, right_batch) = right_batch();
    let third_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("tax", DataType::Int64, false),
    ]));
    let left = pipeline
        .apply("CreateLeft", Create::new(vec![left_batch]))
        .with_arrow_schema(&left_schema);
    let right = pipeline
        .apply("CreateRight", Create::new(vec![right_batch]))
        .with_arrow_schema(&right_schema);
    let third = pipeline
        .apply(
            "CreateThird",
            Create::new(vec![ArrowRecordBatch::new(
                RecordBatch::try_new(
                    Arc::clone(&third_schema),
                    vec![
                        Arc::new(Int64Array::from(vec![1])),
                        Arc::new(Int64Array::from(vec![100])),
                    ],
                )
                .expect("third batch"),
            )]),
        )
        .with_arrow_schema(&third_schema);
    let input = PCollectionTuple::of("left", &left)
        .and("right", &right)
        .and("third", &third);
    let output = input.apply(
        "Sql",
        SqlTransform::query(
            "SELECT l.name, r.amount, t.tax \
             FROM left l \
             JOIN right r ON l.id = r.id \
             JOIN third t ON l.id = t.id",
        ),
    );
    let rows = output.apply(
        "Collect",
        FlatMap::new(|batch: ArrowRecordBatch| {
            let names = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("name");
            let amounts = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("amount");
            let taxes = batch
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("tax");
            (0..batch.num_rows())
                .map(|index| {
                    (
                        names.value(index).to_string(),
                        amounts.value(index),
                        taxes.value(index),
                    )
                })
                .collect::<Vec<_>>()
        }),
    );
    passert::that("AssertRows", &rows).contains_in_any_order([
        ("a".to_string(), 10, 100),
        ("a".to_string(), 11, 100),
        ("c".to_string(), 10, 100),
        ("c".to_string(), 11, 100),
    ]);

    pipeline.run().await.expect("pipeline and assertions");
}

#[tokio::test]
async fn sql_single_pcollection_uses_the_pcollection_table_name() {
    let pipeline = TestPipeline::new();
    let (left_schema, left_batch) = left_batch();
    let left = pipeline
        .apply("CreateLeft", Create::new(vec![left_batch]))
        .with_arrow_schema(&left_schema);
    let output = left.apply(
        "Sql",
        SqlTransform::query("SELECT id, name FROM PCOLLECTION WHERE id = 1"),
    );
    let rows = output.apply(
        "Collect",
        FlatMap::new(|batch: ArrowRecordBatch| {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id");
            let names = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("name");
            (0..batch.num_rows())
                .map(|index| (ids.value(index), names.value(index).to_string()))
                .collect::<Vec<_>>()
        }),
    );
    passert::that("AssertRows", &rows)
        .contains_in_any_order([(1, "a".to_string()), (1, "c".to_string())]);

    pipeline.run().await.expect("pipeline and assertions");
}

#[tokio::test]
async fn sql_composite_join_keys_match_all_columns() {
    let pipeline = TestPipeline::new();
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("tenant", DataType::Int64, false),
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("tenant", DataType::Int64, false),
        Field::new("id", DataType::Int64, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let left = pipeline
        .apply(
            "CreateLeft",
            Create::new(vec![ArrowRecordBatch::new(
                RecordBatch::try_new(
                    Arc::clone(&left_schema),
                    vec![
                        Arc::new(Int64Array::from(vec![1, 1, 2])),
                        Arc::new(Int64Array::from(vec![10, 11, 10])),
                        Arc::new(StringArray::from(vec!["a", "b", "c"])),
                    ],
                )
                .expect("left composite batch"),
            )]),
        )
        .with_arrow_schema(&left_schema);
    let right = pipeline
        .apply(
            "CreateRight",
            Create::new(vec![ArrowRecordBatch::new(
                RecordBatch::try_new(
                    Arc::clone(&right_schema),
                    vec![
                        Arc::new(Int64Array::from(vec![1, 2, 1])),
                        Arc::new(Int64Array::from(vec![10, 10, 12])),
                        Arc::new(Int64Array::from(vec![100, 200, 300])),
                    ],
                )
                .expect("right composite batch"),
            )]),
        )
        .with_arrow_schema(&right_schema);
    let input = PCollectionTuple::of("left", &left).and("right", &right);
    let output = input.apply(
        "Sql",
        SqlTransform::query(
            "SELECT l.name, r.amount FROM left l JOIN right r \
             ON l.tenant = r.tenant AND l.id = r.id",
        ),
    );
    let rows = output.apply(
        "Collect",
        FlatMap::new(|batch: ArrowRecordBatch| {
            let names = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("name");
            let amounts = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("amount");
            (0..batch.num_rows())
                .map(|index| (names.value(index).to_string(), amounts.value(index)))
                .collect::<Vec<_>>()
        }),
    );
    passert::that("AssertRows", &rows)
        .contains_in_any_order([("a".to_string(), 100), ("c".to_string(), 200)]);

    pipeline.run().await.expect("pipeline and assertions");
}

#[test]
fn sql_records_missing_schema_and_invalid_partition_size_as_construction_errors() {
    let pipeline = Pipeline::new();
    let (_, unschematized_batch) = left_batch();
    let unschematized = pipeline.apply("Create", Create::new(vec![unschematized_batch]));
    let _ = unschematized.apply("Sql", SqlTransform::query("SELECT * FROM PCOLLECTION"));
    let error = pipeline
        .validate()
        .expect_err("schema-less input must fail validation");
    assert!(error.to_string().contains("has no Arrow schema"), "{error}");

    let pipeline = Pipeline::new();
    let (left_schema, left_batch) = left_batch();
    let left = pipeline
        .apply("CreateWithSchema", Create::new(vec![left_batch]))
        .with_arrow_schema(&left_schema);
    let _ = left.apply(
        "InvalidPartitionSize",
        SqlTransform::query("SELECT * FROM PCOLLECTION").with_partition_size(0),
    );
    let error = pipeline
        .validate()
        .expect_err("zero partition size must fail validation");
    assert!(error.to_string().contains("greater than zero"), "{error}");
}

#[test]
fn sql_join_rejects_unbounded_inputs() {
    let pipeline = Pipeline::new();
    let (left_schema, _) = left_batch();
    let (right_schema, _) = right_batch();
    let left = pipeline
        .add_pcollection::<ArrowRecordBatch>(
            "Left",
            &ArrowRecordBatch::register_coder(&pipeline),
            IsBounded::Unbounded,
        )
        .with_arrow_schema(&left_schema);
    let right = pipeline
        .add_pcollection::<ArrowRecordBatch>(
            "Right",
            &ArrowRecordBatch::register_coder(&pipeline),
            IsBounded::Bounded,
        )
        .with_arrow_schema(&right_schema);
    let input = PCollectionTuple::of("left", &left).and("right", &right);
    let _ = input.apply(
        "Sql",
        SqlTransform::query("SELECT l.id FROM left l JOIN right r ON l.id = r.id"),
    );
    let error = pipeline
        .validate()
        .expect_err("unbounded inputs must fail validation");
    assert!(error.to_string().contains("bounded inputs only"), "{error}");
}
