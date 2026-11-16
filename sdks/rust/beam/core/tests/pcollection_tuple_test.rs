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

//! `PCollectionTuple` groups named PCollections of independently selected element types.

use beam::prelude::*;
use beam::values::PInput;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Copies the collection under `left`.
#[derive(Clone)]
struct CopyLeft;

impl PTransform<PCollectionTuple> for CopyLeft {
    type Output = PCollection<i64>;

    fn expand(&self, input: &PCollectionTuple) -> Self::Output {
        input
            .get(&TupleTag::<i64>::new("left"))
            .expect("left is present")
            .apply("Copy", Map::new(|n: i64| n))
    }
}

/// Two bounded collections in `p`.
fn named_collections(p: &Pipeline) -> (PCollection<i64>, PCollection<i64>) {
    (
        p.apply("CreateLeft", Create::new(vec![1_i64, 2])),
        p.apply("CreateRight", Create::new(vec![3_i64])),
    )
}

#[test]
fn of_and_exposes_the_named_collections() {
    let p = Pipeline::new();
    let (left, right) = named_collections(&p);
    let left_tag = TupleTag::new("left");
    let right_tag = TupleTag::new("right");
    let tuple = PCollectionTuple::of(&left_tag, &left).and(&right_tag, &right);

    assert_eq!(tuple.len(), 2);
    assert!(!tuple.is_empty());
    assert!(tuple.has(&left_tag));
    assert!(!tuple.has("big"));
    assert!(tuple.get_as::<i64>("big").is_none());
    assert_eq!(
        tuple.get(&right_tag).map(|collection| collection.id()),
        Some(right.id())
    );
    assert_eq!(
        tuple.iter().map(|(tag, _)| tag).collect::<Vec<_>>(),
        vec!["left", "right"]
    );
    assert_eq!(
        tuple,
        PCollectionTuple::empty(p.clone())
            .and(&left_tag, &left)
            .and(&right_tag, &right)
    );
    assert_eq!(
        PInput::pcollection_ids(&tuple),
        [
            ("left".to_string(), left.id().to_string()),
            ("right".to_string(), right.id().to_string()),
        ]
        .into_iter()
        .collect()
    );
}

#[test]
fn tuple_accepts_different_element_types() {
    let p = Pipeline::new();
    let numbers = p.apply("CreateNumbers", Create::new(vec![1_i64, 2]));
    let words = p.apply("CreateWords", Create::new(vec!["a".to_string()]));
    let number_tag = TupleTag::new("numbers");
    let word_tag = TupleTag::new("words");

    let tuple = PCollectionTuple::of(&number_tag, &numbers).and(&word_tag, &words);

    assert_eq!(
        tuple.get(&number_tag).map(PCollection::id),
        Some(numbers.id())
    );
    assert_eq!(tuple.get(&word_tag).map(PCollection::id), Some(words.id()));
    assert!(tuple.get_all()["numbers"].is::<i64>());
    assert!(tuple.get_all()["words"].is::<String>());
    assert!(tuple.get(&TupleTag::<u64>::new("numbers")).is_none());
    assert_eq!(
        tuple
            .iter_as::<i64>()
            .map(|(tag, _)| tag)
            .collect::<Vec<_>>(),
        vec!["numbers"]
    );
    assert!(p.validate().is_ok());
}

#[test]
fn string_tags_keep_insertion_order() {
    let p = Pipeline::new();
    let (left, right) = named_collections(&p);
    let tuple = PCollectionTuple::of("z", &left).and("a", &right);

    assert_eq!(
        tuple.iter().map(|(tag, _)| tag).collect::<Vec<_>>(),
        vec!["z", "a"]
    );
    assert_eq!(
        tuple.get_as::<i64>("a").map(PCollection::id),
        Some(right.id())
    );
}

#[test]
fn tuple_equality_and_hash_ignore_insertion_order() {
    let p = Pipeline::new();
    let (left, right) = named_collections(&p);
    let first = PCollectionTuple::of("left", &left).and("right", &right);
    let second = PCollectionTuple::of("right", &right).and("left", &left);

    assert_eq!(first, second);

    let first_hash = {
        let mut hasher = DefaultHasher::new();
        first.hash(&mut hasher);
        hasher.finish()
    };
    let second_hash = {
        let mut hasher = DefaultHasher::new();
        second.hash(&mut hasher);
        hasher.finish()
    };
    assert_eq!(first_hash, second_hash);
}

#[test]
fn a_duplicate_tag_is_a_construction_error() {
    let p = Pipeline::new();
    let (left, right) = named_collections(&p);
    let tuple = PCollectionTuple::of("left", &left).and("left", &right);

    assert_eq!(tuple.len(), 1);
    assert_eq!(
        tuple
            .get_as::<i64>("left")
            .map(|collection| collection.id()),
        Some(left.id())
    );
    let error = p
        .validate()
        .expect_err("a duplicate tag must fail validation");
    assert!(error.to_string().contains("duplicate tag"), "{error}");
}

#[test]
fn a_collection_from_another_pipeline_is_a_construction_error() {
    let p = Pipeline::new();
    let other = Pipeline::new();
    let left = p.apply("CreateLeft", Create::new(vec![1_i64]));
    let right = other.apply("CreateRight", Create::new(vec![2_i64]));
    let tuple = PCollectionTuple::of("left", &left).and("right", &right);

    assert_eq!(tuple.len(), 1);
    let error = p
        .validate()
        .expect_err("a collection from another pipeline must fail validation");
    assert!(error.to_string().contains("same pipeline"), "{error}");
}

#[test]
fn apply_passes_the_tuple_to_the_transform() {
    let p = Pipeline::new();
    let (left, right) = named_collections(&p);
    let tuple = PCollectionTuple::of("left", &left).and("right", &right);
    let copied = tuple.apply("CopyLeft", CopyLeft);

    assert!(p.validate().is_ok());
    assert_ne!(copied.id(), left.id());
}
