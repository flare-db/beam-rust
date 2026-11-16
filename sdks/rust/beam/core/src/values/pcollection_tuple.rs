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

//! [`PCollectionTuple`], named [`PCollection`]s with independently typed elements.

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use indexmap::IndexMap;

use super::pcollection::PCollection;
use super::pvalue::{PInput, POutput};
use crate::pipeline::{Pipeline, PipelineError};
use crate::transforms::{PTransform, TryPTransform};

/// A typed tag naming one [`PCollection`] in a [`PCollectionTuple`].
///
/// Two tags with the same id are equal. [`PCollectionTuple::and`] and
/// [`PCollectionTuple::get`] check the element type `T`.
pub struct TupleTag<T> {
    id: String,
    element_type: PhantomData<fn() -> T>,
}

impl<T> TupleTag<T> {
    /// Creates a tag with `id`.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            element_type: PhantomData,
        }
    }

    /// Returns this tag's id.
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl<T> Clone for TupleTag<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            element_type: PhantomData,
        }
    }
}

impl<T> fmt::Debug for TupleTag<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TupleTag")
            .field("id", &self.id)
            .field("element_type", &type_name::<T>())
            .finish()
    }
}

impl<T> fmt::Display for TupleTag<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tag<{}>", self.id)
    }
}

impl<T, U> PartialEq<TupleTag<U>> for TupleTag<T> {
    fn eq(&self, other: &TupleTag<U>) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for TupleTag<T> {}

impl<T> Hash for TupleTag<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T> AsRef<str> for TupleTag<T> {
    fn as_ref(&self) -> &str {
        self.id()
    }
}

impl<T> From<&str> for TupleTag<T> {
    fn from(id: &str) -> Self {
        Self::new(id)
    }
}

impl<T> From<String> for TupleTag<T> {
    fn from(id: String) -> Self {
        Self::new(id)
    }
}

impl<T> From<&String> for TupleTag<T> {
    fn from(id: &String) -> Self {
        Self::new(id)
    }
}

impl<T> From<&TupleTag<T>> for TupleTag<T> {
    fn from(tag: &TupleTag<T>) -> Self {
        tag.clone()
    }
}

trait ErasedPCollection: 'static {
    fn as_any(&self) -> &dyn Any;
    fn clone_box(&self) -> Box<dyn ErasedPCollection>;
}

impl<T: 'static> ErasedPCollection for PCollection<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn ErasedPCollection> {
        Box::new(self.clone())
    }
}

/// A collection in a heterogeneous [`PCollectionTuple`].
///
/// Use [`downcast_ref`](Self::downcast_ref) when working with a dynamically selected tag. Code
/// that has a [`TupleTag`] should call [`PCollectionTuple::get`] for a statically typed result.
pub struct AnyPCollection {
    collection: Box<dyn ErasedPCollection>,
    id: String,
    coder_id: String,
    type_id: TypeId,
    type_name: &'static str,
}

impl Clone for AnyPCollection {
    fn clone(&self) -> Self {
        Self {
            collection: self.collection.clone_box(),
            id: self.id.clone(),
            coder_id: self.coder_id.clone(),
            type_id: self.type_id,
            type_name: self.type_name,
        }
    }
}

impl AnyPCollection {
    fn new<T: 'static>(collection: &PCollection<T>) -> Self {
        Self {
            collection: Box::new(collection.clone()),
            id: collection.id().to_string(),
            coder_id: collection.coder_id().to_string(),
            type_id: TypeId::of::<T>(),
            type_name: type_name::<T>(),
        }
    }

    /// Returns the collection id in the pipeline graph.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the coder id in the pipeline graph.
    pub fn coder_id(&self) -> &str {
        &self.coder_id
    }

    /// Returns the fully qualified Rust element type name.
    pub fn element_type_name(&self) -> &'static str {
        self.type_name
    }

    /// Returns whether this collection has element type `T`.
    pub fn is<T: 'static>(&self) -> bool {
        self.type_id == TypeId::of::<T>()
    }

    /// Returns this collection as `PCollection<T>`, or `None` if its element type differs.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&PCollection<T>> {
        self.collection.as_any().downcast_ref()
    }
}

impl fmt::Debug for AnyPCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyPCollection")
            .field("id", &self.id)
            .field("coder_id", &self.coder_id)
            .field("element_type", &self.type_name)
            .finish()
    }
}

/// An immutable tuple of independently typed [`PCollection`]s, keyed by [`TupleTag`].
///
/// All collections must belong to the same [`Pipeline`]. A duplicate tag or a collection from
/// another pipeline records a construction error. [`Pipeline::validate`] reports it.
#[derive(Clone, Debug)]
pub struct PCollectionTuple {
    pipeline: Pipeline,
    collections: IndexMap<String, AnyPCollection>,
}

impl PartialEq for PCollectionTuple {
    fn eq(&self, other: &Self) -> bool {
        self.pipeline.same_as(&other.pipeline)
            && self.collections.len() == other.collections.len()
            && self.collections.iter().all(|(tag, collection)| {
                other
                    .collections
                    .get(tag)
                    .is_some_and(|other| other.id == collection.id)
            })
    }
}

impl Eq for PCollectionTuple {}

impl Hash for PCollectionTuple {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pipeline.identity_key().hash(state);
        let mut entries = self.collections.iter().collect::<Vec<_>>();
        entries.sort_unstable_by_key(|(tag, _)| *tag);
        for (tag, collection) in entries {
            tag.hash(state);
            collection.id.hash(state);
        }
    }
}

impl PCollectionTuple {
    /// Creates an empty tuple attached to `pipeline`.
    pub fn empty(pipeline: Pipeline) -> Self {
        Self {
            pipeline,
            collections: IndexMap::new(),
        }
    }

    /// Creates a tuple containing `collection` under `tag`.
    pub fn of<T: 'static>(tag: impl Into<TupleTag<T>>, collection: &PCollection<T>) -> Self {
        Self::empty(collection.pipeline().clone()).and(tag, collection)
    }

    /// Adds `collection` under `tag` and returns the tuple.
    ///
    /// The new collection may have a different element type from every existing collection.
    /// A tag that is already present, or a collection from another pipeline, records a
    /// construction error, and the collection is not added.
    pub fn and<T: 'static>(
        mut self,
        tag: impl Into<TupleTag<T>>,
        collection: &PCollection<T>,
    ) -> Self {
        let tag = tag.into();
        if !collection.pipeline().same_as(&self.pipeline) {
            self.pipeline
                .record_construction_error(PipelineError::InvalidConstruction {
                    transform_id: format!("PCollectionTuple[{tag}]"),
                    reason:
                        "all collections in a PCollectionTuple must belong to the same pipeline"
                            .to_string(),
                });
            return self;
        }
        if self.collections.contains_key(tag.id()) {
            self.pipeline
                .record_construction_error(PipelineError::InvalidConstruction {
                    transform_id: format!("PCollectionTuple[{tag}]"),
                    reason: format!("duplicate tag '{}' in PCollectionTuple", tag.id()),
                });
            return self;
        }
        self.collections
            .insert(tag.id().to_string(), AnyPCollection::new(collection));
        self
    }

    /// Returns the pipeline this tuple belongs to.
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    /// Returns whether the tuple has a collection under `tag`.
    pub fn has(&self, tag: impl AsRef<str>) -> bool {
        self.collections.contains_key(tag.as_ref())
    }

    /// Returns whether the tuple contains the id named by the typed `tag`.
    pub fn has_tag<T>(&self, tag: &TupleTag<T>) -> bool {
        self.has(tag)
    }

    /// Returns the collection under `tag`, or `None` if the tag is absent.
    pub fn get<T: 'static>(&self, tag: &TupleTag<T>) -> Option<&PCollection<T>> {
        self.get_as(tag.id())
    }

    /// Returns the collection with id `tag` as element type `T`.
    pub fn get_as<T: 'static>(&self, tag: impl AsRef<str>) -> Option<&PCollection<T>> {
        self.collections
            .get(tag.as_ref())
            .and_then(AnyPCollection::downcast_ref)
    }

    /// Returns all collections, keyed by tag and in insertion order.
    pub fn get_all(&self) -> &IndexMap<String, AnyPCollection> {
        &self.collections
    }

    /// Iterates over the collections in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &AnyPCollection)> {
        self.collections
            .iter()
            .map(|(tag, collection)| (tag.as_str(), collection))
    }

    /// Iterates over the collections with element type `T`, in insertion order.
    pub fn iter_as<T: 'static>(&self) -> impl Iterator<Item = (&str, &PCollection<T>)> {
        self.iter()
            .filter_map(|(tag, collection)| collection.downcast_ref().map(|pc| (tag, pc)))
    }

    /// Returns the number of collections in the tuple.
    pub fn len(&self) -> usize {
        self.collections.len()
    }

    /// Returns whether the tuple contains no collections.
    pub fn is_empty(&self) -> bool {
        self.collections.is_empty()
    }

    /// Applies `transform` to this tuple under the name `name`.
    pub fn apply<Tform>(&self, name: impl Into<String>, transform: Tform) -> Tform::Output
    where
        Tform: PTransform<Self>,
    {
        self.pipeline
            .apply_scoped(&name.into(), PInput::pcollection_ids(self), || {
                transform.expand(self)
            })
    }

    /// Applies a fallible transform to this tuple under the name `name`.
    pub fn try_apply<Tform>(
        &self,
        name: impl Into<String>,
        transform: Tform,
    ) -> Result<Tform::Output, Tform::Error>
    where
        Tform: TryPTransform<Self>,
    {
        self.pipeline
            .try_apply_scoped(&name.into(), PInput::pcollection_ids(self), || {
                transform.try_expand(self)
            })
    }
}

impl PInput for PCollectionTuple {
    fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    /// The input collection of each tag.
    fn pcollection_ids(&self) -> HashMap<String, String> {
        self.collections
            .iter()
            .map(|(tag, collection)| (tag.clone(), collection.id().to_string()))
            .collect()
    }
}

impl POutput for PCollectionTuple {
    fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    /// The output collection of each tag.
    fn pcollection_ids(&self) -> HashMap<String, String> {
        self.collections
            .iter()
            .map(|(tag, collection)| (tag.clone(), collection.id().to_string()))
            .collect()
    }
}
