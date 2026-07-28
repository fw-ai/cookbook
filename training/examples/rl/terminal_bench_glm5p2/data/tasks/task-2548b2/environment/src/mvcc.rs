use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// A key in the multi-version store: (raw_key, timestamp).
pub type Key = (Vec<u8>, u64);

/// Values stored in the columns.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Timestamp(u64),
    Vector(Vec<u8>),
}

/// The three column families.
pub enum Column {
    Write,
    Data,
    Lock,
}

/// Multi-column versioned key-value table.
#[derive(Clone, Default)]
pub struct KvTable {
    write: BTreeMap<Key, Value>,
    data: BTreeMap<Key, Value>,
    lock: BTreeMap<Key, Value>,
}

impl KvTable {
    /// Reads the latest entry for `key` in the given column within
    /// the timestamp range [ts_start_inclusive, ts_end_inclusive].
    /// A None bound means unbounded in that direction.
    pub fn read(
        &self,
        key: Vec<u8>,
        column: Column,
        ts_start_inclusive: Option<u64>,
        ts_end_inclusive: Option<u64>,
    ) -> Option<(&Key, &Value)> {
        // YOUR CODE HERE
        todo!()
    }

    /// Writes an entry to the specified column at the given timestamp.
    pub fn write(&mut self, key: Vec<u8>, column: Column, ts: u64, value: Value) {
        // YOUR CODE HERE
        todo!()
    }

    /// Erases an entry from the specified column at the given timestamp.
    pub fn erase(&mut self, key: Vec<u8>, column: Column, ts: u64) {
        // YOUR CODE HERE
        todo!()
    }
}

/// Thread-safe transactional storage layer.
#[derive(Clone, Default)]
pub struct MvccStorage {
    data: Arc<Mutex<KvTable>>,
}

impl MvccStorage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads the committed value for `key` visible at snapshot timestamp `start_ts`.
    /// Returns an empty Vec if no committed value is visible.
    pub fn get(&self, key: Vec<u8>, start_ts: u64) -> Result<Vec<u8>, String> {
        // YOUR CODE HERE
        todo!()
    }

    /// Attempts to prewrite a key-value pair for a transaction identified by `start_ts`.
    /// `primary` identifies the transaction's primary key.
    /// Returns Ok(true) on success, Ok(false) on conflict.
    pub fn prewrite(
        &self,
        key: Vec<u8>,
        value: Vec<u8>,
        start_ts: u64,
        primary: Vec<u8>,
    ) -> Result<bool, String> {
        // YOUR CODE HERE
        todo!()
    }

    /// Finalizes a previously prewritten key at `commit_ts`.
    /// Returns Ok(true) on success, Ok(false) if the prerequisite state is missing.
    pub fn commit(
        &self,
        key: Vec<u8>,
        start_ts: u64,
        commit_ts: u64,
    ) -> Result<bool, String> {
        // YOUR CODE HERE
        todo!()
    }

    /// Resolves a stale lock encountered during a read.
    fn back_off_maybe_clean_up_lock(&self, start_ts: u64, key: Vec<u8>, primary: Vec<u8>) {
        // YOUR CODE HERE
        todo!()
    }
}

#[cfg(test)]
impl MvccStorage {
    pub fn erase_lock_for_test(&self, key: Vec<u8>, ts: u64) {
        let mut table = self.data.lock().unwrap();
        table.erase(key, Column::Lock, ts);
    }

    pub fn erase_data_for_test(&self, key: Vec<u8>, ts: u64) {
        let mut table = self.data.lock().unwrap();
        table.erase(key, Column::Data, ts);
    }
}
