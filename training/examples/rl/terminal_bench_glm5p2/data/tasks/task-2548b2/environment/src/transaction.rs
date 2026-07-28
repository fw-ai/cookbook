use crate::mvcc::MvccStorage;
use crate::timestamp::TimestampOracle;

/// A client-side transaction handle.
///
/// Writes are buffered locally until commit. Reads go directly to the
/// storage at the transaction's start timestamp (reads do NOT see the
/// transaction's own buffered writes).
pub struct Transaction {
    tso: TimestampOracle,
    storage: MvccStorage,
    start_ts: u64,
    writes: Vec<(Vec<u8>, Vec<u8>)>,
}

impl Transaction {
    /// Begins a new transaction.
    pub fn begin(tso: TimestampOracle, storage: MvccStorage) -> Self {
        // YOUR CODE HERE
        todo!()
    }

    /// Reads the value for `key` at this transaction's snapshot.
    pub fn get(&self, key: Vec<u8>) -> Result<Vec<u8>, String> {
        // YOUR CODE HERE
        todo!()
    }

    /// Buffers a write of `value` for `key`.
    pub fn set(&mut self, key: Vec<u8>, value: Vec<u8>) {
        // YOUR CODE HERE
        todo!()
    }

    /// Commits the transaction atomically.
    /// Returns Ok(true) on success, Ok(false) if the transaction must abort.
    pub fn commit(&self) -> Result<bool, String> {
        // YOUR CODE HERE
        todo!()
    }
}
