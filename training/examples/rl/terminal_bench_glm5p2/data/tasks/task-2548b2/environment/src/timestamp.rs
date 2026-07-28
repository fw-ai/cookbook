use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A timestamp oracle that produces strictly increasing timestamps.
/// All transactions obtain their start and commit timestamps from this oracle.
#[derive(Clone)]
pub struct TimestampOracle {
    counter: Arc<AtomicU64>,
}

impl TimestampOracle {
    pub fn new() -> Self {
        TimestampOracle {
            counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Returns the next globally unique, monotonically increasing timestamp.
    pub fn get_timestamp(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::SeqCst) + 1
    }
}

impl Default for TimestampOracle {
    fn default() -> Self {
        Self::new()
    }
}
