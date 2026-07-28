use crate::mvcc::MvccStorage;
use crate::timestamp::TimestampOracle;

pub struct Transaction {
    tso: TimestampOracle,
    storage: MvccStorage,
    start_ts: u64,
    writes: Vec<(Vec<u8>, Vec<u8>)>,
}

impl Transaction {
    pub fn begin(tso: TimestampOracle, storage: MvccStorage) -> Self {
        let start_ts = tso.get_timestamp();
        Transaction {
            tso,
            storage,
            start_ts,
            writes: Vec::new(),
        }
    }

    pub fn get(&self, key: Vec<u8>) -> Result<Vec<u8>, String> {
        self.storage.get(key, self.start_ts)
    }

    pub fn set(&mut self, key: Vec<u8>, value: Vec<u8>) {
        self.writes.push((key, value));
    }

    pub fn commit(&self) -> Result<bool, String> {
        if self.writes.is_empty() {
            return Ok(true);
        }

        let primary = self.writes[0].0.clone();

        // Phase 1: Prewrite all keys (primary first, then secondaries)
        for (key, value) in &self.writes {
            if !self
                .storage
                .prewrite(key.clone(), value.clone(), self.start_ts, primary.clone())?
            {
                return Ok(false);
            }
        }

        // Phase 2: Commit
        let commit_ts = self.tso.get_timestamp();

        // Commit primary first — if this fails, the transaction aborts
        if !self
            .storage
            .commit(primary.clone(), self.start_ts, commit_ts)?
        {
            return Ok(false);
        }

        // Commit secondaries (best-effort; readers will roll forward any
        // remaining secondary locks by checking the primary's commit status)
        for (key, _) in self.writes.iter().skip(1) {
            let _ = self.storage.commit(key.clone(), self.start_ts, commit_ts);
        }

        Ok(true)
    }
}
