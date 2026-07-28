use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// A key in the multi-version store: (raw_key, timestamp).
pub type Key = (Vec<u8>, u64);

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Timestamp(u64),
    Vector(Vec<u8>),
}

pub enum Column {
    Write,
    Data,
    Lock,
}

#[derive(Clone, Default)]
pub struct KvTable {
    write: BTreeMap<Key, Value>,
    data: BTreeMap<Key, Value>,
    lock: BTreeMap<Key, Value>,
}

impl KvTable {
    pub fn read(
        &self,
        key: Vec<u8>,
        column: Column,
        ts_start_inclusive: Option<u64>,
        ts_end_inclusive: Option<u64>,
    ) -> Option<(&Key, &Value)> {
        let col = match column {
            Column::Write => &self.write,
            Column::Data => &self.data,
            Column::Lock => &self.lock,
        };
        let start = ts_start_inclusive.unwrap_or(0);
        let end = ts_end_inclusive.unwrap_or(u64::MAX);
        col.range((key.clone(), start)..=(key, end)).rev().next()
    }

    pub fn write(&mut self, key: Vec<u8>, column: Column, ts: u64, value: Value) {
        let col = match column {
            Column::Write => &mut self.write,
            Column::Data => &mut self.data,
            Column::Lock => &mut self.lock,
        };
        col.insert((key, ts), value);
    }

    pub fn erase(&mut self, key: Vec<u8>, column: Column, ts: u64) {
        let col = match column {
            Column::Write => &mut self.write,
            Column::Data => &mut self.data,
            Column::Lock => &mut self.lock,
        };
        col.remove(&(key, ts));
    }
}

#[derive(Clone, Default)]
pub struct MvccStorage {
    data: Arc<Mutex<KvTable>>,
}

impl MvccStorage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: Vec<u8>, start_ts: u64) -> Result<Vec<u8>, String> {
        // Phase 1: Check for and resolve locks
        {
            let table = self.data.lock().unwrap();
            if let Some(((_, lock_ts), lock_val)) =
                table.read(key.clone(), Column::Lock, Some(0), Some(start_ts))
            {
                let lock_ts = *lock_ts;
                let primary = match lock_val {
                    Value::Vector(p) => p.clone(),
                    _ => return Err("invalid lock value".to_string()),
                };
                drop(table);
                self.back_off_maybe_clean_up_lock(lock_ts, key.clone(), primary);
            }
        }

        // Phase 2: Normal read through write -> data indirection
        let table = self.data.lock().unwrap();
        if let Some(((_, _), write_val)) =
            table.read(key.clone(), Column::Write, Some(0), Some(start_ts))
        {
            if let Value::Timestamp(data_ts) = write_val {
                let data_ts = *data_ts;
                if let Some((_, data_val)) =
                    table.read(key, Column::Data, Some(data_ts), Some(data_ts))
                {
                    if let Value::Vector(v) = data_val {
                        return Ok(v.clone());
                    }
                }
            }
        }

        Ok(Vec::new())
    }

    pub fn prewrite(
        &self,
        key: Vec<u8>,
        value: Vec<u8>,
        start_ts: u64,
        primary: Vec<u8>,
    ) -> Result<bool, String> {
        let mut table = self.data.lock().unwrap();

        // Check for write-write conflict: any write after start_ts
        if table
            .read(key.clone(), Column::Write, Some(start_ts + 1), None)
            .is_some()
        {
            return Ok(false);
        }

        // Check for lock conflict: any existing lock on this key
        if table.read(key.clone(), Column::Lock, None, None).is_some() {
            return Ok(false);
        }

        // Write data and place lock
        table.write(
            key.clone(),
            Column::Data,
            start_ts,
            Value::Vector(value),
        );
        table.write(key, Column::Lock, start_ts, Value::Vector(primary));

        Ok(true)
    }

    pub fn commit(
        &self,
        key: Vec<u8>,
        start_ts: u64,
        commit_ts: u64,
    ) -> Result<bool, String> {
        let mut table = self.data.lock().unwrap();

        // Verify lock still exists
        if table
            .read(key.clone(), Column::Lock, Some(start_ts), Some(start_ts))
            .is_none()
        {
            return Ok(false);
        }

        // Write the write record pointing to the data
        table.write(
            key.clone(),
            Column::Write,
            commit_ts,
            Value::Timestamp(start_ts),
        );

        // Erase the lock
        table.erase(key, Column::Lock, start_ts);

        Ok(true)
    }

    fn back_off_maybe_clean_up_lock(&self, start_ts: u64, key: Vec<u8>, primary: Vec<u8>) {
        let mut table = self.data.lock().unwrap();

        // Check if primary's lock still exists at start_ts
        if table
            .read(primary.clone(), Column::Lock, Some(start_ts), Some(start_ts))
            .is_some()
        {
            // Transaction is still in progress — back off
            return;
        }

        // Primary lock is gone — determine if committed or rolled back
        // Search write column for primary key to find a write with value Timestamp(start_ts)
        let commit_ts = {
            let search_start = (primary.clone(), 0u64);
            let mut found = None;
            for ((k, ts), v) in table.write.range(search_start..) {
                if *k != primary {
                    break;
                }
                if *v == Value::Timestamp(start_ts) {
                    found = Some(*ts);
                    break;
                }
            }
            found
        };

        if let Some(commit_ts) = commit_ts {
            // Transaction committed — roll forward this secondary key
            table.write(
                key.clone(),
                Column::Write,
                commit_ts,
                Value::Timestamp(start_ts),
            );
            table.erase(key, Column::Lock, start_ts);
        } else {
            // Transaction rolled back — clean up this secondary key
            table.erase(key.clone(), Column::Data, start_ts);
            table.erase(key, Column::Lock, start_ts);
        }
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
