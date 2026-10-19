use std::io::{self, Cursor, Read, Seek, SeekFrom};

use crate::{inspect, ModelHeader};

struct Snapshot<'a> {
    bytes: Cursor<&'a [u8]>,
    file_len: u64,
}

impl Read for Snapshot<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        self.bytes.read(out)
    }
}

impl Seek for Snapshot<'_> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let position = match from {
            SeekFrom::Start(value) => Some(value),
            SeekFrom::Current(delta) => self.bytes.position().checked_add_signed(delta),
            SeekFrom::End(delta) => self.file_len.checked_add_signed(delta),
        }
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid snapshot seek"))?;
        self.bytes.set_position(position);
        Ok(position)
    }
}

/// Validate a bounded header snapshot taken from an already authenticated handle.
/// The full file length is used only for range checks; missing header bytes
/// remain missing, rather than being synthesized as zeroes. This lets Windows
/// and POSIX workers inspect precisely the same file they hashed and mapped.
pub fn inspect_header(bytes: &[u8], file_len: u64) -> io::Result<ModelHeader> {
    if bytes.len() as u64 > file_len || bytes.len() > 128 * 1024 * 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid snapshot size",
        ));
    }
    inspect(&mut Snapshot {
        bytes: Cursor::new(bytes),
        file_len,
    })
}
