//! Frames that carry bytes between the server and its worker: a JSON header line, as every frame is, that says how many bytes follow in its
//! `bytes`, then exactly that many raw bytes, no line end after them. A system block's state (hundreds of megabytes) goes this way, never as text.
//!
//! The count is the header's word and is bounded before anything is read: [`BODY_LIMIT`]. A body that ends short leaves the two sides out of step,
//! which neither can mend: the reader reports it and its caller ends the worker.

use std::io::{Read, Write};

use tokio::io::{AsyncRead, AsyncReadExt};

/// The most bytes a frame's body may be: a system block's state is about 50 KB a token, and no block of a hundred thousand tokens is one a
/// prompt starts with.
pub const BODY_LIMIT: u64 = 4 << 30;

/// The frame's body length when its header gave `bytes`, refused above [`BODY_LIMIT`].
pub fn body_length(bytes: u64) -> Result<usize, String> {
    if bytes > BODY_LIMIT {
        return Err(format!("Straydog frame body of {bytes} bytes exceeds its limit of {BODY_LIMIT}"));
    }
    Ok(bytes as usize)
}

/// A buffer for `count` bytes that reports a lack of memory instead of aborting.
fn buffer(count: usize) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    body.try_reserve_exact(count).map_err(|_| format!("no memory for a frame body of {} MB", count >> 20))?;
    Ok(body)
}

/// The size of the reads a body is taken in. A pipe holds 4 KiB, and `read_to_end` sizes its reads by what the last one returned: it never asks for more than
/// a page or two, and a gigabyte then crosses in a quarter of a million reads (7.5 s, for a kept conversation's 1084 MB). A read that asks for megabytes is
/// answered by the writer's write at once.
const READ_CHUNK: usize = 4 << 20;

/// Exactly `count` bytes from `reader` (blocking), an error if it ends before them.
pub fn read_body(reader: &mut impl Read, count: u64) -> Result<Vec<u8>, String> {
    let count = body_length(count)?;
    let mut body = buffer(count)?;
    let mut chunk = vec![0u8; count.min(READ_CHUNK)];
    while body.len() < count {
        let wanted = (count - body.len()).min(chunk.len());
        match reader.read(&mut chunk[..wanted]) {
            Ok(0) => return Err(format!("Straydog frame body ended after {} of {count} bytes", body.len())),
            Ok(read) => body.extend_from_slice(&chunk[..read]),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(body)
}

/// [`read_body`] from an asynchronous reader.
pub async fn read_body_async(reader: &mut (impl AsyncRead + Unpin), count: u64) -> Result<Vec<u8>, String> {
    let count = body_length(count)?;
    let mut body = buffer(count)?;
    let read = (&mut *reader).take(count as u64).read_to_end(&mut body).await.map_err(|error| error.to_string())?;
    if read != count {
        return Err(format!("Straydog frame body ended after {read} of {count} bytes"));
    }
    Ok(body)
}

/// `header` (its `bytes` set to `body`'s length) as a line, then `body`, written to `out` and flushed.
pub fn write_with_body(out: &mut impl Write, header: &serde_json::Value, body: &[u8]) -> std::io::Result<()> {
    let mut header = header.clone();
    header["bytes"] = body.len().into();
    let mut line = serde_json::to_vec(&header).expect("frames serialize");
    line.push(b'\n');
    out.write_all(&line)?;
    out.write_all(body)?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::{BufRead, Cursor};

    /// A frame with a body reads back as it was written, and the frame after it is where it should be.
    #[test]
    fn a_body_follows_its_header_exactly() {
        let body: Vec<u8> = (0..=255u8).cycle().take(100_000).collect();
        let mut wire = Vec::new();
        write_with_body(&mut wire, &json!({"id": "a", "event": "prefix_state"}), &body).unwrap();
        wire.extend_from_slice(b"{\"id\":\"b\"}\n");
        let mut reader = Cursor::new(wire);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let header: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!((header["id"].as_str(), header["bytes"].as_u64()), (Some("a"), Some(100_000)));
        assert_eq!(read_body(&mut reader, header["bytes"].as_u64().unwrap()).unwrap(), body);
        let mut next = String::new();
        reader.read_line(&mut next).unwrap();
        assert_eq!(next, "{\"id\":\"b\"}\n", "the frame after the body is read whole");
    }

    /// A body arrives whole from a reader that gives a few bytes a read, as a pipe does, and one that is interrupted goes on.
    #[test]
    fn a_body_is_taken_from_a_reader_that_gives_a_little_at_a_time() {
        struct Dribble<'a> {
            data: &'a [u8],
            calls: usize,
        }
        impl Read for Dribble<'_> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                self.calls += 1;
                if self.calls % 5 == 0 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                let n = out.len().min(self.data.len()).min(1 + self.calls % 4093);
                out[..n].copy_from_slice(&self.data[..n]);
                self.data = &self.data[n..];
                Ok(n)
            }
        }
        let body: Vec<u8> = (0..=255u8).cycle().take(300_000).collect();
        let mut reader = Dribble { data: &body, calls: 0 };
        assert_eq!(read_body(&mut reader, 300_000).unwrap(), body);
        // (a body longer than one read's chunk, from a reader that gives it all at once)
        let long: Vec<u8> = (0..=255u8).cycle().take(READ_CHUNK * 2 + 17).collect();
        assert_eq!(read_body(&mut Cursor::new(long.clone()), long.len() as u64).unwrap(), long);
        assert_eq!(read_body(&mut Dribble { data: &body[..10], calls: 0 }, 11).unwrap_err(), "Straydog frame body ended after 10 of 11 bytes");
    }

    /// A body that ends before its count is an error naming how far it got, and a count over the limit is refused before a byte is read or a
    /// buffer is made.
    #[test]
    fn a_short_body_and_an_oversized_count_are_refused() {
        let mut reader = Cursor::new(vec![7u8; 10]);
        let short = read_body(&mut reader, 25).unwrap_err();
        assert!(short.contains("ended after 10 of 25 bytes"), "{short}");
        let mut untouched = Cursor::new(vec![7u8; 10]);
        let over = read_body(&mut untouched, BODY_LIMIT + 1).unwrap_err();
        assert!(over.contains("exceeds its limit"), "{over}");
        assert_eq!(untouched.position(), 0, "nothing was read");
        assert_eq!(read_body(&mut Cursor::new(Vec::new()), 0).unwrap(), Vec::<u8>::new());
        assert_eq!(body_length(BODY_LIMIT).unwrap() as u64, BODY_LIMIT);
    }

    /// The same, for the server's side of the pipe: a body written in pieces is read whole, a short one is an error, an oversized count refused.
    #[tokio::test]
    async fn the_server_reads_a_body_whole_or_says_why_not() {
        let (mut worker, server) = tokio::io::duplex(1024);
        let body: Vec<u8> = (0..=255u8).cycle().take(50_000).collect();
        let sent = body.clone();
        let writing = tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            for piece in sent.chunks(777) {
                worker.write_all(piece).await.unwrap();
            }
            worker
        });
        let mut reader = tokio::io::BufReader::new(server);
        assert_eq!(read_body_async(&mut reader, 50_000).await.unwrap(), body);
        drop(writing.await.unwrap());
        // (the worker ended before the rest)
        let short = read_body_async(&mut reader, 10).await.unwrap_err();
        assert!(short.contains("ended after 0 of 10 bytes"), "{short}");
        assert!(read_body_async(&mut reader, BODY_LIMIT + 1).await.unwrap_err().contains("exceeds its limit"));
    }
}
