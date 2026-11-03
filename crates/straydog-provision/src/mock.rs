//! A local HTTP server for the download tests, and scratch directories.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// A request as the server saw it.
#[derive(Debug, Clone)]
pub struct Request {
    pub path: String,
    pub range: Option<String>,
}

/// A server on a loopback port: one thread per connection, one response per connection
/// (the raw bytes `respond` returns), every request recorded.
pub struct Server {
    pub url: String,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl Server {
    pub fn start(respond: impl Fn(&Request) -> Vec<u8> + Send + Sync + 'static) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let (respond, seen) = (Arc::new(respond), requests.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (respond, seen) = (respond.clone(), seen.clone());
                std::thread::spawn(move || serve(stream, &*respond, &seen));
            }
        });
        Server { url, requests }
    }

    pub fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }

    pub fn ranges(&self) -> Vec<Option<String>> {
        self.requests().into_iter().map(|request| request.range).collect()
    }
}

fn serve(mut stream: TcpStream, respond: &(dyn Fn(&Request) -> Vec<u8> + Sync), seen: &Mutex<Vec<Request>>) {
    let mut head = Vec::new();
    let mut byte = [0u8];
    while !head.ends_with(b"\r\n\r\n") {
        if !matches!(stream.read(&mut byte), Ok(1)) {
            return;
        }
        head.push(byte[0]);
    }
    let head = String::from_utf8_lossy(&head);
    let path = head.split_whitespace().nth(1).unwrap_or_default().to_string();
    let range = head.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case("range").then(|| value.trim().to_string())
    });
    let request = Request { path, range };
    let response = respond(&request);
    seen.lock().unwrap().push(request);
    stream.write_all(&response).ok();
    // A client with the whole reply closes first. On Windows a server's close as the reply arrives can reach it as a reset (os error 10054). One with a reply
    // cut short gets its end when this one closes.
    stream.set_read_timeout(Some(std::time::Duration::from_millis(200))).ok();
    let _ = stream.read(&mut byte);
}

/// A complete response.
pub fn reply(status: u16, headers: &[(&str, String)], body: &[u8]) -> Vec<u8> {
    let mut response = head(status, headers, body.len());
    response.extend_from_slice(body);
    response
}

/// A response's status line and headers, declaring a body of `length` bytes.
pub fn head(status: u16, headers: &[(&str, String)], length: usize) -> Vec<u8> {
    let mut text = format!("HTTP/1.1 {status} Test\r\nContent-Length: {length}\r\nConnection: close\r\n");
    for (name, value) in headers {
        text += &format!("{name}: {value}\r\n");
    }
    (text + "\r\n").into_bytes()
}

/// `data` served as a server that honors ranges serves it: 206 with the part asked for, or 200.
pub fn ranged(data: &[u8], range: Option<&str>) -> Vec<u8> {
    let Some((first, last)) = range.and_then(|range| range.strip_prefix("bytes=")?.split_once('-')) else {
        return reply(200, &[], data);
    };
    let first: usize = first.parse().unwrap();
    let last = if last.is_empty() { data.len() - 1 } else { last.parse::<usize>().unwrap().min(data.len() - 1) };
    reply(206, &[("Content-Range", format!("bytes {first}-{last}/{}", data.len()))], &data[first..=last])
}

/// A fresh directory, removed when dropped.
pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new(name: &str) -> Scratch {
        let path = std::env::temp_dir().join(format!("straydog-provision-{}-{name}", std::process::id()));
        std::fs::remove_dir_all(&path).ok();
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}
