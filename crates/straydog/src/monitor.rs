//! What the dashboard reads besides the engine: the server's recent log and its host.

use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

/// Log lines the dashboard can still read.
const LOG_LINES: usize = 2000;
/// The longest line kept; the rest is cut.
const LINE_BYTES: usize = 4096;

/// The latest log lines, numbered from the first line written.
#[derive(Default)]
pub struct LogBuffer {
    inner: Mutex<Lines>,
}

#[derive(Default)]
struct Lines {
    /// The number of the line after the newest one.
    next: u64,
    lines: VecDeque<String>,
    partial: Vec<u8>,
}

impl LogBuffer {
    /// The lines from number `after` on, at most `limit` of them, and the number to ask for
    /// next. Lines that have left the buffer are skipped.
    pub fn read(&self, after: u64, limit: usize) -> (u64, Vec<String>) {
        let inner = self.inner.lock().unwrap();
        let first = inner.next - inner.lines.len() as u64;
        let start = after.clamp(first, inner.next);
        let lines: Vec<String> = inner.lines.iter().skip((start - first) as usize).take(limit).cloned().collect();
        (start + lines.len() as u64, lines)
    }

    fn write(&self, bytes: &[u8]) {
        let mut inner = self.inner.lock().unwrap();
        for &byte in bytes {
            if byte == b'\n' {
                let line = String::from_utf8_lossy(&inner.partial).into_owned();
                inner.partial.clear();
                if inner.lines.len() == LOG_LINES {
                    inner.lines.pop_front();
                }
                inner.lines.push_back(line);
                inner.next += 1;
            } else if inner.partial.len() < LINE_BYTES {
                inner.partial.push(byte);
            }
        }
    }
}

/// A `tracing` writer into a [`LogBuffer`].
pub struct LogSink(pub Arc<LogBuffer>);

impl Write for LogSink {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.write(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The host and this server's processes (the server, its sandbox and the worker).
pub struct Host {
    system: Mutex<System>,
    pid: Pid,
}

impl Default for Host {
    fn default() -> Host {
        let mut system = System::new();
        system.refresh_cpu_all();
        Host { system: Mutex::new(system), pid: Pid::from_u32(std::process::id()) }
    }
}

impl Host {
    /// Memory and CPU now; CPU usage is measured since the previous snapshot.
    pub fn snapshot(&self) -> Value {
        let mut system = self.system.lock().unwrap();
        system.refresh_memory();
        system.refresh_cpu_usage();
        system.refresh_processes_specifics(ProcessesToUpdate::All, true,
                                           ProcessRefreshKind::nothing().with_memory().with_cpu());
        let processes = system.processes();
        let ours = |mut pid: Pid| {
            // Ancestry, bounded against a cycle in a racing process table.
            for _ in 0..64 {
                if pid == self.pid {
                    return true;
                }
                match processes.get(&pid).and_then(|process| process.parent()) {
                    Some(parent) => pid = parent,
                    None => return false,
                }
            }
            false
        };
        // Linux lists each thread as well: count and sum the processes only.
        let tree: Vec<_> = processes.values()
            .filter(|process| process.thread_kind().is_none() && ours(process.pid()))
            .collect();
        let cpus = system.cpus();
        let load = System::load_average();
        json!({
            "host_name": System::host_name(),
            "os": System::long_os_version(),
            "kernel": System::kernel_version(),
            "architecture": System::cpu_arch(),
            "cpu": cpus.first().map(|cpu| cpu.brand().trim().to_string()),
            "logical_cores": cpus.len(),
            "physical_cores": System::physical_core_count(),
            "cpu_usage": system.global_cpu_usage(),
            "load_average": if cfg!(windows) { Value::Null } else { json!([load.one, load.five, load.fifteen]) },
            "memory_total": system.total_memory(),
            "memory_available": system.available_memory(),
            "swap_total": system.total_swap(),
            "swap_used": system.used_swap(),
            "uptime_seconds": System::uptime(),
            "server": {
                "pid": self.pid.as_u32(),
                "processes": tree.len(),
                "memory": tree.iter().map(|process| process.memory()).sum::<u64>(),
                "cpu_usage": tree.iter().map(|process| process.cpu_usage()).sum::<f32>(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_log_keeps_whole_lines_in_order_and_drops_the_oldest() {
        let buffer = Arc::new(LogBuffer::default());
        let mut sink = LogSink(buffer.clone());
        sink.write_all(b"one\ntw").unwrap();
        assert_eq!(buffer.read(0, 10), (1, vec!["one".to_string()]));
        sink.write_all(b"o\nthree\n").unwrap();
        assert_eq!(buffer.read(1, 10), (3, vec!["two".to_string(), "three".to_string()]));
        assert_eq!(buffer.read(0, 1), (1, vec!["one".to_string()]));
        assert_eq!(buffer.read(9, 10), (3, Vec::<String>::new()));
        for index in 0..LOG_LINES {
            writeln!(sink, "{index}").unwrap();
        }
        // "one", "two" and "three" have left: the oldest line kept is number 3.
        assert_eq!(buffer.read(0, 1), (4, vec!["0".to_string()]));
    }

    #[test]
    fn the_host_counts_this_process() {
        let snapshot = Host::default().snapshot();
        assert!(snapshot["memory_total"].as_u64().unwrap() > 0);
        assert!(snapshot["server"]["processes"].as_u64().unwrap() >= 1);
        assert!(snapshot["server"]["memory"].as_u64().unwrap() > 0);
    }
}
