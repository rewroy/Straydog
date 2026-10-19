//! What the dashboard reads besides the engine: the server's recent log and its host.

use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

/// How often the whole process table is read to find the server's processes (the server, its
/// sandbox and the worker); between, only those are refreshed.
const PROCESS_SCAN: Duration = Duration::from_secs(15);
/// How often the sensors are looked for again: a GPU reset renumbers hwmon.
const SENSOR_SCAN: Duration = Duration::from_secs(60);

/// The host and this server's processes, read cheaply enough to sample every second.
pub struct Host {
    state: Mutex<HostState>,
    pid: Pid,
    facts: Value,
    #[cfg(feature = "nvml")]
    nvml: Option<nvml_wrapper::Nvml>,
}

struct HostState {
    system: System,
    tree: Vec<Pid>,
    scanned: Option<Instant>,
    sensors: Sensors,
    found: Option<Instant>,
}

#[derive(Default)]
struct Sensors {
    /// The CPU package temperature input (Linux hwmon).
    cpu: Option<PathBuf>,
    gpus: Vec<GpuSensors>,
    /// Elsewhere, sysinfo's components and the temperature last read from them.
    #[cfg(not(target_os = "linux"))]
    cpu_temperature: Option<f32>,
}

/// An amdgpu device's sysfs directory and the temperature inputs of its hwmon.
struct GpuSensors {
    pci: String,
    device: PathBuf,
    hwmon: Option<PathBuf>,
    temperatures: Vec<(String, PathBuf, Option<f64>)>,
}

impl Default for Host {
    fn default() -> Host {
        let mut system = System::new();
        system.refresh_cpu_all();
        let cpus = system.cpus();
        let facts = json!({
            "host_name": System::host_name(),
            "os": System::long_os_version(),
            "kernel": System::kernel_version(),
            "architecture": System::cpu_arch(),
            "cpu": cpus.first().map(|cpu| cpu.brand().trim().to_string()),
            "logical_cores": cpus.len(),
            "physical_cores": System::physical_core_count(),
        });
        Host {
            state: Mutex::new(HostState { system, tree: Vec::new(), scanned: None, sensors: Sensors::default(), found: None }),
            pid: Pid::from_u32(std::process::id()),
            facts,
            // Absent without an NVIDIA driver: the dashboard then shows no NVIDIA sensors.
            #[cfg(feature = "nvml")]
            nvml: nvml_wrapper::Nvml::init().ok(),
        }
    }
}

impl Host {
    /// Memory, CPU and sensors now; CPU usage is measured since the previous snapshot.
    pub fn snapshot(&self) -> Value {
        let mut guard = self.state.lock().unwrap();
        let state = &mut *guard;
        state.system.refresh_memory();
        state.system.refresh_cpu_usage();
        let kind = ProcessRefreshKind::nothing().with_memory().with_cpu();
        if state.scanned.is_none_or(|at| at.elapsed() >= PROCESS_SCAN) {
            state.system.refresh_processes_specifics(ProcessesToUpdate::All, true, kind);
            state.tree = tree_of(state.system.processes(), self.pid);
            state.scanned = Some(Instant::now());
        } else {
            state.system.refresh_processes_specifics(ProcessesToUpdate::Some(&state.tree), true, kind);
            // A process ended (a restarting worker): find the tree again next time.
            if state.tree.iter().any(|pid| state.system.process(*pid).is_none()) {
                state.scanned = None;
            }
        }
        let tree: Vec<_> = state.tree.iter().filter_map(|pid| state.system.process(*pid)).collect();
        if state.found.is_none_or(|at| at.elapsed() >= SENSOR_SCAN) {
            state.sensors = Sensors::find();
            state.found = Some(Instant::now());
        }
        #[cfg_attr(not(feature = "nvml"), allow(unused_mut))]
        let mut gpus = state.sensors.gpus();
        #[cfg(feature = "nvml")]
        if let Some(nvml) = &self.nvml {
            gpus.extend(nvml_gpus(nvml));
        }
        let system = &state.system;
        let load = System::load_average();
        let mut snapshot = self.facts.clone();
        let live = json!({
            "cpu_usage": system.global_cpu_usage(),
            "cpu_cores": system.cpus().iter().map(|cpu| cpu.cpu_usage()).collect::<Vec<_>>(),
            "cpu_temperature": state.sensors.cpu_temperature(),
            "gpus": gpus,
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
        });
        for (key, value) in live.as_object().unwrap() {
            snapshot[key] = value.clone();
        }
        snapshot
    }
}

/// This server and the processes under it; Linux lists each thread as well, and those are
/// left out.
fn tree_of(processes: &std::collections::HashMap<Pid, sysinfo::Process>, root: Pid) -> Vec<Pid> {
    let ours = |mut pid: Pid| {
        // Ancestry, bounded against a cycle in a racing process table.
        for _ in 0..64 {
            if pid == root {
                return true;
            }
            match processes.get(&pid).and_then(|process| process.parent()) {
                Some(parent) => pid = parent,
                None => return false,
            }
        }
        false
    };
    processes.values()
        .filter(|process| process.thread_kind().is_none() && ours(process.pid()))
        .map(|process| process.pid())
        .collect()
}

fn read_number(path: &std::path::Path) -> Option<f64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

impl Sensors {
    #[cfg(target_os = "linux")]
    fn find() -> Sensors {
        // The CPU package: k10temp's Tctl (AMD), coretemp's package (Intel), zenpower, or an
        // SoC's thermal zone; temp1 is the package in each.
        let hwmons: Vec<PathBuf> = std::fs::read_dir("/sys/class/hwmon").into_iter().flatten().flatten().map(|entry| entry.path()).collect();
        let cpu = ["k10temp", "zenpower", "coretemp", "cpu_thermal"].iter().find_map(|name| {
            hwmons.iter()
                .find(|hwmon| std::fs::read_to_string(hwmon.join("name")).is_ok_and(|found| found.trim() == *name))
                .map(|hwmon| hwmon.join("temp1_input"))
        });
        let mut gpus = Vec::new();
        for card in std::fs::read_dir("/sys/class/drm").into_iter().flatten().flatten() {
            let name = card.file_name().to_string_lossy().into_owned();
            // card0, not its connectors (card0-DP-1).
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }
            let Ok(device) = std::fs::canonicalize(card.path().join("device")) else { continue };
            if std::fs::read_to_string(device.join("vendor")).map_or(true, |vendor| vendor.trim() != "0x1002") {
                continue;
            }
            let pci = device.file_name().map(|name| name.to_string_lossy().to_lowercase()).unwrap_or_default();
            let hwmon = std::fs::read_dir(device.join("hwmon")).ok().and_then(|mut dirs| dirs.next()).and_then(Result::ok)
                .map(|dir| dir.path());
            let temperatures = hwmon.iter().flat_map(|hwmon| {
                (1..=8).filter_map(move |index| {
                    let input = hwmon.join(format!("temp{index}_input"));
                    if !input.exists() {
                        return None;
                    }
                    let label = std::fs::read_to_string(hwmon.join(format!("temp{index}_label")))
                        .map_or_else(|_| format!("temp{index}"), |label| label.trim().to_string());
                    let critical = read_number(&hwmon.join(format!("temp{index}_crit"))).map(|milli| milli / 1000.0);
                    Some((label, input, critical))
                })
            }).collect();
            gpus.push(GpuSensors { pci, device, hwmon, temperatures });
        }
        gpus.sort_by(|a, b| a.pci.cmp(&b.pci));
        Sensors { cpu, gpus }
    }

    #[cfg(not(target_os = "linux"))]
    fn find() -> Sensors {
        // Read once per scan: a platform's sensor query (WMI) is slow.
        let components = sysinfo::Components::new_with_refreshed_list();
        let cpu_temperature = ["cpu", "package", "tctl"].iter().find_map(|name| {
            components.iter()
                .filter(|component| component.label().to_lowercase().contains(name))
                .find_map(|component| component.temperature().filter(|celsius| celsius.is_finite()))
        });
        Sensors { cpu: None, gpus: Vec::new(), cpu_temperature }
    }

    fn cpu_temperature(&self) -> Option<f64> {
        #[cfg(not(target_os = "linux"))]
        if let Some(celsius) = self.cpu_temperature {
            return Some(f64::from(celsius));
        }
        self.cpu.as_ref().and_then(|input| read_number(input)).map(|milli| milli / 1000.0)
    }

    /// AMD GPUs as amdgpu reports them: load, memory, power, fan, clock, temperatures.
    fn gpus(&self) -> Vec<Value> {
        self.gpus.iter().map(|gpu| {
            let hwmon = |file: &str| gpu.hwmon.as_ref().and_then(|hwmon| read_number(&hwmon.join(file)));
            let temperatures: Vec<Value> = gpu.temperatures.iter()
                .filter_map(|(label, input, critical)| {
                    let celsius = read_number(input)? / 1000.0;
                    Some(json!({"label": label, "celsius": celsius, "critical": critical}))
                })
                .collect();
            json!({
                "pci": gpu.pci,
                "busy_percent": read_number(&gpu.device.join("gpu_busy_percent")),
                "vram_used": read_number(&gpu.device.join("mem_info_vram_used")),
                "vram_total": read_number(&gpu.device.join("mem_info_vram_total")),
                "power_watts": hwmon("power1_average").or_else(|| hwmon("power1_input")).map(|micro| micro / 1e6),
                "power_cap_watts": hwmon("power1_cap").map(|micro| micro / 1e6),
                "fan_rpm": hwmon("fan1_input"),
                "fan_percent": null,
                "clock_mhz": hwmon("freq1_input").map(|hertz| hertz / 1e6),
                "temperatures": temperatures,
            })
        }).collect()
    }
}

/// NVIDIA GPUs through NVML.
#[cfg(feature = "nvml")]
fn nvml_gpus(nvml: &nvml_wrapper::Nvml) -> Vec<Value> {
    use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
    let count = nvml.device_count().unwrap_or(0);
    (0..count).filter_map(|index| {
        let device = nvml.device_by_index(index).ok()?;
        let pci = device.pci_info().ok()?;
        let memory = device.memory_info().ok();
        let temperature = device.temperature(TemperatureSensor::Gpu).ok();
        Some(json!({
            "pci": format!("{:04x}:{:02x}:{:02x}.0", pci.domain, pci.bus, pci.device),
            "busy_percent": device.utilization_rates().ok().map(|rates| rates.gpu),
            "vram_used": memory.as_ref().map(|memory| memory.used),
            "vram_total": memory.as_ref().map(|memory| memory.total),
            "power_watts": device.power_usage().ok().map(|milli| f64::from(milli) / 1000.0),
            "power_cap_watts": device.enforced_power_limit().ok().map(|milli| f64::from(milli) / 1000.0),
            "fan_rpm": null, "fan_percent": device.fan_speed(0).ok(), "clock_mhz": null,
            "temperatures": temperature.map_or_else(Vec::new, |celsius| vec![json!({"label": "gpu", "celsius": celsius, "critical": null})]),
        }))
    }).collect()
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
        assert_eq!(snapshot["cpu_cores"].as_array().unwrap().len(), snapshot["logical_cores"].as_u64().unwrap() as usize);
        assert!(snapshot["gpus"].is_array());
    }
}

#[cfg(test)]
mod cost {
    /// `cargo test --release -p straydog --lib cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn snapshot_cost() {
        let host = super::Host::default();
        host.snapshot();
        let started = std::time::Instant::now();
        for _ in 0..50 {
            host.snapshot();
        }
        eprintln!("snapshot: {:.2} ms", started.elapsed().as_secs_f64() * 1000.0 / 50.0);
    }
}
