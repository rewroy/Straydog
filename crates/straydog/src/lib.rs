//! The pieces of the `straydog` command: settings, the server's runtime and the worker.

pub mod config;
pub mod conversation;
#[cfg(any(windows, target_os = "linux"))]
pub mod conversation_store;
pub mod crash;
pub mod dashboard;
pub mod frames;
pub mod monitor;
pub mod prefix;
#[cfg(any(windows, target_os = "linux"))]
pub mod prefix_store;
pub mod progress;
pub mod runtime;
pub mod setup;
pub mod worker;
mod behind;
mod images;
mod worker_on_demand;
