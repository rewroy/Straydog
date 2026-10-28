//! The HTTPS client: redirects followed (the Hub answers from a CDN), plain HTTP refused,
//! and a connection that stays silent for the timeout fails.
//!
//! The reference's timeout bounds each wait on the socket (connecting, each read). ureq's
//! own timeouts bound whole phases instead, and a body of tens of gigabytes has no useful
//! bound, so every wait of its transport is capped at the timeout.

use std::time::Duration;

use ureq::config::ConfigBuilder;
use ureq::typestate::AgentScope;
use ureq::unversioned::resolver::DefaultResolver;
use ureq::unversioned::transport::time::Duration as Wait;
use ureq::unversioned::transport::{Buffers, ConnectionDetails, Connector, DefaultConnector, NextTimeout, Transport};
use ureq::Agent;

/// An agent for pinned HTTPS downloads whose connections fail after `timeout` without progress.
pub fn agent(timeout: Duration) -> Agent {
    build(Agent::config_builder().https_only(true), timeout)
}

/// The same agent for plain-HTTP test servers on this machine, reached without a proxy.
#[cfg(any(test, feature = "mock"))]
pub fn plain(timeout: Duration) -> Agent {
    build(Agent::config_builder().proxy(None), timeout)
}

fn build(config: ConfigBuilder<AgentScope>, timeout: Duration) -> Agent {
    let config = config.timeout_connect(Some(timeout)).build();
    Agent::with_parts(config, DefaultConnector::new().chain(IdleTimeout(timeout)), DefaultResolver::default())
}

#[derive(Debug)]
struct IdleTimeout(Duration);

impl Connector<Box<dyn Transport>> for IdleTimeout {
    type Out = Capped;

    fn connect(&self, _details: &ConnectionDetails, chained: Option<Box<dyn Transport>>)
               -> Result<Option<Capped>, ureq::Error> {
        Ok(chained.map(|inner| Capped { inner, timeout: self.0 }))
    }
}

/// A transport whose waits end after the timeout at the latest.
#[derive(Debug)]
struct Capped {
    inner: Box<dyn Transport>,
    timeout: Duration,
}

impl Capped {
    fn cap(&self, next: NextTimeout) -> NextTimeout {
        if next.after.is_not_happening() || *next.after > self.timeout {
            NextTimeout { after: Wait::Exact(self.timeout), reason: next.reason }
        } else {
            next
        }
    }
}

impl Transport for Capped {
    fn buffers(&mut self) -> &mut dyn Buffers {
        self.inner.buffers()
    }

    fn transmit_output(&mut self, amount: usize, timeout: NextTimeout) -> Result<(), ureq::Error> {
        let timeout = self.cap(timeout);
        self.inner.transmit_output(amount, timeout)
    }

    fn maybe_await_input(&mut self, timeout: NextTimeout) -> Result<bool, ureq::Error> {
        let timeout = self.cap(timeout);
        self.inner.maybe_await_input(timeout)
    }

    fn await_input(&mut self, timeout: NextTimeout) -> Result<bool, ureq::Error> {
        let timeout = self.cap(timeout);
        self.inner.await_input(timeout)
    }

    fn is_open(&mut self) -> bool {
        self.inner.is_open()
    }

    fn is_tls(&self) -> bool {
        self.inner.is_tls()
    }
}
