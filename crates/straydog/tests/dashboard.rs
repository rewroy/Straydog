//! The dashboard's routes: the page is public, the data needs the key, the key never leaves.

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::Value;
use straydog::config::{Effort, Gemm, Settings, SplitMode, Vram};
use straydog::dashboard::{self, Dashboard};
use straydog::monitor::{LogBuffer, LogSink};
use straydog::runtime::Runtime;
use tower::ServiceExt;

fn settings() -> Settings {
    Settings {
        model: PathBuf::from("/models/IQ3_S"),
        bundle: straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, "IQ3_S").unwrap(),
        engram: None,
        expert_floor: None,
        model_draft: None,
        model_vision: None,
        draft_max: 0,
        ctx_size: 8192,
        batch_size: Some(1024),
        gpu: vec![0],
        split_mode: SplitMode::Experts,
        vram: Vram::Gib(vec![6.0]),
        state_ram: 0.0,
        row_cache: None,
        gemm: Gemm::Fp16,
        gemm_min_batch: 2,
        host: "127.0.0.1".into(),
        port: 8080,
        api_key: Some("sk-dashboard".into()),
        api_key_file: None,
        alias: "test".into(),
        temp: None,
        top_k: None,
        top_p: None,
        min_p: None,
        presence_penalty: None,
        seed: None,
        reasoning_effort: Effort::High,
        startup_timeout: 5.0,
        log_file: None,
        verbose: false,
        data_dir: PathBuf::from("/nonexistent/.straydog"),
        expert_ram: None,
        prefix_cache: false,
        keep_conversation: false,
        stop_on_stdin: false,
    }
}

async fn get(router: &axum::Router, path: &str, key: Option<&str>) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let mut request = Request::get(path);
    if let Some(key) = key {
        request = request.header("Authorization", format!("Bearer {key}"));
    }
    let response = router.clone().oneshot(request.body(Body::empty()).unwrap()).await.unwrap();
    let (parts, body) = response.into_parts();
    (parts.status, parts.headers, body.collect().await.unwrap().to_bytes().to_vec())
}

#[tokio::test]
async fn the_page_is_public_and_the_data_needs_the_key() {
    let settings = settings();
    // A worker that cannot start: the engine reports a failure.
    let runtime = Arc::new(Runtime::start(&settings, (PathBuf::from("/nonexistent/straydog-worker"), Vec::new())));
    let logs = Arc::new(LogBuffer::default());
    std::io::Write::write_all(&mut LogSink(logs.clone()), b"2026-10-19T00:00:00Z  INFO hello\n").unwrap();
    let dashboard = Arc::new(Dashboard::new(runtime.clone(), logs, &settings));
    let router = dashboard::router(dashboard.clone());

    for (path, kind) in [("/dashboard", "text/html"), ("/dashboard/app.js", "text/javascript"),
                         ("/dashboard/app.css", "text/css"), ("/dashboard/icon.svg", "image/svg+xml")] {
        let (status, headers, _) = get(&router, path, None).await;
        assert_eq!(status, StatusCode::OK, "{path}");
        assert!(headers["content-type"].to_str().unwrap().starts_with(kind), "{path}");
        assert!(headers["content-security-policy"].to_str().unwrap().contains("script-src 'self'"), "{path}");
    }
    for path in ["/dashboard/status", "/dashboard/logs"] {
        assert_eq!(get(&router, path, None).await.0, StatusCode::UNAUTHORIZED, "{path}");
        assert_eq!(get(&router, path, Some("sk-wrong")).await.0, StatusCode::UNAUTHORIZED, "{path}");
    }

    // Nothing is sampled until a dashboard asks; the first request waits for the first sample.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let (status, headers, body) = get(&router, "/dashboard/status", Some("sk-dashboard")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["cache-control"], "no-store");
    assert!(!String::from_utf8_lossy(&body).contains("sk-dashboard"));
    let status: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(status["server"]["api_key"], true);
    assert_eq!(status["server"]["quantization"], "IQ3_S");
    assert_eq!(status["engine"]["state"], "failed");
    assert_eq!(status["engine"]["activity"]["requests"], 0);
    assert!(status["host"]["memory_total"].as_u64().unwrap() > 0);
    assert!(status["host"].get("disk_read_bytes_per_second").is_some());
    assert!(status["host"].get("server_read_io_bytes_per_second").is_some());
    assert!(status["host"]["server_read_io_source"].is_string());
    let samples = status["samples"].as_array().unwrap();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0]["decode"], 0.0);
    assert!(samples[0].get("disk_read").is_some());
    assert!(samples[0].get("server_read_io").is_some());
    let since = samples[0]["time"].as_f64().unwrap();
    let (_, _, body) = get(&router, &format!("/dashboard/status?since={since}"), Some("sk-dashboard")).await;
    let newer = serde_json::from_slice::<Value>(&body).unwrap()["samples"].as_array().unwrap().clone();
    // The sampler's first read can be slow enough to meet the next one-second tick.
    // What this names is already returned, so it must not come back; a newer point may.
    assert!(newer.len() <= 1, "{newer:?}");
    assert!(newer.iter().all(|sample| sample["time"].as_f64().unwrap() > since), "{newer:?}");
    for sample in &newer {
        assert_eq!(sample["decode"], 0.0);
    }

    let (_, _, body) = get(&router, "/dashboard/logs?after=0", Some("sk-dashboard")).await;
    let logs: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(logs["next"], 1);
    assert_eq!(logs["lines"][0], "2026-10-19T00:00:00Z  INFO hello");
    drop(router);
    dashboard.stop().await;
    drop(dashboard);
    Arc::into_inner(runtime).unwrap().close().await;
}

/// The names of the strings one language of the page defines (`name: "..."`, outside the strings).
fn string_names(script: &str, language: &str, next: &str) -> std::collections::BTreeSet<String> {
    let from = script.find(&format!("  {language}: {{")).expect(language);
    let block = &script[from..][..script[from..].find(next).expect(next)];
    let (mut names, mut word, mut quoted, mut escaped) = (std::collections::BTreeSet::new(), String::new(), false, false);
    for c in block.chars().skip(language.len() + 5) {
        match (quoted, escaped, c) {
            (true, true, _) => escaped = false,
            (true, false, '\\') => escaped = true,
            (true, false, '"') => quoted = false,
            (true, _, _) => {}
            (false, _, '"') => quoted = true,
            (false, _, ':') if !word.is_empty() => {
                names.insert(std::mem::take(&mut word));
            }
            (false, _, c) if c.is_ascii_alphanumeric() || c == '_' => word.push(c),
            (false, _, _) => word.clear(),
        }
    }
    names
}

/// English and Japanese define the same strings, and every `data-i18n` the page names is one of them.
#[test]
fn the_page_has_the_same_strings_in_both_languages() {
    let script = include_str!("../dashboard/app.js");
    let english = string_names(script, "en", "\n  ja: {");
    let japanese = string_names(script, "ja", "\n};");
    assert!(english.len() > 100 && english.contains("speculative"), "{}", english.len());
    assert_eq!(english.difference(&japanese).collect::<Vec<_>>(), Vec::<&String>::new());
    assert_eq!(japanese.difference(&english).collect::<Vec<_>>(), Vec::<&String>::new());
    for attribute in include_str!("../dashboard/index.html").split("data-i18n=\"").skip(1) {
        let name = &attribute[..attribute.find('"').unwrap()];
        assert!(english.contains(name), "data-i18n {name} has no string");
    }
}
