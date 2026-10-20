//! What the tests that need model files share.

/// The environment variable `name` naming a model's files, or None: the test is skipped and
/// says so. With STRAYDOG_REQUIRE_MODEL set (a job that must check the models) a missing one
/// fails the test instead of passing it unchecked.
#[allow(dead_code)]
pub fn model_env(name: &str, what: &str) -> Option<std::ffi::OsString> {
    let value = std::env::var_os(name);
    if value.is_none() {
        assert!(std::env::var_os("STRAYDOG_REQUIRE_MODEL").is_none(),
                "{name} is not set but STRAYDOG_REQUIRE_MODEL is: set {name} to {what}");
        eprintln!("skipped: set {name} to {what}");
    }
    value
}
