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

/// The quantization of the DeepSeek-V4.1 GSQ-RCO bundle in STRAYDOG_DS41: STRAYDOG_DS41_QUANT, else
/// IQ2_XS (GSQ-RCO-IQ2_XXS for the six-shard bundle).
#[allow(dead_code)]
pub fn deepseek41_quantization() -> String {
    std::env::var("STRAYDOG_DS41_QUANT").unwrap_or_else(|_| "IQ2_XS".into())
}

/// The DeepSeek-V4.1 bundle of [`deepseek41_quantization`].
#[allow(dead_code)]
pub fn deepseek41_bundle() -> straydog_engine::bundle::Id {
    let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::DEEPSEEK), None, &deepseek41_quantization());
    bundle.expect("a pinned DeepSeek-V4.1 bundle")
}
