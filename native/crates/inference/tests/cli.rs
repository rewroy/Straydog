use std::process::Command;

#[test]
fn refuses_to_offer_an_unimplemented_generation_command() {
    let output = Command::new(env!("CARGO_BIN_EXE_straydog-inspect"))
        .arg("generate")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("use the Straydog worker"));
}

#[test]
fn reports_missing_model_as_failure() {
    let output = Command::new(env!("CARGO_BIN_EXE_straydog-inspect"))
        .args([
            "inspect",
            "missing-model-in-nonexistent-directory/model.gguf",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("preflight rejected"));
}
