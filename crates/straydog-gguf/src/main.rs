//! Human-facing, read-only preflight. There is no inference command yet.
#![forbid(unsafe_code)]

use std::fs::File;
use std::io::Read;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2
        || !["inspect", "inspect-json", "inspect-header"]
            .iter()
            .any(|mode| args[0] == *mode)
    {
        eprintln!("usage: straydog-inspect inspect|inspect-json <model-shard.gguf>");
        eprintln!("Inspector only; use the Straydog worker for inference.");
        return ExitCode::from(2);
    }
    let result = if args[0] == "inspect-header" {
        let length = args[1].to_str().and_then(|value| value.parse::<u64>().ok());
        let mut bytes = Vec::new();
        match (
            length,
            std::io::stdin()
                .take(128 * 1024 * 1024 + 1)
                .read_to_end(&mut bytes),
        ) {
            (Some(length), Ok(_)) => straydog_gguf::inspect_header(&bytes, length),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid header input",
            )),
        }
    } else {
        File::open(&args[1]).and_then(|mut file| straydog_gguf::inspect(&mut file))
    };
    match result {
        Ok(header) => {
            if args[0] != "inspect" {
                match serde_json::to_writer(std::io::stdout().lock(), &header) {
                    Ok(()) => return ExitCode::SUCCESS,
                    Err(error) => {
                        eprintln!("cannot write inspection: {error}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            println!("architecture: {}", header.architecture);
            println!("shard: {}/{}", header.shard_index + 1, header.shard_count);
            println!("tensors: {}", header.tensors.len());
            println!("file_bytes: {}", header.file_bytes);
            match header.declared_context {
                Some(value) => println!("declared_context: {value}"),
                None => println!("declared_context: unknown"),
            }
            println!("identity_verified: false");
            println!("inference_available: false");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("GGUF preflight rejected: {error}");
            ExitCode::FAILURE
        }
    }
}
