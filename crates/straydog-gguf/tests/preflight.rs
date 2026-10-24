use std::io::Cursor;

use straydog_gguf::inspect;

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn text(key: &str, value: &str) -> Vec<u8> {
    let mut out = Vec::new();
    string(&mut out, key);
    out.extend_from_slice(&8u32.to_le_bytes());
    string(&mut out, value);
    out
}

fn number(key: &str, value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    string(&mut out, key);
    out.extend_from_slice(&10u32.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
    out
}

struct Tensor<'a> {
    name: &'a str,
    dims: Vec<u64>,
    kind: u32,
    offset: u64,
}

fn dense(name: &str, offset: u64) -> Tensor<'_> {
    Tensor {
        name,
        dims: vec![4],
        kind: 0,
        offset,
    }
}

fn fixture(metadata: Vec<Vec<u8>>, tensors: Vec<Tensor<'_>>, data_bytes: usize) -> Vec<u8> {
    let mut out = Vec::from(*b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for item in metadata {
        out.extend(item);
    }
    for tensor in tensors {
        string(&mut out, tensor.name);
        out.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in tensor.dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&tensor.kind.to_le_bytes());
        out.extend_from_slice(&tensor.offset.to_le_bytes());
    }
    out.resize(out.len().div_ceil(32) * 32 + data_bytes, 0);
    out
}

fn metadata() -> Vec<Vec<u8>> {
    vec![
        text("general.architecture", "qwen4exp"),
        number("qwen4exp.context_length", 262_144),
    ]
}

fn rejects(bytes: Vec<u8>, expected: &str) {
    let error = inspect(&mut Cursor::new(bytes)).unwrap_err();
    assert!(
        error.to_string().contains(expected),
        "{error} did not contain {expected}"
    );
}

#[test]
fn reports_absolute_ranges_and_declarations_without_reading_weights() {
    let bytes = fixture(metadata(), vec![dense("embedding", 0)], 16);
    let header = inspect(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(header.architecture, "qwen4exp");
    assert_eq!(header.declared_context, Some(262_144));
    assert_eq!(header.shard_count, 1);
    assert_eq!(header.tensors[0].bytes, 16);
    assert_eq!(header.tensors[0].offset, header.data_offset);
    assert_eq!(header.tensors[0].offset + 16, bytes.len() as u64);
}

#[test]
fn accepts_both_candidate_quant_blocks_and_mixed_dense_weights() {
    for (kind, row, block_bytes) in [(42, 64, 18), (17, 256, 74)] {
        let bytes = fixture(
            metadata(),
            vec![
                Tensor {
                    name: "experts",
                    dims: vec![row, 2],
                    kind,
                    offset: 0,
                },
                Tensor {
                    name: "norm",
                    dims: vec![4],
                    kind: 30,
                    offset: 160,
                },
            ],
            168,
        );
        let header = inspect(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(header.tensors[0].bytes, 2 * block_bytes);
        assert_eq!(header.tensors[1].bytes, 8);
    }
}

#[test]
fn every_truncated_prefix_is_rejected() {
    let bytes = fixture(metadata(), vec![dense("embedding", 0)], 16);
    for end in 0..bytes.len() {
        assert!(
            inspect(&mut Cursor::new(&bytes[..end])).is_err(),
            "accepted prefix {end}"
        );
    }
}

#[test]
fn rejects_huge_counts_before_allocating() {
    let valid = fixture(metadata(), vec![dense("embedding", 0)], 16);
    for position in [8, 16] {
        let mut bytes = valid.clone();
        bytes[position..position + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        rejects(bytes, "count");
    }
    let mut bytes = valid;
    bytes[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
    rejects(bytes, "string exceeds");
}

#[test]
fn rejects_arrays_with_unbounded_work_or_recursion() {
    for (element_kind, count) in [(8u32, u64::MAX), (9, 1), (99, 1)] {
        let mut array = Vec::new();
        string(&mut array, "tokenizer.ggml.tokens");
        array.extend_from_slice(&9u32.to_le_bytes());
        array.extend_from_slice(&element_kind.to_le_bytes());
        array.extend_from_slice(&count.to_le_bytes());
        let mut meta = metadata();
        meta.push(array);
        rejects(fixture(meta, vec![dense("embedding", 0)], 16), "array");
    }
}

#[test]
fn bounds_the_array_elements_kept_across_the_header() {
    // Five arrays of a million one-byte elements each fit the byte bound but not the element
    // budget: the fifth is refused before it is decoded.
    let mut meta = metadata();
    for index in 0..5 {
        let mut array = Vec::new();
        string(&mut array, &format!("bytes.{index}"));
        array.extend_from_slice(&9u32.to_le_bytes());
        array.extend_from_slice(&0u32.to_le_bytes());
        array.extend_from_slice(&1_000_000u64.to_le_bytes());
        array.extend(std::iter::repeat_n(0u8, 1_000_000));
        meta.push(array);
    }
    rejects(fixture(meta, vec![dense("embedding", 0)], 16), "oversized GGUF array");
}

#[test]
fn keeps_tokenizer_arrays_as_metadata() {
    let mut array = Vec::new();
    string(&mut array, "tokenizer.ggml.tokens");
    array.extend_from_slice(&9u32.to_le_bytes());
    array.extend_from_slice(&8u32.to_le_bytes());
    array.extend_from_slice(&2u64.to_le_bytes());
    string(&mut array, "猫");
    string(&mut array, "<|im_end|>");
    let mut meta = metadata();
    meta.push(array);
    let header = inspect(&mut Cursor::new(fixture(
        meta,
        vec![dense("embedding", 0)],
        16,
    )))
    .unwrap();
    assert_eq!(header.tensors[0].name, "embedding");
    let tokens: Vec<&str> = header.metadata["tokenizer.ggml.tokens"].array().unwrap().iter()
        .map(|token| token.text().unwrap()).collect();
    assert_eq!(tokens, ["猫", "<|im_end|>"]);
}

#[test]
fn rejects_ambiguous_metadata_and_tensor_names() {
    let mut meta = metadata();
    meta.push(text("general.architecture", "qwen4exp"));
    rejects(
        fixture(meta, vec![dense("embedding", 0)], 16),
        "duplicate GGUF metadata",
    );
    rejects(
        fixture(metadata(), vec![dense("same", 0), dense("same", 32)], 48),
        "duplicate tensor",
    );
}

#[test]
fn reports_any_architecture_but_rejects_malformed_names_and_mistyped_capacity() {
    let header = inspect(&mut Cursor::new(fixture(
        vec![text("general.architecture", "deepseek41"), number("deepseek41.context_length", 1_048_576)],
        vec![dense("x", 0)],
        16,
    )))
    .unwrap();
    assert_eq!((header.architecture.as_str(), header.declared_context), ("deepseek41", Some(1_048_576)));
    rejects(
        fixture(
            vec![text("general.architecture", "Llama 3")],
            vec![dense("x", 0)],
            16,
        ),
        "architecture name",
    );
    rejects(
        fixture(
            vec![
                text("general.architecture", "qwen4exp"),
                text("qwen4exp.context_length", "262144"),
            ],
            vec![dense("x", 0)],
            16,
        ),
        "unsigned integer",
    );
}

/// Integer tables (I32, I64) take 4 and 8 bytes a value; the one registered sidecar architecture
/// with a hyphen is accepted, no other.
#[test]
fn integer_tables_and_the_engram_sidecar() {
    let tensors = vec![
        Tensor { name: "engram_token_map.weight", dims: vec![5], kind: 26, offset: 0 },
        Tensor { name: "engram_primes.weight", dims: vec![3, 2], kind: 27, offset: 32 },
    ];
    let header = inspect(&mut Cursor::new(fixture(metadata(), tensors, 96))).unwrap();
    assert_eq!(header.tensors.iter().map(|tensor| tensor.bytes).collect::<Vec<_>>(), [20, 48]);
    let sidecar = fixture(vec![text("general.architecture", "deepseek41-engram")], vec![dense("blk.1.engram_embd.weight", 0)], 16);
    let header = inspect(&mut Cursor::new(sidecar)).unwrap();
    assert_eq!((header.architecture.as_str(), header.shard_index, header.shard_count), ("deepseek41-engram", 0, 1));
    for name in ["deepseek41-other", "qwen4-exp"] {
        rejects(fixture(vec![text("general.architecture", name)], vec![dense("x", 0)], 16), "architecture name");
    }
}

#[test]
fn rejects_unknown_types_partial_rows_and_shape_overflow() {
    for (dims, kind, expected) in [
        (vec![1], 99, "unsupported tensor type"),
        (vec![64], 17, "whole quantization blocks"),
        (vec![u64::MAX, 2], 0, "shape overflow"),
        (vec![u64::MAX], 0, "byte size overflow"),
        (vec![0], 0, "zero tensor"),
        (vec![], 0, "rank"),
    ] {
        rejects(
            fixture(
                metadata(),
                vec![Tensor {
                    name: "x",
                    dims,
                    kind,
                    offset: 0,
                }],
                16,
            ),
            expected,
        );
    }
}

#[test]
fn rejects_alignment_overlaps_and_offset_overflow() {
    for alignment in [0, 3, 8192] {
        let mut meta = metadata();
        meta.push(number("general.alignment", alignment));
        rejects(fixture(meta, vec![dense("x", 0)], 16), "alignment");
    }
    rejects(fixture(metadata(), vec![dense("x", 1)], 17), "unaligned");
    rejects(
        fixture(metadata(), vec![dense("x", 0), dense("y", 0)], 16),
        "overlapping",
    );
    rejects(
        fixture(metadata(), vec![dense("x", u64::MAX - 31)], 16),
        "offset overflow",
    );
}

#[test]
fn enforces_shard_numbering_but_does_not_claim_manifest_identity() {
    let mut meta = metadata();
    meta.push(number("split.count", 2));
    rejects(
        fixture(meta.clone(), vec![dense("x", 0)], 16),
        "missing its shard number",
    );
    meta.push(number("split.no", 1));
    let header = inspect(&mut Cursor::new(fixture(
        meta.clone(),
        vec![dense("x", 0)],
        16,
    )))
    .unwrap();
    assert_eq!((header.shard_index, header.shard_count), (1, 2));
    meta.pop();
    meta.push(number("split.no", 2));
    rejects(fixture(meta, vec![dense("x", 0)], 16), "shard numbering");
}

#[test]
fn malformed_strings_types_versions_and_booleans_are_rejected() {
    let valid = fixture(metadata(), vec![dense("x", 0)], 16);
    for position in [0, 4] {
        let mut bytes = valid.clone();
        bytes[position] = 99;
        rejects(bytes, "version 3");
    }
    let mut bytes = valid.clone();
    bytes[32] = 0xff;
    rejects(bytes, "UTF-8");
    for (kind, payload, message) in [(99u32, 0u8, "metadata type"), (7, 2, "boolean")] {
        let mut value = Vec::new();
        string(&mut value, "extra");
        value.extend_from_slice(&kind.to_le_bytes());
        value.push(payload);
        let mut meta = metadata();
        meta.push(value);
        rejects(fixture(meta, vec![dense("x", 0)], 16), message);
    }
}

#[test]
fn header_snapshot_preserves_exact_arrays_and_requires_real_header_bytes() {
    let mut array = Vec::new();
    string(&mut array, "qwen4exp.ple.layer_multipliers");
    array.extend_from_slice(&9u32.to_le_bytes());
    array.extend_from_slice(&11u32.to_le_bytes());
    array.extend_from_slice(&1u64.to_le_bytes());
    array.extend_from_slice(&23_703_573_157_769i64.to_le_bytes());
    let mut meta = metadata();
    meta.push(array);
    let bytes = fixture(meta, vec![dense("embedding", 0)], 16);
    let original = inspect(&mut Cursor::new(&bytes)).unwrap();
    let snapshot = &bytes[..original.data_offset as usize];
    let header = straydog_gguf::inspect_header(snapshot, bytes.len() as u64).unwrap();
    assert_eq!(
        header.integer_arrays["qwen4exp.ple.layer_multipliers"],
        [23_703_573_157_769]
    );
    assert_eq!(header.tensors[0].offset, original.tensors[0].offset);
    assert!(straydog_gguf::inspect_header(&snapshot[..24], bytes.len() as u64).is_err());
    assert!(straydog_gguf::inspect_header(snapshot, snapshot.len() as u64).is_err());
}

#[test]
fn ancillary_shard_may_omit_architecture_but_first_shard_must_declare_it() {
    let meta = vec![number("split.no", 1), number("split.count", 2)];
    let bytes = fixture(meta, vec![dense("ple", 0)], 16);
    assert_eq!(inspect(&mut Cursor::new(&bytes)).unwrap().architecture, "");
    let bytes = fixture(
        vec![number("split.no", 0), number("split.count", 2)],
        vec![dense("ple", 0)],
        16,
    );
    assert!(inspect(&mut Cursor::new(&bytes)).is_err());
}
