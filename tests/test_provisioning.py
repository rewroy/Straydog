"""Provisioning remains explicit, resumable and pinned to complete hashes."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from straydog import provisioning as provision



class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}

    def geturl(self):
        return "https://cdn.huggingface.co/asset"


def test_fetch_resumes_and_authenticates_before_publishing(tmp_path, monkeypatch):
    data = b"complete pinned bytes"
    asset = {"name": "model.gguf", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(provision, "_asset_entries", lambda *_args: [(asset, "https://huggingface.co/fixed/revision/model")])
    (tmp_path / "model.gguf.part").write_bytes(data[:5])

    def response(request, timeout):
        assert request.headers["Range"] == "bytes=5-"
        return Response(data[5:], 206, {"Content-Range": f"bytes 5-{len(data) - 1}/{len(data)}"})

    monkeypatch.setattr(provision, "urlopen", response)
    provision._fetch(tmp_path)
    assert (tmp_path / "model.gguf").read_bytes() == data
    provision._fetch(tmp_path)  # Already present and verified: no network request.
    (tmp_path / "model.gguf").write_bytes(b"x" * len(data))
    with pytest.raises(ValueError, match="SHA256"):
        provision._verify(tmp_path / "model.gguf", asset)


def test_fetch_server_ignoring_range_restarts_and_bad_range_fails(tmp_path, monkeypatch):
    data = b"test"
    asset = {"name": "model", "bytes": 4, "sha256": hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(provision, "_asset_entries", lambda *_args: [(asset, "https://huggingface.co/fixed/model")])
    (tmp_path / "model.part").write_bytes(b"te")
    monkeypatch.setattr(provision, "urlopen", lambda *args, **kwargs: Response(data))
    provision._fetch(tmp_path)
    assert (tmp_path / "model").read_bytes() == data
    (tmp_path / "model").unlink()
    (tmp_path / "model.part").write_bytes(b"te")
    monkeypatch.setattr(
        provision, "urlopen", lambda *args, **kwargs: Response(b"xx", 206, {"Content-Range": "bytes 0-1/4"})
    )
    with pytest.raises(ValueError, match="range"):
        provision._fetch(tmp_path)
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_build_command_is_explicit(monkeypatch, tmp_path, platform):
    calls = []
    monkeypatch.setattr(provision.sys, "platform", platform)
    monkeypatch.setattr(provision, "no_window_kwargs", lambda: {})
    monkeypatch.setattr(provision.shutil, "which", lambda name, path=None: f"/tools/{name}")
    monkeypatch.delenv("CMAKE_GENERATOR", raising=False)
    argv = ["build", "--backend", "HIP", "--architecture", "gfx1201", "--build-root", str(tmp_path)]
    executable, library_name = provision._artifact_names()
    inspector = tmp_path / "rust/release" / executable
    inspector.parent.mkdir(parents=True)
    inspector.write_bytes(b"test inspector")
    library = tmp_path / "gpu-build" / ("Release" if platform == "win32" else "") / library_name
    library.parent.mkdir(parents=True)
    library.write_bytes(b"test GPU library")
    monkeypatch.setattr(provision.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    assert provision.main(argv) == 0
    commands = [command for command, _kwargs in calls]
    assert len(commands) == 3 and "--locked" in commands[0]
    # Tools run by their resolved paths in the checked build environment.
    assert [command[0] for command in commands] == ["/tools/cargo", "/tools/cmake", "/tools/cmake"]
    assert all(isinstance(kwargs.get("env"), dict) for _command, kwargs in calls)
    assert "-DCMAKE_HIP_ARCHITECTURES=gfx1201" in commands[1]
    # Windows builds with Ninja: the Visual Studio generator needs CUDA's MSBuild integration.
    assert ("Ninja Multi-Config" in commands[1]) == (platform == "win32")
    # Windows links the C/C++ runtime statically (Python ships no msvcp140.dll).
    assert ("-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded" in commands[1]) == (platform == "win32")
    assert ("crt-static" in calls[0][1]["env"].get("RUSTFLAGS", "")) == (platform == "win32")
    # Compiler output reaches the caller even when the child gets no console window.
    assert all(kwargs.get("stdin") is provision.subprocess.DEVNULL for _command, kwargs in calls)
    assert (tmp_path / "bin" / executable).read_bytes() == b"test inspector"
    if platform == "win32":
        calls.clear()
        monkeypatch.setenv("CMAKE_GENERATOR", "Visual Studio 17 2022")
        assert provision.main(argv) == 0
        assert "-G" not in calls[1][0]
    entries = list(provision._asset_entries())
    assert len(entries) == 5
    assert all("resolve/" in url and "/main/" not in url for _, url in entries)


def test_build_names_every_missing_tool_before_running_anything(monkeypatch):
    monkeypatch.setattr(provision.sys, "platform", "win32")
    monkeypatch.setattr(provision.shutil, "which", lambda name, path=None: None)
    monkeypatch.setattr(provision, "_msvc_environment", lambda env: env)
    monkeypatch.setattr(provision.subprocess, "run", lambda *args, **kwargs: pytest.fail("ran a build step"))
    with pytest.raises(ValueError) as failure:
        provision.build_environment("CUDA", environ={"PATH": "", "CARGO_HOME": "Z:/none"})
    message = str(failure.value)
    for tool in ("Rust", "CMake", "Visual Studio", "CUDA Toolkit"):
        assert tool in message


def test_windows_build_loads_the_visual_studio_environment_and_rustup_cargo(monkeypatch, tmp_path):
    installation = tmp_path / "VS"
    vcvars = installation / "VC/Auxiliary/Build/vcvars64.bat"
    vcvars.parent.mkdir(parents=True)
    vcvars.write_text("@rem test")
    vswhere = tmp_path / "x86/Microsoft Visual Studio/Installer/vswhere.exe"
    vswhere.parent.mkdir(parents=True)
    vswhere.write_bytes(b"")
    cargo = tmp_path / "cargo/bin"
    cargo.mkdir(parents=True)
    (cargo / "cargo.exe").write_bytes(b"")
    tools = {"cl": installation / "bin", "cmake": installation / "cmake", "nvcc": tmp_path / "cuda/bin",
             "cargo": cargo}

    def which(name, path=None):
        return str(tools[name] / name) if str(tools[name]) in (path or "").split(os.pathsep) else None

    def run(command, **kwargs):
        if isinstance(command, list):   # vswhere
            assert command[0] == str(vswhere)
            return provision.subprocess.CompletedProcess(command, 0, f"{installation}\r\n".encode(), b"")
        assert str(vcvars) in command and command.startswith("cmd.exe /d /s /c")
        # `set` prints the inherited variables too.
        lines = [f"Path={tools['cl']}{os.pathsep}{tools['cmake']}", "INCLUDE=msvc", f"CUDA_PATH={tmp_path / 'cuda'}",
                 f"CARGO_HOME={tmp_path / 'cargo'}"]
        return provision.subprocess.CompletedProcess(command, 0, "\r\n".join(lines).encode(), b"")

    monkeypatch.setattr(provision.sys, "platform", "win32")
    monkeypatch.setattr(provision, "no_window_kwargs", lambda: {})
    monkeypatch.setattr(provision.shutil, "which", which)
    monkeypatch.setattr(provision.subprocess, "run", run)
    env = provision.build_environment("CUDA", environ={
        "Path": "", "ProgramFiles(x86)": str(tmp_path / "x86"), "CARGO_HOME": str(tmp_path / "cargo")})
    assert env["INCLUDE"] == "msvc"
    entries = env["PATH"].split(os.pathsep)
    assert str(cargo) in entries and str(tools["cl"]) in entries


def test_cuda_build_accepts_nvcc_found_through_the_prefix(monkeypatch):
    monkeypatch.setattr(provision.sys, "platform", "linux")
    cuda_bin = str(Path("/opt/cuda") / "bin")
    monkeypatch.setattr(
        provision.shutil, "which",
        lambda name, path=None: f"/x/{name}" if name != "nvcc" or cuda_bin in (path or "").split(os.pathsep) else None,
    )
    assert provision.build_environment("CUDA", "/opt/cuda", environ={"PATH": "/usr/bin"})["PATH"] == "/usr/bin"
    with pytest.raises(ValueError, match="CUDA Toolkit"):
        provision.build_environment("CUDA", environ={"PATH": "/usr/bin"})


@pytest.mark.parametrize("platform_name, names, where", [
    ("linux-x86_64", {"straydog-inspect", "libstraydog_gpu.so"}, "on Linux: --backend HIP --architecture gfx1201"),
    ("win32-amd64", {"straydog-inspect.exe", "straydog_gpu.dll"},
     'on Windows: --backend CUDA --architecture "86-real;89-real;120"'),
])
def test_packaged_native_build_is_current_and_authenticated(platform_name, names, where):
    root = Path(__file__).resolve().parents[1]
    packaged = provision._package_dir(root / "src/straydog/native", platform_name)
    manifest = json.loads((packaged / "manifest.json").read_text())
    assert manifest["platform"] == platform_name
    assert manifest["source_sha256"] == provision.native_source_digest(root), (
        f"native sources changed: rebuild the packaged build with straydog bundle {where}"
    )
    for item in manifest["files"]:
        provision._verify(packaged / item["name"], item)
    assert {item["name"] for item in manifest["files"]} == names
    if platform_name.startswith("linux"):
        assert os.access(packaged / "straydog-inspect", os.X_OK)


def test_bundle_places_build_with_manifest_and_lookup_checks_platform_and_hashes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    crate = source / "crates/straydog-gguf"
    (source / "gpu").mkdir(parents=True)
    (crate / "src").mkdir(parents=True)
    (crate / "Cargo.toml").write_text("[package]")
    (source / "gpu/kernels.cpp").write_text("kernels")
    built = tmp_path / "built"
    built.mkdir()
    (built / "straydog-inspect").write_bytes(b"inspector")
    (built / "libstraydog_gpu.so").write_bytes(b"library")
    monkeypatch.setattr(provision, "build_artifacts", lambda **kwargs: (built / "straydog-inspect",
                                                                        built / "libstraydog_gpu.so"))
    monkeypatch.setattr(provision, "_needed_libraries", lambda library: ["libamdhip64.so.7"])
    monkeypatch.setattr(provision, "_host_platform", lambda: "linux-x86_64")
    monkeypatch.setattr(provision, "_gpu_architectures", lambda: {"gfx1201"})
    monkeypatch.setattr(provision.sys, "platform", "linux")
    target = provision.bundle_artifacts(source_root=source, build_root=tmp_path / "build", backend="HIP",
                                        architecture="gfx1200;gfx1201")
    manifest = json.loads((target / "manifest.json").read_text())
    assert target == source / "src/straydog/native"
    assert manifest["architectures"] == ["gfx1200", "gfx1201"] and manifest["requires"] == ["libamdhip64.so.7"]
    assert manifest["source_sha256"] == provision.native_source_digest(source)
    inspector, library, found = provision.packaged_build(target)
    assert inspector.read_bytes() == b"inspector" and library.read_bytes() == b"library" and found == manifest
    (source / "gpu/kernels.cpp").write_text("changed")
    assert provision.native_source_digest(source) != manifest["source_sha256"]
    monkeypatch.setattr(provision, "_host_platform", lambda: "linux-aarch64")
    assert provision.packaged_build(target) is None
    monkeypatch.setattr(provision, "_host_platform", lambda: "linux-x86_64")
    monkeypatch.setattr(provision, "_gpu_architectures", lambda: {"gfx1100"})   # e.g. an RDNA3 card
    assert provision.packaged_build(target) is None
    monkeypatch.setattr(provision, "_gpu_architectures", lambda: {"gfx1201"})
    library.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size|SHA256"):
        provision.packaged_build(target)


def test_windows_cuda_bundle_has_its_own_directory_and_matches_the_gpu(tmp_path, monkeypatch):
    source = tmp_path / "source"
    crate = source / "crates/straydog-gguf"
    (source / "gpu").mkdir(parents=True)
    (crate / "src").mkdir(parents=True)
    (crate / "Cargo.toml").write_text("[package]")
    built = tmp_path / "built"
    built.mkdir()
    (built / "straydog-inspect.exe").write_bytes(b"inspector")
    (built / "straydog_gpu.dll").write_bytes(b"library")
    monkeypatch.setattr(provision, "build_artifacts", lambda **kwargs: (built / "straydog-inspect.exe",
                                                                        built / "straydog_gpu.dll"))
    monkeypatch.setattr(provision, "_needed_libraries", lambda library: ["cublas64_12.dll"])
    monkeypatch.setattr(provision, "_toolchain", lambda backend, prefix: "CUDA 12.8.0")
    monkeypatch.setattr(provision, "_host_platform", lambda: "win32-amd64")
    monkeypatch.setattr(provision.sys, "platform", "win32")
    target = provision.bundle_artifacts(source_root=source, build_root=tmp_path / "build", backend="CUDA",
                                        architecture="86-real;89-real;120")
    root = source / "src/straydog/native"
    manifest = json.loads((target / "manifest.json").read_text())
    assert target == root / "win32-amd64"
    assert manifest["architectures"] == ["86", "89", "120"] and manifest["ptx"] == ["120"]
    assert manifest["requires"] == ["cublas64_12.dll"] and manifest["toolchain"] == "CUDA 12.8.0"
    # Machine code for 8.6 / 8.9 / 12.0; newer GPUs compile the 12.0 PTX; older ones build from source.
    for gpus, usable in ((["89"], True), (["120"], True), (["121"], True), (["75"], False), ([], False)):
        monkeypatch.setattr(provision, "cuda_architectures", lambda gpus=gpus: gpus)
        found = provision.packaged_build(root)
        assert (found is not None) == usable
        if found:
            assert found[0] == target / "straydog-inspect.exe" and found[1] == target / "straydog_gpu.dll"


def test_library_load_failure_names_the_packaged_runtime(tmp_path):
    library = tmp_path / "libstraydog_gpu.so"
    library.write_bytes(b"not a library")
    manifest = {"backend": "HIP", "architectures": ["gfx1201"], "toolchain": "ROCm 10.0.0",
                "requires": ["libamdhip64.so.7"]}
    with pytest.raises(ValueError, match=r"cannot load.*ROCm 10\.0\.0 \(libamdhip64\.so\.7\).*straydog bundle"):
        provision.check_library_loads(library, manifest)


def test_library_load_failure_reports_a_localized_loader_message(tmp_path, monkeypatch):
    # Windows loader errors arrive in the OEM code page (cp932 here), never as UTF-8.
    message = "OSError: 指定されたモジュールが見つかりません。".encode("cp932")
    monkeypatch.setattr(provision.subprocess, "run",
                        lambda *args, **kwargs: provision.subprocess.CompletedProcess(args, 1, b"", message))
    monkeypatch.setattr(provision, "decode_console_output", lambda data: data.decode("cp932"))
    with pytest.raises(ValueError, match="cannot load: OSError: 指定されたモジュールが見つかりません"):
        provision.check_library_loads(tmp_path / "straydog_gpu.dll")


def test_native_source_digest_ignores_checkout_line_endings(tmp_path):
    digests = {}
    for name, newline in (("lf", "\n"), ("crlf", "\r\n")):
        crate = tmp_path / name / "crates/straydog-gguf"
        crate.mkdir(parents=True)
        (tmp_path / name / "gpu").mkdir()
        (crate / "Cargo.toml").write_bytes(f"[package]{newline}name = 'straydog-gguf'{newline}".encode())
        (tmp_path / name / "gpu/kernels.cpp").write_bytes(f"int a;{newline}int b;{newline}".encode())
        digests[name] = provision.native_source_digest(tmp_path / name)
    assert digests["lf"] == digests["crlf"]
    (tmp_path / "crlf/gpu/kernels.cpp").write_bytes(b"int a;\r\nint c;\r\n")
    assert provision.native_source_digest(tmp_path / "crlf") != digests["lf"]


def test_gpu_architectures_come_from_the_kfd_topology(tmp_path, monkeypatch):
    for node, version in (("0", "0"), ("1", "120001"), ("2", "90010")):
        (tmp_path / node).mkdir()
        (tmp_path / node / "properties").write_text(f"cpu_cores_count 6\ngfx_target_version {version}\n")
    monkeypatch.setattr(provision, "_KFD_NODES", tmp_path)
    assert provision._gpu_architectures() == {"gfx1201", "gfx90a"}
    monkeypatch.setattr(provision, "_KFD_NODES", tmp_path / "absent")
    assert provision._gpu_architectures() == set()


def test_present_files_are_left_to_the_worker_unless_authentication_is_asked(tmp_path, monkeypatch):
    data = b"pinned"
    asset = {"name": "model", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(provision, "_asset_entries", lambda *_args: [(asset, "https://unused")])
    (tmp_path / "model").write_bytes(b"broken")   # right size, wrong bytes
    provision.fetch_assets(tmp_path, offline=True, authenticate_existing=False)
    provision.fetch_assets(tmp_path, authenticate_existing=False)   # present: no request either
    with pytest.raises(ValueError, match="SHA256"):
        provision.fetch_assets(tmp_path, offline=True)
    (tmp_path / "model").write_bytes(b"short")
    with pytest.raises(ValueError, match="size mismatch"):
        provision.fetch_assets(tmp_path, offline=True, authenticate_existing=False)


def test_download_is_hashed_as_it_arrives_and_a_corrupt_one_is_removed(tmp_path, monkeypatch):
    data = b"complete pinned bytes"
    asset = {"name": "model.gguf", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(provision, "_asset_entries", lambda *_args: [(asset, "https://huggingface.co/fixed/model")])
    monkeypatch.setattr(provision, "file_sha256", lambda *args: pytest.fail("a download must not be read back"))
    tmp_path = tmp_path / "bundle"   # its own parent: no sibling bundle to reuse a file from
    requests = []

    def response(request, timeout):
        requests.append(request.headers.get("Range"))
        return Response(b"X" * len(data))

    monkeypatch.setattr(provision, "urlopen", response)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        provision._fetch(tmp_path)
    assert not (tmp_path / "model.gguf.part").exists() and not (tmp_path / "model.gguf").exists()
    monkeypatch.setattr(provision, "urlopen", lambda request, timeout: requests.append(None) or Response(data))
    provision._fetch(tmp_path)   # starts over rather than resuming the bad bytes
    assert requests == [None, None] and (tmp_path / "model.gguf").read_bytes() == data
