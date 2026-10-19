"""Model and drafter provisioning and native builds; never imported by the inference runtime."""

from __future__ import annotations

# TODO(STRAYDOG-PACKAGING): Sign artifacts and validate staged rollback before distributing builds.
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from straydog.progress import ProgressLine
from straydog.bundles import QUANTIZATIONS, model_lock
from straydog.digest import file_sha256, files_sha256, update_from_file
from straydog.platform.spawn import decode_console_output, no_window_kwargs

_ROOT = Path(__file__).resolve().parents[2]
# Packaged native build, relative to a source tree.
_PACKAGED = Path("src/straydog/native")
_KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")
_SYSTEM_LIBRARIES = ("libc.", "libm.", "libdl.", "libpthread.", "librt.", "libstdc++.", "libgcc_s.", "ld-linux")


def _asset_entries(quantization: str = "Q2_0"):
    lock = model_lock(quantization)
    for item in lock["assets"]:
        yield item, f"https://huggingface.co/{lock['repository']}/resolve/{lock['revision']}/{item['source_path']}"
    for item in lock["resources"]:
        yield (
            item,
            f"https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/{lock['tokenizer_revision']}/{item['name']}",
        )


def _verify(path: Path, asset: dict, *, label: str | None = None) -> None:
    """Size, then SHA256, of a pinned file; ``label`` shows the hashing on a progress line."""
    if path.stat().st_size != asset["bytes"]:
        raise ValueError(f"size mismatch: {path.name}")
    with path.open("rb") as file:
        if label is None:
            digest = file_sha256(file)
        else:
            with ProgressLine(label, asset["bytes"]) as line:
                digest = file_sha256(file, line.add)
    if digest != asset["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path.name}")


def _check_present(files: list[tuple[Path, dict]], *, authenticate: bool) -> None:
    """Sizes of present pinned files and, when ``authenticate``, their SHA256 (two files at a time)."""
    for path, asset in files:
        if path.stat().st_size != asset["bytes"]:
            raise ValueError(f"size mismatch: {path.name}")
    if authenticate and files:
        handles = []
        try:
            for path, _asset in files:
                handles.append(path.open("rb"))
            total = sum(asset["bytes"] for _path, asset in files)
            with ProgressLine(f"authenticating {len(files)} present file(s)", total) as line:
                digests = files_sha256(handles, line.add)
        finally:
            for handle in handles:
                handle.close()
        for (path, asset), digest in zip(files, digests, strict=True):
            if digest != asset["sha256"]:
                raise ValueError(f"SHA256 mismatch: {path.name}")
    for path, _asset in files:
        print(f"{'verified' if authenticate else 'present'} {path.name}", flush=True)


def _reuse(target: Path, asset: dict) -> bool:
    """Link (or copy) an identical pinned file from a sibling bundle directory instead of downloading.

    The bundles share their second shard byte for byte; a candidate is used only
    after its size and SHA256 match the pin of ``target``.
    """
    for candidate in sorted(target.parent.parent.glob("*/*.gguf")):
        if candidate.parent == target.parent or candidate.is_symlink() or candidate.stat().st_size != asset["bytes"]:
            continue
        try:
            _verify(candidate, asset, label=f"checking {candidate.parent.name}/{candidate.name} for reuse")
        except ValueError:
            continue
        partial = target.with_suffix(target.suffix + ".part")
        partial.unlink(missing_ok=True)
        try:
            partial.hardlink_to(candidate)
        except OSError:
            shutil.copyfile(candidate, partial)
        partial.replace(target)
        print(f"reused {candidate} as {target.name}", flush=True)
        return True
    return False


def _download(target: Path, asset: dict, url: str) -> None:
    """Fetch (or resume) one pinned file, hashing the bytes as they arrive; publish it only if they match."""
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > asset["bytes"]:
        raise ValueError(f"oversized partial asset: {partial}")
    digest = hashlib.sha256()
    if offset:
        with partial.open("rb") as file, ProgressLine(f"checking the partial {target.name}", offset) as line:
            update_from_file(digest, file, line.add)
    if offset < asset["bytes"]:
        request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urlopen(request, timeout=180) as response:
            if not response.geturl().startswith("https://"):
                raise ValueError("non-HTTPS model redirect")
            if offset and response.status == 200:
                offset, digest = 0, hashlib.sha256()
            if response.status not in {200, 206}:
                raise ValueError("unexpected model response status")
            if response.status == 206 and not response.headers.get("Content-Range", "").startswith(
                f"bytes {offset}-"
            ):
                raise ValueError("model resume range mismatch")
            with (
                partial.open("ab" if offset else "wb") as file,
                ProgressLine(f"downloading {target.name}", asset["bytes"], done=offset) as line,
            ):
                written = offset
                while block := response.read(1024 * 1024):
                    written += len(block)
                    if written > asset["bytes"]:
                        raise ValueError("model download exceeded pinned size")
                    file.write(block)
                    digest.update(block)
                    line.add(len(block))
    if partial.stat().st_size != asset["bytes"]:
        raise ValueError(f"size mismatch: {partial.name} (the download ended early; run again to resume)")
    if digest.hexdigest() != asset["sha256"]:
        partial.unlink()
        raise ValueError(f"SHA256 mismatch: {target.name} (the download was removed; run again)")
    partial.replace(target)
    print(f"verified {target.name}", flush=True)


def _fetch(root: Path, quantization: str = "Q2_0", *, authenticate_existing: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    entries = list(_asset_entries(quantization))
    _check_present([(root / asset["name"], asset) for asset, _url in entries if (root / asset["name"]).exists()],
                   authenticate=authenticate_existing)
    for asset, url in entries:
        target = root / asset["name"]
        if target.exists() or (target.suffix == ".gguf" and _reuse(target, asset)):
            continue
        _download(target, asset, url)


def fetch_assets(
    root: Path, *, offline: bool = False, quantization: str = "Q2_0", authenticate_existing: bool = True,
) -> None:
    """Fetch the fixed bundle explicitly, or check existing files offline.

    Downloads are always authenticated. Files already present are authenticated here too unless
    ``authenticate_existing`` is false, as when init starts Straydog right after: the worker
    authenticates every file before using it, so hashing them here as well would only double the wait.
    """
    if offline:
        entries = [(root / asset["name"], asset) for asset, _url in _asset_entries(quantization)]
        _check_present(entries, authenticate=authenticate_existing)
    else:
        _fetch(root, quantization, authenticate_existing=authenticate_existing)


def _msvc_environment(env: dict[str, str]) -> dict[str, str]:
    """The latest Visual Studio installation's vcvars64 environment, or ``env`` without one."""
    root = env.get("PROGRAMFILES(X86)") or r"C:\Program Files (x86)"
    vswhere = Path(root) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return env
    found = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, stdin=subprocess.DEVNULL, env=env, **no_window_kwargs(),
    )
    installations = decode_console_output(found.stdout).splitlines()
    vcvars = Path(installations[0].strip()) / "VC/Auxiliary/Build/vcvars64.bat" if installations else None
    if found.returncode or not vcvars or not vcvars.is_file():
        return env
    # cmd's own quoting (/s strips the outer pair); `set` prints the resulting environment.
    loaded = subprocess.run(
        f'cmd.exe /d /s /c ""{vcvars}" >nul 2>&1 && set"',
        capture_output=True, stdin=subprocess.DEVNULL, env=env, **no_window_kwargs(),
    )
    if loaded.returncode:
        return env
    entries = (line.partition("=") for line in decode_console_output(loaded.stdout).splitlines())
    loaded_env = {key.upper(): value for key, sep, value in entries if sep and key}
    return loaded_env if loaded_env.get("PATH") else env


def build_environment(backend: str, prefix: str | None = None,
                      environ: dict[str, str] | None = None) -> dict[str, str]:
    """The environment the native build runs in, after checking every tool it needs.

    Outside a Visual Studio developer prompt, Windows has neither MSVC nor Visual Studio's
    CMake and Ninja on PATH: the build then runs in the latest installation's vcvars64
    environment (found with vswhere), and rustup's default cargo is added. A missing tool
    fails here, naming what to install, rather than as a bare "file not found".
    """
    env = dict(os.environ if environ is None else environ)
    if sys.platform == "win32":
        env = {key.upper(): value for key, value in env.items()}   # Windows names are case-insensitive
        if not shutil.which("cl", path=env.get("PATH")):
            env = _msvc_environment(env)
        cargo_bin = Path(env.get("CARGO_HOME") or Path.home() / ".cargo") / "bin"
        if not shutil.which("cargo", path=env.get("PATH")) and (cargo_bin / "cargo.exe").is_file():
            env["PATH"] = f"{cargo_bin}{os.pathsep}{env.get('PATH', '')}"

    def found(name: str, *extra: str) -> bool:
        return bool(shutil.which(name, path=os.pathsep.join([*extra, env.get("PATH", "")])))

    missing = []
    if not found("cargo"):
        missing.append("Rust (https://rustup.rs)")
    if not found("cmake"):
        missing.append("CMake 3.24 or later (on Windows, Visual Studio's \"C++ CMake tools for Windows\")")
    if sys.platform == "win32" and not found("cl"):
        missing.append("Visual Studio 2022 or its Build Tools with \"Desktop development with C++\"")
    if backend == "CUDA":
        # CMake also finds nvcc through CUDACXX, CUDA_PATH and the prefix.
        places = [str(Path(root) / "bin") for root in (prefix, env.get("CUDA_PATH"), "/usr/local/cuda") if root]
        if not (env.get("CUDACXX") and Path(env["CUDACXX"]).is_file()) and not found("nvcc", *places):
            missing.append("the CUDA Toolkit (nvcc; 12.8 or later for RTX 50 / sm_120)")
    if missing:
        raise ValueError("building Straydog's native files needs " + "; ".join(missing))
    return env


def build_artifacts(
    *,
    source_root: Path,
    build_root: Path,
    backend: str,
    architecture: str,
    prefix: str | None = None,
    offline: bool = False,
) -> tuple[Path, Path]:
    """Build Straydog's native sources and return the built artifact paths."""
    source_root, build_root = source_root.resolve(), build_root.resolve()
    if not (source_root / "Cargo.toml").is_file() or not (source_root / "gpu/CMakeLists.txt").is_file():
        raise ValueError("a Straydog source checkout is required; pass --source-root")
    env = build_environment(backend, prefix)

    def tool(name: str) -> str:
        # Windows resolves a bare program name against this process's PATH, not the child's.
        return shutil.which(name, path=env.get("PATH")) or name

    gpu_dir, rust_dir = build_root / "gpu-build", build_root / "rust"
    configure = [
        tool("cmake"),
        "-S",
        str(source_root / "gpu"),
        "-B",
        str(gpu_dir),
        f"-DSTRAYDOG_BACKEND={backend}",
        f"-DCMAKE_{backend}_ARCHITECTURES={architecture}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if prefix:
        configure.append(f"-DCMAKE_PREFIX_PATH={prefix}")
    if sys.platform == "win32" and not env.get("CMAKE_GENERATOR") and shutil.which("ninja", path=env.get("PATH")):
        # CMake's default Visual Studio generator compiles CUDA only with the toolkit's
        # MSBuild integration installed; Ninja needs just nvcc and cl. Multi-Config keeps Release/.
        configure += ["-G", "Ninja Multi-Config"]
    if sys.platform == "win32":
        # The static C/C++ runtime: Python ships vcruntime140 but not msvcp140, and a machine
        # without the Visual C++ Redistributable could load neither artifact otherwise.
        configure.append("-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded")
        env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} -C target-feature=+crt-static".strip()
    cargo = [
        tool("cargo"),
        "build",
        "--locked",
        "--release",
        "--manifest-path",
        str(source_root / "Cargo.toml"),
        "--target-dir",
        str(rust_dir),
        "-p",
        "straydog-gguf",
    ]
    if offline:
        cargo.append("--offline")
    build = [tool("cmake"), "--build", str(gpu_dir), "--config", "Release", "--parallel", "4"]
    for command in (cargo, configure, build):
        # Explicit standard handles: with CREATE_NO_WINDOW alone the child writes its
        # compiler errors to a hidden console of its own.
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL, env=env, **no_window_kwargs())
    executable, library_name = _artifact_names()
    library = gpu_dir / ("Release" if sys.platform == "win32" else "") / library_name
    inspector = build_root / "bin" / executable
    inspector.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rust_dir / "release" / executable, inspector)
    if not library.is_file():
        raise ValueError("Straydog GPU build produced no library")
    return inspector, library


def _artifact_names() -> tuple[str, str]:
    if sys.platform == "win32":
        return "straydog-inspect.exe", "straydog_gpu.dll"
    return "straydog-inspect", "libstraydog_gpu.so"


def _host_platform() -> str:
    return f"{sys.platform}-{platform.machine().lower()}"


def _package_dir(package_root: Path, platform_name: str) -> Path:
    """Where a platform's packaged build lives: Linux at the package root (its first place), others below it."""
    return package_root if platform_name == "linux-x86_64" else package_root / platform_name


def native_source_digest(source_root: Path) -> str:
    """SHA256 over the native sources the inspector and GPU library are built from.

    Line endings are hashed as LF, so a Windows checkout (``core.autocrlf``) of
    unchanged sources has the same digest as a Linux one.
    """
    crate = source_root / "crates/straydog-gguf"
    digest = hashlib.sha256()
    gpu = [path for path in sorted((source_root / "gpu").glob("*")) if path.suffix in {".cpp", ".hpp", ".h", ".txt"}]
    for path in [crate / "Cargo.toml", *gpu, *sorted((crate / "src").glob("*"))]:
        if path.is_file():
            data = path.read_bytes().replace(b"\r\n", b"\n")
            digest.update(f"{path.relative_to(source_root).as_posix()}\0{len(data)}\0".encode() + data)
    return digest.hexdigest()


def _pe_imports(path: Path) -> list[str]:
    """The DLL names a PE32+ image imports (its import directory)."""
    data = path.read_bytes()

    def u16(at: int) -> int:
        return int.from_bytes(data[at:at + 2], "little")

    def u32(at: int) -> int:
        return int.from_bytes(data[at:at + 4], "little")

    header = u32(0x3C)
    optional = header + 24
    if data[header:header + 4] != b"PE\0\0" or u16(optional) != 0x20B:
        raise ValueError(f"{path.name} is not a PE32+ image")
    sections = [(u32(at + 12), max(u32(at + 8), u32(at + 16)), u32(at + 20))
                for at in range(optional + u16(header + 20), optional + u16(header + 20) + 40 * u16(header + 6), 40)]

    def offset(rva: int) -> int:
        return next(raw + rva - start for start, size, raw in sections if start <= rva < start + size)

    names, entry = [], offset(u32(optional + 112 + 8)) if u32(optional + 112 + 8) else None
    while entry is not None and any(data[entry:entry + 20]):
        name = offset(u32(entry + 12))
        names.append(data[name:data.index(b"\0", name)].decode("ascii"))
        entry += 20
    return names


def _needed_libraries(library: Path) -> list[str]:
    if sys.platform == "win32":
        # Windows' own DLLs and the NVIDIA driver (nvcuda.dll) live in System32.
        system = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32"
        return [name for name in _pe_imports(library)
                if not name.lower().startswith("api-ms-") and not (system / name).is_file()]
    readelf = shutil.which("readelf")
    if not readelf:
        return []
    output = subprocess.run([readelf, "-d", str(library)], check=True, capture_output=True, text=True).stdout
    needed = [line.rsplit("[", 1)[1].rstrip("]") for line in output.splitlines() if "(NEEDED)" in line]
    return [name for name in needed if not name.startswith(_SYSTEM_LIBRARIES)]


def bundle_artifacts(
    *, source_root: Path, build_root: Path, backend: str, architecture: str, prefix: str | None = None
) -> Path:
    """Build the native components and place them, with a manifest, in the source tree's package."""
    source_root = source_root.resolve()
    inspector, library = build_artifacts(
        source_root=source_root, build_root=build_root, backend=backend, architecture=architecture, prefix=prefix
    )
    target = _package_dir(source_root / _PACKAGED, _host_platform())
    target.mkdir(parents=True, exist_ok=True)
    files = []
    for built in (inspector, library):
        shutil.copy2(built, target / built.name)
        with built.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        files.append({"name": built.name, "bytes": built.stat().st_size, "sha256": digest})
    names = [name for name in architecture.replace(",", ";").split(";") if name]
    manifest = {
        "platform": _host_platform(),
        "backend": backend,
        # CMake: "86-real" is machine code only, "120-virtual" PTX only, "120" both.
        "architectures": [name.split("-")[0] for name in names if not name.endswith("-virtual")],
        **({"ptx": [name.split("-")[0] for name in names if not name.endswith("-real")]} if backend == "CUDA" else {}),
        "toolchain": _toolchain(backend, prefix),
        "requires": _needed_libraries(library),
        "source_sha256": native_source_digest(source_root),
        "files": files,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def _toolchain(backend: str, prefix: str | None) -> str:
    """The GPU toolkit a packaged build was made with (named in load-failure messages)."""
    if backend == "HIP":
        version = Path(prefix or "") / ".info/version"
        return f"ROCm {version.read_text().strip()}" if prefix and version.is_file() else ""
    root = prefix or os.environ.get("CUDA_PATH")
    try:
        version = json.loads((Path(root) / "version.json").read_text())["cuda"]["version"] if root else ""
    except (OSError, ValueError, KeyError, TypeError):
        version = ""
    return f"CUDA {version}" if version else "CUDA"


def cuda_architectures() -> list[str]:
    """Every NVIDIA GPU's compute capability as a CMake CUDA architecture (12.0 -> "120")."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        output = subprocess.run([smi, "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True,
                                text=True, timeout=10, **no_window_kwargs()).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip().replace(".", "") for line in output.splitlines()
            if line.strip().replace(".", "").isdigit()]


def _cuda_covered(manifest: dict) -> bool:
    """Whether a GPU here runs a CUDA build: machine code for its architecture, or PTX it can compile."""
    machine = {int(name) for name in manifest.get("architectures", []) if name.isdigit()}
    ptx = [int(name) for name in manifest.get("ptx", []) if name.isdigit()]
    return any(int(found) in machine or (ptx and int(found) >= min(ptx)) for found in cuda_architectures())


def _gpu_architectures() -> set[str]:
    """AMD GPU architectures from the kernel's KFD topology (empty without the AMD driver)."""
    found = set()
    for properties in _KFD_NODES.glob("*/properties"):
        try:
            text = properties.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            name, _, value = line.partition(" ")
            if name == "gfx_target_version" and value.strip().isdigit() and int(value):
                version = int(value)
                found.add(f"gfx{version // 10000}{version // 100 % 100:x}{version % 100:x}")
    return found


def packaged_build(root: Path | None = None) -> tuple[Path, Path, dict] | None:
    """The packaged inspector, GPU library and manifest when they were built for this machine."""
    root = _package_dir(root or Path(__file__).resolve().parent / "native", _host_platform())
    path = root / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("platform") != _host_platform():
        return None
    if manifest.get("backend") == "HIP" and not _gpu_architectures() & set(manifest.get("architectures", [])):
        return None
    if manifest.get("backend") == "CUDA" and not _cuda_covered(manifest):
        return None
    for item in manifest["files"]:
        _verify(root / item["name"], item)
    inspector_name, library_name = _artifact_names()
    inspector = root / inspector_name
    if not os.access(inspector, os.X_OK):
        inspector.chmod(inspector.stat().st_mode | 0o111)
    return inspector, root / library_name, manifest


def check_library_loads(library: Path, manifest: dict | None = None) -> None:
    """Load the GPU library in a child process so a missing GPU runtime fails before any model download."""
    load = (
        "import sys; from pathlib import Path; from straydog.library import load_library; "
        "load_library(Path(sys.argv[1]))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", load, str(library)], capture_output=True, timeout=60, **no_window_kwargs(),
    )
    if result.returncode:
        # The loader's message is localized on Windows (OEM code page), not UTF-8.
        reason = (decode_console_output(result.stderr).strip().splitlines() or ["unknown error"])[-1]
        needs = ""
        if manifest:
            needs = f"; the packaged {manifest['backend']} build for {', '.join(manifest['architectures'])} needs "
            needs += f"{manifest.get('toolchain') or 'its GPU runtime'} ({', '.join(manifest.get('requires', []))})"
            if manifest.get("backend") == "CUDA":
                needs += " from the nvidia-cublas-cu12 package or the CUDA Toolkit, and an NVIDIA driver"
        raise ValueError(
            f"Straydog GPU library cannot load: {reason}{needs}. Install that runtime, or build for this machine"
            " (straydog bundle --backend --architecture)"
        )


def fetch_drafter(root: Path, library: Path, *, keep_source: bool = False) -> Path:
    """Build the pinned MTP drafter GGUF in ``root`` from authenticated upstream tensors."""
    from straydog.mtp import SOURCE_FILE, convert_mtp, fetch_mtp_source

    drafter = model_lock()["drafter"]
    root.mkdir(parents=True, exist_ok=True)
    target = root / drafter["name"]
    asset = {key: drafter[key] for key in ("name", "bytes", "sha256")}
    if target.is_file():
        try:
            _verify(target, asset, label=f"authenticating {target.name}")
            return target
        except ValueError:
            target.unlink()
    source = root / SOURCE_FILE
    if not source.is_file():
        total = sum(tensor["bytes"] for tensor in drafter["source_tensors"].values())
        with ProgressLine("downloading the official MTP tensors", total) as line:
            fetch_mtp_source(drafter, source, progress=line.add)
    with ProgressLine("building the drafter (Q2_0 experts)", unit="values") as line:
        convert_mtp(source, target, library, revision=drafter["revision"], progress=line.update)
    _verify(target, asset, label=f"authenticating {target.name}")
    if not keep_source:
        source.unlink()
    return target


def main(argv: list[str] | None = None) -> int:
    """Provision the pinned model and drafter, or build the native components."""
    parser = argparse.ArgumentParser(prog="straydog", description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("build", "bundle"):
        build = commands.add_parser(
            name, help="build, then place the result in the source tree's package" if name == "bundle" else None
        )
        build.add_argument("--backend", choices=["HIP", "CUDA"], default="HIP")
        build.add_argument("--architecture", required=True, help="gfx1201 for HIP; e.g. 89 for CUDA")
        build.add_argument("--prefix", help="optional ROCm/CUDA CMake prefix")
        build.add_argument("--source-root", type=Path, default=_ROOT)
        build.add_argument("--build-root", type=Path, default=_ROOT / ".build")
    for name in ("fetch", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--quantization", choices=QUANTIZATIONS, default="Q2_0",
                             help="model bundle: Q2_0 (66.4 GB), IQ2_XS (68.0 GB), IQ3_XXS (75.8 GB)"
                             " or IQ3_S (83.6 GB)")
    mtp = commands.add_parser("fetch-mtp", help="build the pinned MTP drafter used for speculative decoding")
    mtp.add_argument("--root", type=Path, required=True)
    mtp.add_argument("--gpu-library", type=Path, required=True, help="built Straydog library (its CPU quantizer)")
    mtp.add_argument("--keep-source", action="store_true", help="keep the fetched BF16 tensors")
    args = parser.parse_args(argv)
    if args.action == "fetch":
        _fetch(args.root.resolve(), args.quantization)
    elif args.action == "fetch-mtp":
        built = fetch_drafter(args.root.resolve(), args.gpu_library.resolve(), keep_source=args.keep_source)
        print(f"verified {built}")
    elif args.action == "verify":
        fetch_assets(args.root, offline=True, quantization=args.quantization)
    elif args.action == "bundle":
        target = bundle_artifacts(
            source_root=args.source_root,
            build_root=args.build_root,
            backend=args.backend,
            architecture=args.architecture,
            prefix=args.prefix,
        )
        print(f"bundled {target}")
    else:
        build_artifacts(
            source_root=args.source_root,
            build_root=args.build_root,
            backend=args.backend,
            architecture=args.architecture,
            prefix=args.prefix,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
