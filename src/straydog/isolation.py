"""GPU worker containment (Linux bubblewrap, Windows AppContainer), with no fallback to an unrestricted child."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from straydog.bundles import model_lock


def isolation_unavailable_reason() -> str | None:
    """Why this host cannot contain the worker, or None when it can."""
    if sys.platform == "win32":
        return None   # AppContainer: every supported Windows (10 and later) provides it
    if sys.platform == "linux" and shutil.which("bwrap"):
        return None
    return "Straydog requires Linux bubblewrap or Windows AppContainer GPU isolation"


def check_exposable(paths: Iterable[Path], secrets: Iterable[Path]) -> None:
    """Refuse to expose a secret to the worker: no exposed path may be one, hold one or lie inside one."""
    secrets = [Path(secret).resolve() for secret in secrets]
    for path in paths:
        path = Path(path).resolve()
        for secret in secrets:
            if path == secret or secret in path.parents or path in secret.parents:
                raise ValueError(f"Straydog never exposes a secret, or a directory holding one, to its worker: {path}")


def worker_files(
    *, model: Path, inspector: Path, library: Path, drafter: Path | None = None, quantization: str = "Q2_0"
) -> list[Path]:
    """The authenticated assets and native artifacts the worker reads; symbolic links are refused."""
    files = [inspector.resolve(), library.resolve()]
    lock = model_lock(quantization)
    for asset in lock["assets"] + lock["resources"]:
        path = model.resolve() / asset["name"]
        if path.is_symlink():
            raise ValueError("Straydog model assets must be regular files, not symbolic links")
        files.append(path)
    if drafter is not None:
        path = drafter.resolve() / lock["drafter"]["name"]
        if path.is_symlink():
            raise ValueError("Straydog drafter asset must be a regular file, not a symbolic link")
        files.append(path)
    return files


def isolated_worker_command(
    command: list[str], *, model: Path, inspector: Path, library: Path, drafter: Path | None = None,
    quantization: str = "Q2_0", secrets: Sequence[Path] = (),
) -> list[str]:
    """Expose only runtime libraries, authenticated assets and the GPU, never ``secrets``.

    Linux: a bubblewrap command. Windows: the AppContainer launcher
    (straydog.appcontainer), which grants read access to the same files.
    No namespace-probe result can turn off these mounts or network isolation.
    """
    reason = isolation_unavailable_reason()
    if reason:
        raise RuntimeError(reason)
    files = worker_files(model=model, inspector=inspector, library=library, drafter=drafter,
                         quantization=quantization)
    if sys.platform == "win32":
        from straydog.appcontainer import appcontainer_worker_command

        directories = [model.resolve()] + ([drafter.resolve()] if drafter is not None else [])
        return appcontainer_worker_command(command, files=files, directories=directories, secrets=list(secrets))
    mounts = {
        Path("/usr"),
        Path("/sys"),
        Path("/lib"),
        Path("/lib64"),
        Path("/opt/rocm"),
        Path(sys.base_prefix),
        Path(sys.prefix),
        Path(__file__).resolve().parents[1],   # the directory holding the straydog package
        *files,
    }
    executable = Path(sys.executable)
    for _ in range(16):
        if not executable.is_symlink():
            break
        target = executable.readlink()
        if not target.is_absolute():
            target = executable.parent / target
        mounts.add(target.parent.parent)
        executable = target
    check_exposable([*mounts, model], secrets)
    args = [
        shutil.which("bwrap"),
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/dev/shm",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "TMPDIR",
        "/tmp",
    ]
    args.extend(["--dir", str(model.resolve())])
    for path in sorted(mounts):
        if path.exists():
            args.extend(["--ro-bind", str(path), str(path)])
    # Packaged ROCm may reach /opt/rocm/lib through alternatives links.
    for path in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/alternatives"):
        if Path(path).exists():
            args.extend(["--ro-bind", path, path])
    for path in (
        "/dev/kfd",
        "/dev/dri",
        "/dev/nvidia0",
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/nvidia-uvm-tools",
        "/dev/nvidia-caps",
    ):
        if Path(path).exists():
            args.extend(["--dev-bind", path, path])
    for path in sorted(Path("/dev").glob("nvidia[0-9]*")):
        if str(path) != "/dev/nvidia0":
            args.extend(["--dev-bind", str(path), str(path)])
    return args + ["--chdir", str(model.resolve()), "--"] + command
