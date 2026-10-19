"""The Windows Straydog launcher grants only what the worker reads and runs it in its own AppContainer."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from straydog import appcontainer as launcher


def test_command_names_runtime_trees_files_and_the_worker_after_the_separator(tmp_path, monkeypatch):
    monkeypatch.setattr("straydog.library.cuda_runtime_directories", lambda: [tmp_path / "cuda/bin"])
    model = tmp_path / "model"
    command = launcher.appcontainer_worker_command(
        ["python", "-m", "straydog.worker"], files=[model / "a.gguf", tmp_path / "gpu.dll"],
        directories=[model],
    )
    assert command[:4] == [sys.executable, "-I", "-m", "straydog.appcontainer"]
    assert command[command.index("--"):] == ["--", "python", "-m", "straydog.worker"]
    pairs = list(zip(command, command[1:], strict=False))
    trees = {value for flag, value in pairs if flag == "--tree"}
    assert {str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve()),
            str((tmp_path / "cuda/bin").resolve())} <= trees
    assert str(Path(launcher.__file__).resolve().parents[1]) in trees   # the directory holding the package
    assert [value for flag, value in pairs if flag == "--file"] == [str(model / "a.gguf"), str(tmp_path / "gpu.dll")]
    assert [value for flag, value in pairs if flag == "--directory"] == [str(model)]


def test_a_secret_and_the_directories_holding_it_are_never_granted(tmp_path, monkeypatch):
    from straydog.isolation import check_exposable

    key = tmp_path / "home/.straydog/api-key"
    for path in (key, tmp_path / "home/.straydog", tmp_path / "home"):
        with pytest.raises(ValueError, match="secret"):
            check_exposable([path], [key])
    check_exposable([tmp_path / "home/.straydog/models/IQ3_S"], [key])
    secrets = tmp_path / "secrets"
    with pytest.raises(ValueError, match="secret"):
        check_exposable([secrets / "inside"], [secrets])
    monkeypatch.setattr("straydog.platform.windows._create_app_container_profile",
                        lambda name: pytest.fail("profile created before the check"))
    with pytest.raises(ValueError, match="secret"):
        launcher.grant_worker_access(trees=[tmp_path / "home"], files=[], directories=[], secrets=[key])


def _fake_acl(monkeypatch, ungranted: set[str], *, inheritance_reaches: bool = True):
    """ACL calls against a set of paths that lack the grant; a tree-level grant clears them when inheritance reaches."""
    from straydog.platform import windows as acl

    calls = []

    def grant(path, sid, mask, inherit=True):
        calls.append((path, mask, inherit))
        if inherit and inheritance_reaches:
            ungranted.difference_update({entry for entry in ungranted if entry.startswith(path)})
        ungranted.discard(path)

    monkeypatch.setattr(acl, "_create_app_container_profile", lambda name: 7)
    monkeypatch.setattr(acl, "_ensure_path_grant", lambda path, sid, mask, inherit=True: None)
    monkeypatch.setattr(acl, "_path_has_grant", lambda path, sid, mask, inherit: path not in ungranted)
    monkeypatch.setattr(acl, "_grant_path_access", grant)
    return calls


def _environment(tmp_path):
    site = tmp_path / "venv/Lib/site-packages"
    module = site / "numpy/_core/_multiarray_umath.cp311-win_amd64.pyd"
    module.parent.mkdir(parents=True)
    module.write_bytes(b"MZ")
    (site / "numpy/__init__.py").write_text("")
    (site / "_editable_impl_straydog.pth").write_text("src")
    return tmp_path / "venv", site, module


def test_environment_files_linked_without_the_grant_get_it_again(tmp_path, monkeypatch):
    """uv hard-links packages from its cache, so new files keep the cache's ACL, not the tree's grant."""
    from straydog.platform.windows import _RX, FILE_GENERIC_READ

    venv, site, module = _environment(tmp_path)
    calls = _fake_acl(monkeypatch, {str(module), str(site / "_editable_impl_straydog.pth")})
    assert sorted(launcher._ungranted_entries(venv, 7)) == sorted(
        [(str(module), _RX), (str(site / "_editable_impl_straydog.pth"), FILE_GENERIC_READ)])
    assert launcher.grant_worker_access(trees=[venv, tmp_path / "src"], files=[], directories=[]) == 7
    assert calls == [(str(venv), _RX, True)]   # re-propagated once; the source tree has no site-packages
    calls.clear()
    launcher.grant_worker_access(trees=[venv], files=[], directories=[])
    assert calls == []


def test_a_reinstalled_module_inheritance_cannot_reach_gets_its_own_grant(tmp_path, monkeypatch):
    # After `uv sync` the numpy directory inherits the grant but its extension module did not:
    # the worker failed with "DLL load failed while importing _multiarray_umath: Access is denied".
    from straydog.platform.windows import _RX

    venv, _site, module = _environment(tmp_path)
    calls = _fake_acl(monkeypatch, {str(module)}, inheritance_reaches=False)
    launcher.grant_worker_access(trees=[venv], files=[], directories=[])
    assert calls == [(str(venv), _RX, True), (str(module), _RX, False)]


def test_an_entry_that_refuses_the_grant_names_itself(tmp_path, monkeypatch):
    from straydog.platform import windows as acl

    venv, _site, module = _environment(tmp_path)
    _fake_acl(monkeypatch, {str(module)}, inheritance_reaches=False)

    def refuse(path, sid, mask, inherit=True):
        if not inherit:
            raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(acl, "_grant_path_access", refuse)
    with pytest.raises(ValueError, match="_multiarray_umath"):
        launcher.grant_worker_access(trees=[venv], files=[], directories=[])


def test_main_grants_then_launches_the_command_after_the_separator(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(launcher.sys, "platform", "win32")
    monkeypatch.setattr(launcher, "grant_worker_access", lambda **kwargs: seen.append(kwargs) or 7)
    monkeypatch.setattr(launcher, "launch", lambda command, sid: seen.append((command, sid)) or 3)
    monkeypatch.setattr("sys.argv", ["launcher", "--tree", str(tmp_path), "--file", str(tmp_path / "a"),
                                     "--directory", str(tmp_path), "--secret", str(tmp_path / "key"),
                                     "--", "python", "-c", "--file"])
    assert launcher.main() == 3
    assert seen == [{"trees": [tmp_path], "files": [tmp_path / "a"], "directories": [tmp_path],
                     "secrets": [tmp_path / "key"]},
                    (["python", "-c", "--file"], 7)]
    monkeypatch.setattr(launcher, "grant_worker_access",
                        lambda **kwargs: (_ for _ in ()).throw(ValueError("missing worker file")))
    assert launcher.main() == 1
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    with pytest.raises(SystemExit):
        launcher.main()


_CHILD = r"""
import json, os, socket, sys
from pathlib import Path
line = sys.stdin.readline().strip()
result = {"echo": line}
try:
    socket.create_connection(("127.0.0.1", 445), timeout=3).close(); result["loopback"] = "open"
except OSError:
    result["loopback"] = "blocked"
for name, path in (("cwd", Path.cwd() / "probe.txt"), ("home", Path.home() / "straydog-probe.txt")):
    try:
        path.write_text("x"); path.unlink(); result[name] = "written"
    except OSError:
        result[name] = "denied"
try:
    Path(sys.argv[1]).read_bytes(); result["ungranted"] = "read"
except OSError:
    result["ungranted"] = "denied"
print(json.dumps(result), flush=True)
"""


@pytest.mark.gpu   # opt-in: grants the Straydog AppContainer profile read access to this Python and source tree
@pytest.mark.timeout(1800)   # the first grant on a tree propagates through it
@pytest.mark.skipif(sys.platform != "win32", reason="Windows AppContainer")
def test_worker_runs_contained_over_the_server_pipes(tmp_path):
    from straydog.platform import process_group
    from straydog.platform.environment import sanitized_subprocess_env

    secret = tmp_path / "not-granted.txt"
    secret.write_text("private")

    async def run() -> dict:
        command = launcher.appcontainer_worker_command([sys.executable, "-I", "-c", _CHILD, str(secret)], files=[],
                                                       directories=[])
        process = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=sanitized_subprocess_env({}),
            **process_group.detached_spawn_kwargs(),
        )
        process_group.register_child_for_tree_kill(process)
        process.stdin.write(b"over the private pipe\n")
        await process.stdin.drain()
        line = await process.stdout.readline()
        stderr = await process.stderr.read()
        assert await process.wait() == 0, stderr.decode(errors="replace")
        return json.loads(line)

    result = asyncio.run(run())
    assert result == {"echo": "over the private pipe", "loopback": "blocked", "cwd": "denied", "home": "denied",
                      "ungranted": "denied"}
