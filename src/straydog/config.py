"""Server settings: one schema for command-line options, STRAYDOG_* variables and the config file.

Precedence: options, then environment variables, then the config file, then defaults.
Settings left unset (None or empty) are derived by :func:`resolve` from the others and
from the machine: the model bundle's quantization, the packaged native build, the
prefill batch and the host memory the context's state needs.
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from straydog.bundles import QUANTIZATIONS, model_lock

ENV_PREFIX = "STRAYDOG_"
CONFIG_ENV = "STRAYDOG_CONFIG"
MODEL_NAME = "Qwen3.8-Flash-Next"


def straydog_home() -> Path:
    """Where the config, models, drafter and native builds live by default."""
    return Path.home() / ".straydog"


def default_config_path() -> Path:
    return straydog_home() / "config.yaml"


def _option(default: Any, *flags: str, help: str, metavar: str | None = None, **constraints: Any) -> Any:
    return Field(default, description=help, json_schema_extra={"flags": list(flags), "metavar": metavar},
                 **constraints)


class Settings(BaseModel):
    """Every setting of ``straydog serve``; the field name is the option's long name with underscores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = _option("", "-m", "--model", metavar="DIR",
                         help="model bundle directory (the pinned GGUF shards and tokenizer files)")
    quant: Literal["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"] | None = _option(
        None, "--quant", metavar="NAME", help="the bundle's quantization (default: detected from its files)")
    model_draft: str = _option("", "-md", "--model-draft", metavar="DIR", help="MTP drafter directory")
    draft_max: int | None = _option(None, "--draft-max", metavar="N", ge=0, le=3,
                                    help="MTP drafts per pass, 0 turns speculation off (default: 3 with a drafter)")
    ctx_size: int = _option(262144, "-c", "--ctx-size", metavar="N", ge=1, le=262144, help="context in tokens")
    batch_size: int | None = _option(None, "-b", "--batch-size", metavar="N", ge=1, le=4096,
                                     help="prefill batch (default: 1024, 4096 with 48 GiB or more of GPU budget)")
    device: list[int] = _option([0], "-dev", "--device", metavar="LIST", min_length=1, max_length=4,
                                help="GPU ordinals, comma separated")
    split_mode: Literal["layer", "experts"] = _option(
        "experts", "-sm", "--split-mode", metavar="MODE",
        help="with two GPUs holding the whole model: experts splits every layer's experts, layer splits the layers")
    vram: float = _option(6, "--vram", metavar="GIB", ge=0.01, le=256, help="VRAM budget per GPU in GiB")
    state_ram: float | None = _option(None, "--state-ram", metavar="GIB", ge=0.01, le=256,
                                      help="host memory for sequence state in GiB (default: what the context needs)")
    gemm: Literal["fp32", "fp16"] = _option("fp16", "--gemm", metavar="PRECISION",
                                            help="input precision of large matrix products (fp32 or fp16)")
    gemm_min_batch: int = _option(2, "--gemm-min-batch", metavar="N", ge=2, le=257,
                                  help="smallest batch computed with GEMM")
    host: str = _option("127.0.0.1", "--host", metavar="ADDRESS", help="listen address")
    port: int = _option(8080, "--port", metavar="PORT", ge=1, le=65535, help="listen port")
    api_key: SecretStr | None = _option(None, "--api-key", metavar="KEY",
                                        help="Bearer key the server then requires (default: none required)")
    api_key_file: str = _option("", "--api-key-file", metavar="FILE",
                                help="file holding the API key, kept out of the process list")
    alias: str = _option("", "--alias", metavar="NAME", help="model id in /v1/models and answers")
    temp: float = _option(0, "--temp", metavar="T", ge=0, le=2, help="temperature when a request gives none")
    seed: int = _option(0, "--seed", metavar="N", ge=0, help="sampling seed when a request gives none")
    inspector: str = _option("", "--inspector", metavar="PATH", help="GGUF inspector (default: the packaged build)")
    gpu_library: str = _option("", "--gpu-library", metavar="PATH",
                               help="GPU library (default: the packaged build for this machine)")
    startup_timeout: float = _option(180, "--startup-timeout", metavar="S", gt=0, le=1800,
                                     help="seconds the engine may take to load")
    log_file: str = _option("", "--log-file", metavar="FILE", help="log file (default: stderr)")
    verbose: bool = _option(False, "-v", "--verbose", help="log the worker's own output")

    @field_validator("device", mode="before")
    @classmethod
    def _device_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, int) and not isinstance(value, bool):
            return [value]
        return value

    @field_validator("device")
    @classmethod
    def _distinct_devices(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value) or min(value) < 0:
            raise ValueError("GPU ordinals must be distinct and nonnegative")
        return value

    @property
    def speculative(self) -> int:
        return (self.draft_max or 0) if self.model_draft else 0

    @property
    def model_id(self) -> str:
        return self.alias or f"{MODEL_NAME}-{self.quant}"


def option_flags(name: str) -> list[str]:
    extra = Settings.model_fields[name].json_schema_extra
    flags = extra.get("flags") if isinstance(extra, dict) else None
    return [str(flag) for flag in flags] if isinstance(flags, list) else []


def add_options(parser: argparse.ArgumentParser) -> None:
    """Add every setting, and ``--config``, as an option; only options given reach the namespace."""
    for name, field in Settings.model_fields.items():
        extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
        kwargs: dict[str, Any] = {"dest": name, "default": argparse.SUPPRESS, "help": field.description}
        if field.annotation is bool:
            kwargs["action"] = "store_true"
        else:
            kwargs["metavar"] = extra.get("metavar")
            kinds = [kind for kind in get_args(field.annotation) if kind is not type(None)] or [field.annotation]
            if get_origin(kinds[0]) is Literal:
                kwargs["choices"] = get_args(kinds[0])
        parser.add_argument(*option_flags(name), **kwargs)
    parser.add_argument("--config", type=Path, metavar="FILE", default=argparse.SUPPRESS,
                        help="settings file (default: ~/.straydog/config.yaml when present)")


def given_options(namespace: argparse.Namespace) -> tuple[dict[str, Any], Path | None]:
    """The settings given as options, and the ``--config`` file."""
    values = {name: value for name, value in vars(namespace).items() if name in Settings.model_fields}
    return values, getattr(namespace, "config", None)


def env_name(name: str) -> str:
    return ENV_PREFIX + name.upper()


def load_settings(
    options: Mapping[str, Any] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    config_file: Path | None = None,
) -> tuple[Settings, dict[str, str]]:
    """Merge the three sources; returns the settings and where each given value came from.

    ``options`` holds only the options given on the command line. The config file is
    ``config_file``, else ``$STRAYDOG_CONFIG``, else ``~/.straydog/config.yaml`` when it exists.
    """
    environ = os.environ if environ is None else environ
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    path = config_file or (Path(environ[CONFIG_ENV]) if environ.get(CONFIG_ENV) else None)
    if path is None and default_config_path().is_file():
        path = default_config_path()
    if path is not None:
        path = path.expanduser()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must hold a mapping of settings")
        unknown = sorted(set(map(str, data)) - set(Settings.model_fields))
        if unknown:
            raise ValueError(f"{path}: unknown settings {', '.join(unknown)}")
        values.update(data)
        sources.update({key: f"config {path}" for key in data})
    for name in Settings.model_fields:
        if env_name(name) in environ:
            values[name] = environ[env_name(name)]
            sources[name] = f"env {env_name(name)}"
    for name, value in (options or {}).items():
        if name not in Settings.model_fields:
            raise ValueError(f"unknown setting {name}")
        values[name] = value
        sources[name] = "option " + option_flags(name)[-1]
    return Settings.model_validate(values), sources


def detect_quantization(model: Path) -> str:
    """The one quantization whose pinned shards are all in ``model``."""
    found = [quant for quant in QUANTIZATIONS
             if all((model / asset["name"]).is_file() for asset in model_lock(quant)["assets"])]
    if len(found) != 1:
        what = "no pinned model bundle" if not found else "more than one bundle (" + ", ".join(found) + ")"
        raise ValueError(f"{model} holds {what}; give --quant")
    return found[0]


def resolve(settings: Settings, *, native: tuple[Path, Path] | None = None) -> Settings:
    """Fill every derived setting; ``native`` is the packaged (inspector, GPU library) when not given."""
    if not settings.model:
        raise ValueError("no model: give -m DIR or set model in the config file")
    model = Path(settings.model).expanduser().resolve()
    if not model.is_dir():
        raise ValueError(f"model directory not found: {model}")
    updates: dict[str, Any] = {"model": str(model)}
    quant = settings.quant or detect_quantization(model)
    updates["quant"] = quant
    if settings.model_draft:
        updates["model_draft"] = str(Path(settings.model_draft).expanduser().resolve())
    updates["draft_max"] = settings.draft_max if settings.draft_max is not None else (3 if settings.model_draft else 0)
    if settings.batch_size is None:
        updates["batch_size"] = 4096 if len(settings.device) * settings.vram >= 48 else 1024
    if settings.state_ram is None:
        from straydog.model import context_state_bytes

        # Enough for the allocated context's state (K/V stays on the host when it exceeds a
        # quarter of the GPU budget), rounded up to half a GiB.
        updates["state_ram"] = max(1.0, math.ceil(context_state_bytes(settings.ctx_size) / 2**30 * 2) / 2)
    if not settings.inspector or not settings.gpu_library:
        if native is None:
            from straydog.provisioning import packaged_build

            packaged = packaged_build()
            if packaged is None:
                raise ValueError("no packaged native build for this machine: build one (straydog bundle),"
                                 " or give --inspector and --gpu-library")
            native = packaged[0], packaged[1]
        updates["inspector"] = settings.inspector or str(native[0])
        updates["gpu_library"] = settings.gpu_library or str(native[1])
    for name in ("inspector", "gpu_library", "api_key_file", "log_file"):
        value = updates.get(name, getattr(settings, name))
        if value:
            updates[name] = str(Path(value).expanduser().resolve())
    if settings.api_key is not None and settings.api_key_file:
        raise ValueError("give either --api-key or --api-key-file, not both")
    updates["alias"] = settings.alias or f"{MODEL_NAME}-{quant}"
    return settings.model_copy(update=updates)


def api_key(settings: Settings) -> str | None:
    """The key the server requires, or None when none was given."""
    if settings.api_key is not None:
        key = settings.api_key.get_secret_value()
    elif settings.api_key_file:
        key = Path(settings.api_key_file).read_text(encoding="utf-8").strip()
    else:
        return None
    if not key:
        raise ValueError("the API key is empty")
    return key


def describe(settings: Settings, sources: Mapping[str, str], given: Settings | None = None) -> list[str]:
    """One line per setting for ``--print-config``: its value and where it came from.

    ``given`` is the settings before :func:`resolve`; a value resolve filled in is marked derived.
    """
    lines = []
    for name in Settings.model_fields:
        value = getattr(settings, name)
        source = sources.get(name) or ("derived" if given is not None and getattr(given, name) != value else "default")
        if name == "api_key" and value is not None:
            value = "(set)"
        elif isinstance(value, list):
            value = ",".join(map(str, value))
        lines.append(f"{name:<16} {'' if value is None else value}  [{source}]")
    return lines
