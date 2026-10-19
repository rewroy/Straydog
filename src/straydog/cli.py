"""The ``straydog`` command."""

from __future__ import annotations

import sys

_PROVISIONING = ("build", "bundle", "fetch", "verify", "fetch-mtp")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in _PROVISIONING:
        from straydog.provisioning import main as provisioning

        return provisioning(argv)
    print("usage: straydog {" + ",".join(_PROVISIONING) + "} ...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
