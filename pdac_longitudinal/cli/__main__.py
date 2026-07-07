"""`pdac_longitudinal`; single entry point dispatching to the framework's commands."""

from __future__ import annotations

import importlib
import sys

# Each name maps to pdac_longitudinal.cli.<name>; only the selected command's
# module is imported, so one command never pulls in another's heavy imports.
_COMMANDS = ("train", "evaluate", "analyze", "preprocess", "verify")


def main(argv: list | None = None) -> None:
    """Dispatch `pdac_longitudinal <command>` to the matching CLI entry point.

    Args:
        argv: Command-line args, `[command, *rest]`; defaults to `sys.argv[1:]`.

    Raises:
        SystemExit: If `argv[0]` isn't a known command (exit code 2).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = "usage: pdac_longitudinal <command> [options]\n\ncommands:\n  " + "\n  ".join(_COMMANDS)

    if not argv or argv[0] in ("-h", "--help"):
        print(usage)
        return
    command = argv[0]
    if command not in _COMMANDS:
        print(f"pdac_longitudinal: unknown command '{command}'\n\n{usage}", file=sys.stderr)
        raise SystemExit(2)
    module = importlib.import_module(f"pdac_longitudinal.cli.{command.replace('-', '_')}")
    module.main(argv[1:])


if __name__ == "__main__":
    main()
