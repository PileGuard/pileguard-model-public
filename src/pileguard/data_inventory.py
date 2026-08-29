"""Validate the external dataset layout without modifying source data."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_ROOT_ENV = "PILEGUARD_DATA_ROOT"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})

EXPECTED_MSU_COUNTS = {
    ("train", "positives"): 2_283,
    ("train", "negatives"): 5_279,
    ("val", "positives"): 261,
    ("val", "negatives"): 592,
    ("test", "positives"): 1_276,
    ("test", "negatives"): 1_572,
}
EXPECTED_NESTLER_ANNOTATIONS = 6
EXPECTED_PIO_COUNTS = {
    ("train", "images"): 1_035,
    ("train", "labels"): 1_035,
    ("val", "images"): 452,
    ("val", "labels"): 452,
}


@dataclass(frozen=True)
class InventoryCheck:
    """One expected dataset count and its observed value."""

    name: str
    path: str
    expected: int
    actual: int

    @property
    def ok(self) -> bool:
        return self.actual == self.expected

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def resolve_data_root(cli_value: str | None = None) -> Path:
    """Resolve the data root from a CLI value or the project environment variable."""

    value = cli_value or os.environ.get(DATA_ROOT_ENV)
    if not value:
        raise ValueError(
            f"Set {DATA_ROOT_ENV} or pass --data-root with the external data directory."
        )
    return Path(value).expanduser().resolve()


def count_files(directory: Path, suffixes: Iterable[str] | None = None) -> int:
    """Count regular files, optionally filtering by lowercase suffix."""

    if not directory.is_dir():
        return 0
    allowed = frozenset(suffixes) if suffixes is not None else None
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and (allowed is None or path.suffix.lower() in allowed)
    )


def inspect_data_root(data_root: Path) -> list[InventoryCheck]:
    """Inspect the official MSU, NESTLER, and PIO dataset inventories."""

    msu_root = (
        data_root
        / "extracted"
        / "msu_v2"
        / "MSU Poultry Piling Dataset"
        / "piling_dataset"
    )
    checks = [
        InventoryCheck(
            name=f"msu/{split}/{label}",
            path=str(msu_root / split / label),
            expected=expected,
            actual=count_files(msu_root / split / label, IMAGE_SUFFIXES),
        )
        for (split, label), expected in EXPECTED_MSU_COUNTS.items()
    ]

    nestler_root = data_root / "extracted" / "nestler_v1"
    checks.append(
        InventoryCheck(
            name="nestler/annotations",
            path=str(nestler_root),
            expected=EXPECTED_NESTLER_ANNOTATIONS,
            actual=len(list(nestler_root.rglob("annotations_*.json")))
            if nestler_root.is_dir()
            else 0,
        )
    )
    pio_root = data_root / "extracted" / "pio_v1" / "data"
    for (split, kind), expected in EXPECTED_PIO_COUNTS.items():
        directory = pio_root / kind / split
        if kind == "images":
            actual = count_files(directory, IMAGE_SUFFIXES)
        else:
            actual = (
                sum(
                    1
                    for path in directory.glob("*.txt")
                    if path.is_file() and path.name != "classes.txt"
                )
                if directory.is_dir()
                else 0
            )
        checks.append(
            InventoryCheck(
                name=f"pio/{split}/{kind}",
                path=str(directory),
                expected=expected,
                actual=actual,
            )
        )
    return checks


def format_report(data_root: Path, checks: list[InventoryCheck]) -> str:
    """Format a compact human-readable inventory report."""

    lines = [f"Data root: {data_root}"]
    for check in checks:
        state = "OK" if check.ok else "FAIL"
        lines.append(f"[{state}] {check.name}: {check.actual}/{check.expected}")
    lines.append("Inventory: PASS" if all(check.ok for check in checks) else "Inventory: FAIL")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        help=f"External data root. Defaults to the {DATA_ROOT_ENV} environment variable.",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        data_root = resolve_data_root(args.data_root)
    except ValueError as error:
        print(f"Configuration error: {error}")
        return 2

    checks = inspect_data_root(data_root)
    if args.json:
        print(
            json.dumps(
                {
                    "data_root": str(data_root),
                    "ok": all(check.ok for check in checks),
                    "checks": [check.to_dict() for check in checks],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_report(data_root, checks))
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
