#!/usr/bin/env python3
"""Check kette examples.

Two modes, per file:

- assert mode (default): the file's `Test.assert(actual, expected)` calls
  are checked. The kette test runner (test_runner.ktt) is prepended and
  `Test.summary()` appended; the file passes when the process exits 0 and
  the summary reports zero failures.
- expect mode: a file with an `// expect:` block runs standalone and its
  printed output must match the block exactly (one comment line per
  output line, ended by the first non-comment line).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def expectations(path: Path) -> list[str]:
    expected = []
    in_block = False
    for line in path.read_text().splitlines():
        if not in_block:
            if line.strip() == "// expect:":
                in_block = True
            continue
        if line.startswith("//"):
            expected.append(line[2:].removeprefix(" "))
            continue
        break
    return expected


def run(root: Path, path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["cargo", "run", "-q", "-p", "vm", "--", str(path)],
        capture_output=True,
        text=True,
        cwd=root,
    )


def check_expect(root: Path, f: Path, expected: list[str]) -> tuple[bool, str]:
    proc = run(root, f)
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
        return False, f"exited {proc.returncode}: {detail}"
    actual = proc.stdout.splitlines()
    if actual == expected:
        return True, f"{len(expected)} output lines"
    lines = []
    for i in range(max(len(expected), len(actual))):
        want = expected[i] if i < len(expected) else "<missing>"
        got = actual[i] if i < len(actual) else "<missing>"
        if want != got:
            lines.append(f"expected {want!r}, got {got!r}")
    return False, "; ".join(lines)


def check_asserts(root: Path, f: Path, runner: str) -> tuple[bool, str]:
    source = runner + "\n" + f.read_text() + "\nTest.summary()\n"
    fd, tmp = tempfile.mkstemp(suffix=".ktt", text=True)
    try:
        with os.fdopen(fd, "w") as h:
            h.write(source)
        proc = run(root, Path(tmp))
    finally:
        os.unlink(tmp)
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
        return False, f"exited {proc.returncode}: {detail}"
    lines = proc.stdout.splitlines()
    if "failed" in lines:
        where = lines[lines.index("failed") + 1] if len(lines) > 1 else "?"
        total = lines[-1] if lines else "?"
        return False, f"{where} of {total} asserts failed:\n" + "\n".join(
            "      " + l for l in lines[: lines.index("failed")]
        )
    if "passed" not in lines:
        return False, f"no summary in output: {lines!r}"
    count = lines[-1] if lines else "0"
    return True, f"{count} asserts"


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent
    files = [Path(a) for a in sys.argv[1:]] or sorted(here.glob("*.ktt"))
    runner = (here / "test_runner.ktt").read_text()

    checked = 0
    failures = 0
    for f in files:
        if f.name == "test_runner.ktt":
            continue
        expected = expectations(f)
        checked += 1
        if expected:
            ok, detail = check_expect(root, f, expected)
            mode = "expect"
        else:
            ok, detail = check_asserts(root, f, runner)
            mode = "assert"
        if ok:
            print(f"ok    {f.name} [{mode}] {detail}")
        else:
            print(f"FAIL  {f.name} [{mode}]")
            print("      " + detail)
            failures += 1

    if failures:
        print(f"{failures} of {checked} failed")
        sys.exit(1)
    print(f"{checked} checked, all ok")


if __name__ == "__main__":
    main()
