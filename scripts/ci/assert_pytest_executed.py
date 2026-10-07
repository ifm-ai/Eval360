#!/usr/bin/env python3
"""Assert a pytest run actually executed tests, by reading its JUnit XML.

WHY THIS EXISTS. A pytest exit status of 0 means "nothing failed", which is not
the same as "something passed". A run that collected nothing, or skipped
everything, exits 0 and reports green.

The Slurm cluster suite is exactly the shape that gets bitten by this.
`tests/slurm/conftest.py` skips the whole suite when `sbatch` is not on PATH —
correct on a laptop, where `pytest tests/` should not fail because Docker is not
running. But it means the difference between "the cluster came up and every
test passed" and "the cluster never came up, everything skipped" is invisible
in the exit status. Without a floor, breaking the cluster bring-up would turn
this gate green rather than red, which is the precise failure a cluster gate
exists to prevent.

WHAT "EXECUTED" MEANS: collected, minus skipped. A skipped test proves nothing,
so it does not count toward the floor. Failures and errors DO count as executed
— reporting them is the pytest exit status's job, and counting them here as
"not executed" would make a failing run look like an empty one.

THE FLOOR IS A MINIMUM WITH MARGIN, not an exact count: it should survive adding
a test and fail on losing a suite. Raising it when the suite grows, or lowering
it when a skip becomes permanent, is a visible line in a diff — which is the
point of writing it down rather than inferring it.

    python scripts/ci/assert_pytest_executed.py REPORT.xml --min 90 [--label NAME]
    python scripts/ci/assert_pytest_executed.py --self-test
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def _suites(root: ET.Element) -> list[ET.Element]:
    # pytest emits <testsuites><testsuite>… on current versions and a bare
    # <testsuite> on older ones. Handle both rather than depending on which
    # pytest the runner resolved.
    if root.tag == "testsuite":
        return [root]
    return list(root.iter("testsuite"))


def check(report: Path, minimum: int, label: str) -> int:
    if not report.is_file():
        # An absent report is a FAILURE, not a pass. If the report is missing,
        # the run did not get far enough to write one — exactly the case a bare
        # exit status hides.
        print(
            f"FAIL [{label}]: no JUnit report at {report}. The run did not get "
            "far enough to write one.",
            file=sys.stderr,
        )
        return 1

    try:
        root = ET.parse(report).getroot()
    except ET.ParseError as error:
        print(f"FAIL [{label}]: unreadable JUnit report {report}: {error}",
              file=sys.stderr)
        return 1

    collected = skipped = failures = errors = 0
    for suite in _suites(root):
        collected += int(suite.get("tests", 0))
        skipped += int(suite.get("skipped", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))

    executed = collected - skipped
    summary = (
        f"collected={collected} skipped={skipped} executed={executed} "
        f"failures={failures} errors={errors} floor={minimum}"
    )

    if executed < minimum:
        print(f"FAIL [{label}]: {executed} test(s) executed, floor is "
              f"{minimum}. {summary}", file=sys.stderr)
        if skipped and executed == 0:
            print(
                "       Everything skipped. For the cluster suite this almost "
                "always means Slurm was not reachable — check that "
                "`scripts/slurm_cluster.sh up` succeeded.",
                file=sys.stderr,
            )
        return 1

    print(f"OK [{label}]: {summary}")
    return 0


def self_test() -> int:
    """Prove both directions. A gate nobody has watched go red is not tested.

    Runs in CI (about a tenth of a second) rather than being assumed, because
    the entire value of this script is its failure path.
    """
    cases = [
        ("a run with enough executed tests passes",
         '<testsuite tests="12" skipped="1" failures="0" errors="0"/>', 9, 0),
        ("a fully skipped run FAILS despite pytest exiting 0",
         '<testsuite tests="11" skipped="11" failures="0" errors="0"/>', 9, 1),
        ("an under-floor run fails",
         '<testsuite tests="5" skipped="0" failures="0" errors="0"/>', 9, 1),
        ("failures still count as executed",
         '<testsuite tests="11" skipped="0" failures="3" errors="0"/>', 9, 0),
        ("the <testsuites> wrapper is handled",
         '<testsuites><testsuite tests="11" skipped="0" failures="0" '
         'errors="0"/></testsuites>', 9, 0),
        ("an empty run fails",
         '<testsuite tests="0" skipped="0" failures="0" errors="0"/>', 1, 1),
    ]

    problems = 0
    with tempfile.TemporaryDirectory() as tmp:
        for description, xml, minimum, expected in cases:
            report = Path(tmp) / "report.xml"
            report.write_text(xml)
            actual = check(report, minimum, "self-test")
            ok = actual == expected
            problems += not ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {description}"
                  f"{'' if ok else f' (expected {expected}, got {actual})'}")

        # A missing report must fail too — the case where the run died before
        # writing anything.
        missing = Path(tmp) / "does-not-exist.xml"
        actual = check(missing, 1, "self-test")
        ok = actual == 1
        problems += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] a missing report fails")

    if problems:
        print(f"SELF-TEST FAILED: {problems} case(s) wrong", file=sys.stderr)
        return 1
    print("SELF-TEST OK: the floor fails when it should and passes when it should")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path,
                        help="path to a pytest --junitxml report")
    parser.add_argument("--min", type=int, help="minimum executed tests")
    parser.add_argument("--label", default="pytest", help="name used in messages")
    parser.add_argument("--self-test", action="store_true",
                        help="prove the check can fail, then exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.report is None or args.min is None:
        parser.error("REPORT and --min are required unless --self-test is given")
    return check(args.report, args.min, args.label)


if __name__ == "__main__":
    sys.exit(main())
