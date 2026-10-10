#!/usr/bin/env python3
"""Every self-hosted job a pull request can trigger carries the fleet trust gate.

freya is a PUBLIC fork and the org's `Default` runner group admits public
repositories: a fork pull request reaches these persistent machines with
PR-authored code unless each lane refuses it. The fleet rule (maxi-config#1019,
ported from maxi-core#4875) is a `trust gate` FIRST STEP -- never a job-level
`if`, which guards only the workflow file as committed to the base branch and
says nothing about the copy a pull request runs (GitHub executes the workflow
from the PR's merge commit, so a fork supplies the `run:` blocks too).

The test is DERIVED, not a list: every job in .github/workflows whose
`runs-on` selects a self-hosted runner (by label, by a runner group, or by a
dynamic expression) and which is reachable from `pull_request`,
`pull_request_target`, `issue_comment` or `merge_group` must open with the
trust-gate step carrying the canonical admission predicate. A job that loses
the gate fails here, as does a gate whose `if:` has drifted from the canonical
string pinned in ci_trusted_senders-style constants below.

The expression is parsed structurally where it is cheap to do so: the gate must
be the first step, must be a `run:` step (not a `uses:`), must carry the
SENDER_LOGIN env passthrough, and its `if:` must admit push, schedule,
Dependabot (by actor), both trusted sender ids, same-repo pull_request,
workflow_dispatch, and merge_group/checks_requested -- and must NOT admit
anything else. If the canonical predicate is ever rotated fleet-wide, update it
here in the same commit.
"""

import re
import sys
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"

PR_CARRYING_TRIGGERS = ("pull_request", "pull_request_target", "issue_comment", "merge_group")

TRUSTED_SENDER_IDS = ("874012", "264883562")
ADMITTED_BOT_ACTOR = "dependabot[bot]"

# The canonical gate, spelled the way every site carries it. Keep byte-identical
# with the maxi-config#1019 template so a fleet-wide rotation is a two-line diff.
GATE_IF = (
    "${{ github.actor != 'dependabot[bot]' && !(github.event_name == 'push' "
    "|| github.event_name == 'schedule' || ((github.event.sender.id == 874012 "
    "|| github.event.sender.id == 264883562) && (github.event_name == "
    "'workflow_dispatch' || (github.event_name == 'pull_request' && "
    "github.event.pull_request.head.repo.full_name == github.repository))) "
    "|| (github.event_name == 'merge_group' && github.event.action == "
    "'checks_requested')) }}"
)


def normalise(expr: str) -> str:
    """Collapse whitespace so a refloat of the same predicate still matches."""
    return re.sub(r"\s+", " ", expr.strip())


def triggers(doc) -> set:
    on = doc.get(True)
    if on is None:
        on = doc.get("on")
    if on is None:
        return set()
    if isinstance(on, list):
        return set(on)
    if isinstance(on, dict):
        return set(on.keys())
    return set()


def resolve_runs_on(runs_on) -> str:
    """Render a runs-on value into a string we can classify."""
    if runs_on is None:
        return ""
    if isinstance(runs_on, list):
        return " ".join(str(x) for x in runs_on)
    if isinstance(runs_on, dict):
        # {group: x} / {labels: [...]} -- a group partitions self-hosted
        # capacity, and a bare label like `tier-fast` or `nas-unstick` names a
        # fleet machine, so both classify as self-hosted unless the only term
        # is a known hosted image.
        parts = []
        if isinstance(runs_on.get("labels"), list):
            parts += [str(x) for x in runs_on["labels"]]
        elif isinstance(runs_on.get("labels"), str):
            parts.append(runs_on["labels"])
        if "group" in runs_on:
            parts.append(str(runs_on["group"]))
        return " ".join(parts)
    return str(runs_on)


# GitHub-hosted image names. Everything that is not provably one of these, and
# not empty, is treated as the fleet: the failure mode of a false positive is a
# test failure on a hosted job (annoying), the failure mode of a false negative
# is untrusted code on a persistent runner (the thing this file exists to stop).
HOSTED_IMAGES = re.compile(
    r"^(ubuntu|macos|windows)-(latest|24\.04|22\.04|20\.04)(-arm|-xlarge|-large)*$"
)


def is_self_hosted(runs_on) -> bool:
    text = resolve_runs_on(runs_on)
    if not text:
        # No runs-on: defaults to ubuntu-latest, hosted.
        return False
    labels = [t.strip() for t in text.split()]
    if "${{" in text:
        # Dynamic routing. The expression may resolve either way -- the qodana
        # full-scan ternary is exactly that shape. Classify as the fleet unless
        # it provably never names a self-hosted label.
        return True
    non_hosted = [l for l in labels if HOSTED_IMAGES.match(l)]
    return len(non_hosted) < len(labels)


def gate_steps(job) -> list:
    steps = job.get("steps")
    if not isinstance(steps, list):
        return []
    return [s for s in steps if isinstance(s, dict) and s.get("name") == "trust gate"]


def job_has_uses(job) -> bool:
    return "uses" in job


def failures() -> list:
    problems = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            problems.append(f"{path.name}: unparseable YAML: {exc}")
            continue
        if not isinstance(doc, dict):
            continue
        trig = triggers(doc)
        pr_reachable = bool(trig & set(PR_CARRYING_TRIGGERS))
        for name, job in (doc.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            if job_has_uses(job):
                # A reusable-workflow call: the gate cannot be injected from
                # the caller. The callee's own guards are that lane's boundary;
                # noted in the PR that added this test rather than skipped
                # silently here.
                continue
            if not is_self_hosted(job.get("runs-on")):
                continue
            if not pr_reachable:
                continue
            gates = gate_steps(job)
            if not gates:
                problems.append(
                    f"{path.name}: self-hosted job `{name}` is reachable from "
                    f"{sorted(trig & set(PR_CARRYING_TRIGGERS))} and has no "
                    "`trust gate` step"
                )
                continue
            steps = job.get("steps")
            if not (isinstance(steps[0], dict) and steps[0].get("name") == "trust gate"):
                problems.append(
                    f"{path.name}: job `{name}` has a trust gate but not as its "
                    "FIRST step; the gate must run before checkout"
                )
            gate = gates[0]
            expr = gate.get("if") or ""
            if normalise(expr) != normalise(GATE_IF):
                problems.append(
                    f"{path.name}: job `{name}` trust-gate `if:` drifted from the "
                    f"canonical predicate.\n  got: {normalise(expr)}\n  want: {normalise(GATE_IF)}"
                )
            if "uses" in gate:
                problems.append(
                    f"{path.name}: job `{name}` trust gate is a `uses:` step; it "
                    "must be a `run:` step a fork cannot swap for a different action"
                )
            env = gate.get("env") or {}
            if "SENDER_LOGIN" not in env:
                problems.append(
                    f"{path.name}: job `{name}` trust gate lost the SENDER_LOGIN "
                    "env passthrough; the error message would print a blank sender"
                )
    return problems


def main() -> int:
    problems = failures()
    if problems:
        print("trust-gate guard FAILED:\n")
        for p in problems:
            print(f"  - {p}")
        print(
            "\nEvery self-hosted job a pull request can trigger must carry the "
            "fleet trust gate as its first step (maxi-config#1019)."
        )
        return 1
    print("trust-gate guard: every self-hosted PR-reachable job carries the gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
