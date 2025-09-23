"""Aggregate synthesis eval judgments into concise summary metrics.

Usage:
    uv run python -m tests.evals.synthesis.aggregate \
        [--run-dir eval_results/synthesis_features/2025...] \
        [--bootstrap] [--csv]

Outputs:
    - summary.json (preference rate, tie rate, per-criterion deltas)
    - (optional) deltas.csv with per-criterion mean/std and CIs when bootstrapping
    - Prints a one-line headline to stdout
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path
from typing import Any

CRITERIA = [
    "readability_flow",
    "thematic_clarity",
    "evidence_integration",
    "insightfulness",
    "contradiction_handling",
]


def _latest_run_dir(base: Path) -> Path | None:
    candidates = sorted(base.glob("*/judgments.jsonl"))
    if not candidates:
        return None
    return candidates[-1].parent


def _bootstrap_ci(values: list[float], *, iters: int = 1000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    n = len(values)
    samples: list[float] = []
    for _ in range(iters):
        resample = [values[random.randrange(n)] for _ in range(n)]
        samples.append(st.mean(resample))
    samples.sort()
    lo = samples[int(0.025 * iters)]
    hi = samples[int(0.975 * iters)]
    return (lo, hi)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "preference_rate": None,
            "tie_rate": None,
            "deltas": {},
        }

    prefs = [r.get("judgment", {}).get("preference", "") for r in records]
    ties = sum(1 for p in prefs if p == "TIE")
    n_non_tie = n - ties
    treatment_pref = sum(1 for p in prefs if p == "TREATMENT")
    preference_rate = (treatment_pref / n_non_tie) if n_non_tie > 0 else None
    tie_rate = ties / n

    deltas: dict[str, list[float]] = {c: [] for c in CRITERIA}
    for r in records:
        j = r.get("judgment", {})
        sc = j.get("scores_control", {})
        stx = j.get("scores_treatment", {})
        for c in CRITERIA:
            c_val = sc.get(c)
            t_val = stx.get(c)
            if isinstance(c_val, int) and isinstance(t_val, int):
                deltas[c].append(float(t_val - c_val))

    delta_stats: dict[str, dict[str, float]] = {}
    for c, vals in deltas.items():
        if not vals:
            delta_stats[c] = {"mean": 0.0, "std": 0.0}
            continue
        mean = st.mean(vals)
        std = st.pstdev(vals) if len(vals) > 1 else 0.0
        delta_stats[c] = {"mean": mean, "std": std}

    return {
        "n": n,
        "preference_rate": preference_rate,
        "tie_rate": tie_rate,
        "deltas": delta_stats,
    }


def aggregate(
    run_dir: Path | None = None,
    *,
    bootstrap: bool = False,
    csv: bool = False,
) -> Path:
    base = Path("eval_results/synthesis_features")
    if run_dir is None:
        run_dir = _latest_run_dir(base)
        if run_dir is None:
            raise FileNotFoundError("No eval runs found under eval_results/synthesis_features")

    judgments_path = run_dir / "judgments.jsonl"
    if not judgments_path.exists():
        raise FileNotFoundError(f"Missing judgments file: {judgments_path}")

    records = _read_jsonl(judgments_path)
    summary = _compute_metrics(records)

    if bootstrap and summary["n"] > 1:
        # Compute bootstrap CIs per criterion delta
        by_crit: dict[str, list[float]] = {c: [] for c in CRITERIA}
        for r in records:
            j = r.get("judgment", {})
            sc = j.get("scores_control", {})
            stx = j.get("scores_treatment", {})
            for c in CRITERIA:
                c_val = sc.get(c)
                t_val = stx.get(c)
                if isinstance(c_val, int) and isinstance(t_val, int):
                    by_crit[c].append(float(t_val - c_val))
        for c, vals in by_crit.items():
            lo, hi = _bootstrap_ci(vals) if vals else (0.0, 0.0)
            summary["deltas"].setdefault(c, {})
            summary["deltas"][c]["ci95_lo"] = lo
            summary["deltas"][c]["ci95_hi"] = hi

    out_json = run_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if csv:
        lines = ["criterion,mean,std"]
        for c in CRITERIA:
            d = summary["deltas"].get(c, {"mean": 0.0, "std": 0.0})
            lines.append(f"{c},{d.get('mean', 0.0):.4f},{d.get('std', 0.0):.4f}")
        (run_dir / "deltas.csv").write_text("\n".join(lines), encoding="utf-8")

    # Console headline
    pref = summary["preference_rate"]
    tie = summary["tie_rate"]
    rf = summary["deltas"]["readability_flow"]["mean"]
    tc = summary["deltas"]["thematic_clarity"]["mean"]
    ei = summary["deltas"]["evidence_integration"]["mean"]
    print(
        f"Treatment preferred: {pref if pref is not None else 'n/a'}; "
        f"ties={tie:.2f}; RF {rf:+.2f}, TC {tc:+.2f}, EI {ei:+.2f}"
    )

    return out_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate synthesis eval judgments")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory (defaults to latest under eval_results/synthesis_features)",
    )
    parser.add_argument(
        "--bootstrap", action="store_true", help="Compute 95% CIs via bootstrapping"
    )
    parser.add_argument("--csv", action="store_true", help="Also write per-criterion deltas.csv")
    args = parser.parse_args()

    aggregate(args.run_dir, bootstrap=args.bootstrap, csv=args.csv)


if __name__ == "__main__":
    main()
