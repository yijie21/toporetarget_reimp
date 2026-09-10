#!/usr/bin/env python3
"""Render RESULTS.md from the JSON produced by the evaluation scripts.

Every number in this repository's documentation is generated here.  Nothing is
typed by hand, so the tables cannot drift away from the code that produced them.

    python scripts/make_results.py --out RESULTS.md
"""
import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import yaml

from toporetarget.paths import CONFIGS

# The paper publishes relative improvements over the baseline average, not
# absolute millimetres, so those are the quantities that can be set side by side.
PAPER = {"E_prec_vs_baseline_avg_pct": -55.0, "D_pen_vs_baseline_avg_pct": -92.0}
BASELINES = ["dexpilot", "mink"]
ABLATIONS = ["ours:no_IM", "ours:no_pen", "ours:no_bone", "ours:no_reg"]
PRETTY = {"ours": "**ours**", "dexpilot": "DexPilot", "mink": "Mink",
          "ours:no_IM": "ours, no `E_IM`", "ours:no_pen": "ours, no `E_pen`",
          "ours:no_bone": "ours, no `E_bone`", "ours:no_reg": "ours, no `E_reg`"}


def git(*args, default="unknown"):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return default


def table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(new, old):
    return None if not old else 100.0 * (new - old) / old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contactpose", default="runs/contactpose.json")
    ap.add_argument("--sequence", default="runs/grab_sequence.json")
    ap.add_argument("--tuning-dir", default="runs")
    ap.add_argument("--out", default="RESULTS.md")
    ap.add_argument("--readme", default="README.md",
                    help="headline table is injected between the RESULTS markers")
    args = ap.parse_args()

    cp = json.loads(Path(args.contactpose).read_text())
    summary, rows = cp["summary"], cp["rows"]
    eval_set = yaml.safe_load(open(CONFIGS / "eval_set.yaml"))

    L = []
    L.append("# Results\n")
    L.append("**Generated** by `scripts/make_results.py` — do not edit by hand.\n")
    L.append(table([[date.today().isoformat(), git("rev-parse", "--short", "HEAD"),
                     git("rev-parse", "--abbrev-ref", "HEAD")]],
                   ["date", "commit", "branch"]))
    L.append("")
    L.append(f"Evaluation set: ContactPose participant `{eval_set['participant']}`, "
             f"intent `{eval_set['intent']}`, right hand, "
             f"**{eval_set['n_objects']} grasps**. Hyper-parameters for every method "
             f"were fitted on a *different participant* "
             f"(`{yaml.safe_load(open(CONFIGS / 'tuning_split.yaml'))['participant']}`) "
             f"and then frozen.\n")

    for subset, title in (("all", f"All {eval_set['n_objects']} grasps"),
                          ("paper_comparable",
                           f"Paper-comparable subset "
                           f"({eval_set['paper_comparable']['n']} grasps: "
                           f"{', '.join(eval_set['paper_comparable']['excluded'])} removed)")):
        L.append(f"## Main table — {title}\n")
        body = []
        for m in ["ours", *BASELINES]:
            v = summary.get(f"{m}|{subset}")
            if v:
                body.append([PRETTY[m], v["n"], f"{v['E_prec_mm']:.2f}",
                             f"{v['E_prec_median_mm']:.2f}",
                             f"{v['D_pen_max_mean_mm']:.2f}",
                             f"{v['D_pen_max_mm']:.2f}", f"{v['seconds_mean']:.2f}"])
        L.append(table(body, ["method", "n", "E_prec mean (mm)", "E_prec median (mm)",
                              "D_pen_max mean (mm)", "D_pen_max worst (mm)",
                              "solve time (s)"]))
        L.append("")

        ours = summary.get(f"ours|{subset}")
        base = [summary.get(f"{b}|{subset}") for b in BASELINES]
        base = [b for b in base if b]
        if ours and base:
            e_avg = sum(b["E_prec_mm"] for b in base) / len(base)
            d_avg = sum(b["D_pen_max_mean_mm"] for b in base) / len(base)
            L.append("Against the average of the baselines, next to what the paper "
                     "reports. Two caveats on this comparison: the paper publishes "
                     "relative improvements rather than absolute millimetres, so "
                     "only relative figures can be set side by side; and its "
                     "average is over four baselines (OmniRetarget, DexPilot, "
                     "Mink, GeoRT) while ours is over the two we reproduce, so the "
                     "denominators differ.\n")
            L.append(table(
                [["contact precision error",
                  f"{pct(ours['E_prec_mm'], e_avg):+.1f} %",
                  f"{PAPER['E_prec_vs_baseline_avg_pct']:+.1f} %"],
                 ["max penetration depth",
                  f"{pct(ours['D_pen_max_mean_mm'], d_avg):+.1f} %",
                  f"{PAPER['D_pen_vs_baseline_avg_pct']:+.1f} %"]],
                ["quantity", "this reimplementation", "paper (arXiv:2606.16272)"]))
            L.append("")

    L.append("## Ablations\n")
    L.append("Each row zeroes one weight and leaves the rest frozen.\n")
    body = []
    for m in ["ours", *ABLATIONS]:
        v = summary.get(f"{m}|all")
        if v:
            body.append([PRETTY[m], f"{v['E_prec_mm']:.2f}",
                         f"{v['D_pen_max_mean_mm']:.2f}", f"{v['D_pen_max_mm']:.2f}"])
    L.append(table(body, ["variant", "E_prec mean (mm)", "D_pen_max mean (mm)",
                          "D_pen_max worst (mm)"]))
    L.append("")

    seq_path = Path(args.sequence)
    if seq_path.exists():
        seq = json.loads(seq_path.read_text())
        L.append("## Sequence experiment — what `E_reg` actually buys\n")
        L.append(f"ContactPose grasps are static, so the main table cannot test a "
                 f"temporal term. This is GRAB `{seq['sequence']}` "
                 f"(`{seq['object']}`, {seq['framerate']:.0f} Hz, stride "
                 f"{seq['stride']}, {len(seq['frames'])} in-contact frames), each "
                 f"frame warm-started from the previous one. Jitter is the mean "
                 f"magnitude of the trajectory's second difference.\n")
        body = []
        for name, v in seq["results"].items():
            body.append([PRETTY.get(name, name), f"{v['E_prec_mm']:.2f}",
                         f"{v['D_pen_max_over_t_mm']:.2f}",
                         f"{v['joint_jitter_rad']:.5f}",
                         f"{v['keypoint_jitter_mm']:.3f}"])
        body.append(["*human demonstration*", "—", "—", "—",
                     f"{seq['human_keypoint_jitter_mm']:.3f}"])
        L.append(table(body, ["method", "E_prec mean (mm)", "max_t D_pen (mm)",
                              "joint jitter (rad)", "keypoint jitter (mm)"]))
        L.append("")

    L.append("## Hyper-parameter search\n")
    L.append("One protocol for every method: the same held-out participant, the "
             "same objective `mean(E_prec_mm + D_pen_max_mm)`, and the same budget "
             "of sampled configurations.\n")
    body = []
    for m in ["ours", *BASELINES]:
        p = Path(args.tuning_dir) / f"tuning_{m}.json"
        if p.exists():
            t = json.loads(p.read_text())
            params = ", ".join(f"`{k}`={v:.4g}" for k, v in t["best"]["params"].items())
            body.append([PRETTY[m], t["budget"], f"{t['best']['objective']:.3f}", params])
    L.append(table(body, ["method", "budget", "best objective (mm)",
                          "frozen parameters"]))
    L.append("")

    L.append("## Per-grasp detail\n")
    objs = sorted({r["object"] for r in rows})
    lut = {(r["object"], r["method"]): r for r in rows}
    header = ["object", "solidity", "watertight"]
    for m in ["ours", *BASELINES]:
        header += [f"{m} E_prec", f"{m} D_pen"]
    body = []
    for o in objs:
        st = eval_set["object_stats"][o]
        line = [o + ("" if o not in eval_set["paper_comparable"]["excluded"] else " *"),
                f"{st['solidity']:.3f}", "yes" if st["watertight"] else "no"]
        for m in ["ours", *BASELINES]:
            r = lut.get((o, m))
            line += [f"{r['E_prec_mm']:.2f}" if r else "—",
                     f"{r['D_pen_max_mm']:.2f}" if r else "—"]
        body.append(line)
    L.append(table(body, header))
    L.append("\n`*` excluded from the paper-comparable subset (three lowest solidity).\n")

    Path(args.out).write_text("\n".join(L))
    print(f"wrote {args.out} ({len(L)} blocks)")
    inject_readme(Path(args.readme), summary, eval_set)


BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"


def inject_readme(path, summary, eval_set):
    """Replace the README's headline table so it can never drift from the data."""
    if not path.exists():
        return
    text = path.read_text()
    if BEGIN not in text or END not in text:
        print(f"{path}: no RESULTS markers, skipped")
        return
    body = []
    for m in ["ours", *BASELINES]:
        v = summary.get(f"{m}|all")
        if v:
            body.append([PRETTY[m], f"{v['E_prec_mm']:.2f}",
                         f"{v['D_pen_max_mean_mm']:.2f}", f"{v['seconds_mean']:.2f}"])
    block = "\n".join([
        BEGIN,
        f"ContactPose `{eval_set['participant']}_{eval_set['intent']}`, "
        f"all {eval_set['n_objects']} grasps, one frozen parameter set per method:",
        "",
        table(body, ["method", "contact precision error (mm)",
                     "max penetration (mm)", "solve time (s)"]),
        "",
        f"Generated {date.today().isoformat()} at commit "
        f"`{git('rev-parse', '--short', 'HEAD')}` — see "
        f"[RESULTS.md](RESULTS.md) for ablations, per-grasp detail and the "
        f"comparison with the paper's reported improvements.",
        END])
    head, _, rest = text.partition(BEGIN)
    _, _, tail = rest.partition(END)
    path.write_text(head + block + tail)
    print(f"injected the headline table into {path}")


if __name__ == "__main__":
    main()
