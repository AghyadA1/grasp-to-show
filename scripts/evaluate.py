"""Evaluate the pipeline (grasp2show/pipeline.py) on held-out scenes, and train its chooser.

Objects: the real scans with something to hook (object_set_hookreal.json), the v2 scans (mostly
without a handle: the hook is no option there) and --gen held-out generated objects (every kind
of handle). Each gets --scenes scenes (pipeline.sample_scene: start pose, scanner pose, friction,
weight, and a small error in the object's pose that the robot does not know). In each scene both
strategies run their checks (the chooser's inputs) and both are executed (its targets), each
with two rankings of its candidates ("score": the checks' margins; "robust": also each
candidate's robustness, the share of its small shifts that still hold), the first three of each.

Stages (all by default; resumable: run the same command again):
  prepare   per object, cached: pinch grasps and their robustness (data/grasps_choose), the hook
            network's hooks (data/hooks_net_<tag>, pipeline.test_hooks);
  episodes  <run>/records.jsonl, one line per scene;
  fit       the chooser cross-validated by object (5 folds: no object in both training and test)
            against always-pinch, always-hook, the rule and the oracle; overall, on real and
            generated objects, by friction and weight; then trained on everything:
            <run>/chooser.pt and data/models/chooser.pt (what the pipeline loads).

Every run has its own folder, data/runs/<date>_<name>/ (log, records, results, figures, chooser).

Usage:  python scripts/evaluate.py --workers 4                       (data/runs/<today>_chooser/)
        python scripts/evaluate.py --name heavy --scenes 10
        python scripts/evaluate.py --run data/runs/2026-10-31_chooser --stage fit   (refit / replot)
        python scripts/evaluate.py --smoke                            (4 objects, 2 scenes)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("G2S_OBJECT_SET", "v2")
from grasp2show.chooser import Chooser, rule_choice, vector
from grasp2show.hanging import LIFT_VERSION
from grasp2show.parallel import Workers
from grasp2show.pipeline import GRASP_DIR, SCANNER_SCALE, crc, hook_dir

RUNS = ROOT / "data" / "runs"
TOP = 3              # candidates executed per strategy, ranking and scene (the first is the one the robot runs)
VARIANTS = ("score", "robust")


# ---------------------------------------------------------------- objects

def objects(n_gen: int, smoke: bool) -> dict[str, str]:
    """name -> group ("real" / "gen")."""
    real = json.loads((ROOT / "grasp2show" / "object_set_hookreal.json").read_text())["objects"]
    v2 = json.loads((ROOT / "grasp2show" / "object_set_v2.json").read_text())["objects"]
    g = json.loads((ROOT / "grasp2show" / "object_set_gen.json").read_text())
    test = set(g["test"])
    kind = lambda o: ("container-" if o["body"].get("style") == "container" else "") + o["handle"]["where"]
    by = defaultdict(list)
    for o in g["details"]:
        if o["name"] in test:
            by[kind(o)].append(o["name"])
    rng = np.random.default_rng(0)
    gen, kinds = [], sorted(by)
    for i, k in enumerate(kinds):                               # about the same number of each kind
        want = n_gen // len(kinds) + (i < n_gen % len(kinds))
        gen += list(rng.choice(by[k], min(want, len(by[k])), replace=False))
    if smoke:
        from grasp2show.objects import GENERATED_DIR

        real = ["ACE_Coffee_Mug_Kristen_16_oz_cup", "Threshold_Porcelain_Teapot_White"]
        v2 = ["Cole_Hardware_Saucer_Glazed_6"]
        gen = [n for n in gen if (GENERATED_DIR / n).exists()][:1]
    out = {n: "real" for n in list(dict.fromkeys(real + v2))}
    out.update({n: "gen" for n in gen})
    return out




def grasp_job(name: str):
    from grasp2show.pipeline import label_grasps, pinch_robustness_of

    t0 = time.time()
    try:
        label_grasps(name)
        pinch_robustness_of(name)
    except Exception as e:  # noqa: BLE001 (an object that does not load: left out, say why)
        return name, f"{type(e).__name__}: {e}", time.time() - t0
    return name, "", time.time() - t0


def hook_job(args):
    name, cands, hooks_dir = args
    from grasp2show.hanging import load_hook_file
    from grasp2show.pipeline import test_hooks

    t0 = time.time()
    try:
        n = len(load_hook_file(test_hooks(name, cands, hooks_dir)))
    except Exception as e:  # noqa: BLE001 (an object that does not load: left out, say why)
        return name, -1, time.time() - t0, f"{type(e).__name__}: {e}"
    return name, n, time.time() - t0, ""


# ---------------------------------------------------------------- one scene

def episode(args) -> dict:
    """One scene (pipeline.sample_scene): both strategies' checks and, for both rankings, each
    strategy's first TOP candidates that can be planned, executed with the same unknown error in
    the object's pose."""
    name, k, group, scale, hooks_dir = args
    from grasp2show.pipeline import Assets, check, execute_hook, execute_pinch, sample_scene

    t0 = time.time()
    scene, off = sample_scene(name, k, scale)
    ck = check(scene, Assets.load(name, hooks_dir))
    R = ck.assets.robust
    rec = {"object": name, "scene": k, "group": group, "mu": scene.mu, "mass_scale": scene.mass_scale,
           "mass_kg": round(ck.sc.obj.mass, 3), "hook_n_all": len(ck.assets.hooks),
           "pose_error": [round(off[0] * 1000, 2), round(off[1] * 1000, 2), round(float(np.degrees(off[2])), 2)],
           **ck.features}
    for strat in ("pinch", "hook"):
        t1 = time.time()
        done: dict = {}
        for v in VARIANTS:
            if strat == "pinch":
                ranked = ck.pinch_ranked(v == "robust")
                why = "no grasp"
            else:
                ranked = ck.hook_ranked(v == "robust")
                why = ("no hook" if not ck.assets.hooks else "no hook holds at this friction" if not ck.hooks else
                       "no hook passes the checks" if not ranked else "plan")
            tries = []
            for c in ranked[:20]:
                key = ((c.grasp, c.twin, c.roll_idx) if strat == "pinch" else (c.hook, c.twin, c.pitch_deg, c.extra_deg))
                if key not in done:
                    done[key] = (execute_pinch(ck, c, off) if strat == "pinch" else execute_hook(ck, c, off))
                ok, reason = done[key]
                if ok is None:
                    why = why if tries else reason
                    continue
                r = float(R[c.grasp]) if strat == "pinch" else float(ck.hooks[c.hook].get("robust", 0.5))
                tries.append((ok, reason, r))
                if len(tries) == TOP:
                    break
            rec.update({f"{strat}_success_{v}": bool(tries and tries[0][0]),
                        f"{strat}_reason_{v}": tries[0][1] if tries else why,
                        f"{strat}_tries_{v}": [t[0] for t in tries], f"{strat}_r_{v}": tries[0][2] if tries else 0.0})
        rec[f"{strat}_s"] = round(time.time() - t1, 1)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# ---------------------------------------------------------------- fit and report

def folds(names: list[str], k: int = 5) -> dict[str, int]:
    return {n: crc("fold" + n) % k for n in names}


def evaluate(recs: list[dict], v: str = "robust", margin: float = 0.0) -> tuple[dict, np.ndarray]:
    """The chooser for ranking v, cross-validated by object, against the fixed policies."""
    X = np.stack([vector(r, v) for r in recs])
    yp = np.array([r[f"pinch_success_{v}"] for r in recs], float)
    yh = np.array([r[f"hook_success_{v}"] for r in recs], float)
    fold = folds(sorted({r["object"] for r in recs}))
    fi = np.array([fold[r["object"]] for r in recs])
    choice = np.empty(len(recs), "<U5")
    prob = np.zeros((len(recs), 2))
    for f in range(5):
        tr, te = fi != f, fi == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        ch = Chooser.fit(X[tr], yp[tr], yh[tr], margin=margin)
        prob[te] = ch.predict(X[te])
        choice[te] = ch.choose(X[te])
    rule = np.array([rule_choice(r) for r in recs])
    out = {}
    for grp in ("all", "real", "gen"):
        m = np.array([grp == "all" or r["group"] == grp for r in recs])
        if not m.any():
            continue
        succ = lambda c, m=m: float(np.mean(np.where(c[m] == "hook", yh[m], yp[m])))
        any3 = lambda key, m=m: float(np.mean([any(r.get(key, [])) for r, mm in zip(recs, m) if mm]))
        out[grp] = {"scenes": int(m.sum()), "objects": len({r["object"] for r, mm in zip(recs, m) if mm}),
                    "always_pinch": float(yp[m].mean()), "always_hook": float(yh[m].mean()),
                    "rule": succ(rule), "learned": succ(choice), "oracle": float(np.maximum(yp, yh)[m].mean()),
                    "learned_hook_share": float(np.mean(choice[m] == "hook")),
                    "pinch_any_of_top3": any3(f"pinch_tries_{v}"), "hook_any_of_top3": any3(f"hook_tries_{v}"),
                    "only_hook_works": float(np.mean((yh > yp)[m])), "only_pinch_works": float(np.mean((yp > yh)[m]))}
    return out, np.stack([prob[:, 0], prob[:, 1], choice == "hook", yp, yh], 1)


def ranking_table(recs: list[dict]) -> dict:
    """Score vs robust ranking, per strategy: success of the candidate each would run, over the
    scenes where that strategy had a candidate; and where the two picked different ones."""
    out = {}
    for s in ("pinch", "hook"):
        m = [r for r in recs if r[f"{s}_tries_score"] or r[f"{s}_tries_robust"]]
        diff = [r for r in m if r[f"{s}_r_score"] != r[f"{s}_r_robust"] or
                r[f"{s}_tries_score"] != r[f"{s}_tries_robust"]]
        out[s] = {"scenes_with_a_candidate": len(m),
                  "score": float(np.mean([r[f"{s}_success_score"] for r in m])) if m else float("nan"),
                  "robust": float(np.mean([r[f"{s}_success_robust"] for r in m])) if m else float("nan"),
                  "scenes_where_they_differ": len(diff),
                  "score_where_differ": float(np.mean([r[f"{s}_success_score"] for r in diff])) if diff else float("nan"),
                  "robust_where_differ": float(np.mean([r[f"{s}_success_robust"] for r in diff])) if diff else float("nan"),
                  "r_of_choice_score": float(np.mean([r[f"{s}_r_score"] for r in m])) if m else float("nan"),
                  "r_of_choice_robust": float(np.mean([r[f"{s}_r_robust"] for r in m])) if m else float("nan")}
    return out


def plot(recs: list[dict], res: dict, pred: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axs[0]
    pols = ("always_pinch", "always_hook", "rule", "learned", "oracle")
    cols = ("#4C78A8", "#E45756", "#B279A2", "#54A24B", "#9D9D9D")
    groups = [g for g in ("all", "real", "gen") if g in res]
    w = 0.8 / len(groups)
    for gi, g in enumerate(groups):
        for pi, (p, c) in enumerate(zip(pols, cols)):
            v = res[g][p] * 100
            ax.bar(pi + (gi - (len(groups) - 1) / 2) * w, v, w * 0.95, color=c, alpha=1.0 - 0.3 * gi)
            ax.text(pi + (gi - (len(groups) - 1) / 2) * w, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(pols)), [p.replace("_", "\n") for p in pols])
    ax.set_ylabel("episodes succeeded (%)")
    ax.set_title("Held-out objects (5-fold by object); bars: " + " / ".join(groups), fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    ax = axs[1]
    mu = np.array([r["mu"] for r in recs])
    mass = np.array([r["mass_kg"] for r in recs])
    yp, yh = pred[:, 3].astype(bool), pred[:, 4].astype(bool)
    cat = np.where(yp & yh, 0, np.where(yp, 1, np.where(yh, 2, 3)))
    for c, (lab, col) in enumerate((("both", "#9D9D9D"), ("only pinch", "#4C78A8"), ("only hook", "#E45756"),
                                    ("neither", "#E0E0E0"))):
        m = cat == c
        ax.scatter(mu[m], mass[m], s=10 if c < 3 else 5, c=col, label=f"{lab} ({m.sum()})", alpha=0.8, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("friction mu")
    ax.set_ylabel("mass (kg)")
    ax.set_title("Which strategy succeeded, by friction and weight", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    ax = axs[2]
    load = np.array([r["load"] for r in recs])
    bins = np.quantile(load, np.linspace(0, 1, 7))
    idx = np.clip(np.digitize(load, bins[1:-1]), 0, 5)
    ch = pred[:, 2].astype(bool)
    learned = np.where(ch, yh, yp)
    for y, lab, col in ((yp, "always pinch", "#4C78A8"), (yh, "always hook", "#E45756"),
                        (learned, "learned chooser", "#54A24B"), (np.maximum(yp, yh), "oracle", "#9D9D9D")):
        ax.plot([np.median(load[idx == b]) for b in range(6)], [y[idx == b].mean() * 100 for b in range(6)], "o-",
                color=col, label=lab, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("load on a pinch: m g / (2 mu F)  (>1: friction cannot hold it, even centred)")
    ax.set_ylabel("episodes succeeded (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Success as the pinch gets harder", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


# ---------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", "prepare", "episodes", "fit"), default="all")
    p.add_argument("--name", default="chooser", help="the run's folder: data/runs/<today>_<name>")
    p.add_argument("--run", type=Path, default=None, help="an existing run folder (resume, or refit with --stage fit)")
    p.add_argument("--scenes", type=int, default=6, help="scenes per object")
    p.add_argument("--gen", type=int, default=120, help="held-out generated objects")
    p.add_argument("--scanner-scale", type=float, default=SCANNER_SCALE, help="2: a 1280 x 960 scanner")
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "hook_net_table.pt",
                   help="hook network (default: learn.hooknet.DEFAULT_HOOK_NET; not imported here: the workers "
                        "would all load torch)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.scenes, args.name = 2, "smoke"
    out = args.run or RUNS / f"{time.strftime('%Y-%m-%d')}_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    hooks_dir = hook_dir(args.model.stem.replace("hook_net_", ""))
    log_f = open(out / "log.txt", "a", encoding="utf-8")  # noqa: SIM115 (open for the whole run)

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    objs = objects(args.gen, args.smoke)
    log(f"== evaluate {time.strftime('%Y-%m-%d %H:%M')} -> {out}: {sum(g == 'real' for g in objs.values())} real and "
        f"{sum(g == 'gen' for g in objs.values())} generated objects, {args.scenes} scenes each; hook network "
        f"{args.model.name} ({hooks_dir.name})")
    (out / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                 "hooks_dir": str(hooks_dir)}, indent=1))
    rec_f = out / "records.jsonl"
    if args.stage in ("all", "prepare", "episodes"):
        from self_train_hooks import download_missing

        names = download_missing(sorted(objs))
        pool = Workers(args.workers)
        try:
            if args.stage in ("all", "prepare"):
                todo = [n for n in names if not (GRASP_DIR / f"{n}.robust.npy").exists()]
                log(f"grasps and their robustness: {len(todo)} objects")
                for i, (n, r) in enumerate(pool.run(grasp_job, todo), 1):
                    log(f"  [{i}/{len(todo)}] {n}: " + ("lost" if r is None else r[1] or f"{r[2]:.0f} s"))
                todo = [n for n in names if not (hooks_dir / f"{n}.npz").exists()
                        or int(np.load(hooks_dir / f"{n}.npz").get("lift_version", 0)) < LIFT_VERSION]
                log(f"hooks: {len(todo)} objects")
                if todo:
                    import torch

                    from grasp2show.learn.hooknet import load_hook_net
                    from grasp2show.pipeline import propose_hooks

                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                    net = load_hook_net(args.model, dev)
                    jobs = []
                    for n in todo:
                        try:
                            jobs.append((n, propose_hooks(net, n, device=dev), hooks_dir))
                        except Exception as e:  # noqa: BLE001 (no cloud: no hooks; say why)
                            log(f"  {n}: no cloud ({type(e).__name__}: {e})")
                    for i, (job, r) in enumerate(pool.run(hook_job, jobs), 1):
                        log(f"  [{i}/{len(jobs)}] {job[0]}: " + ("lost" if r is None else r[3] or
                                                                 f"{r[1]} hooks ({r[2]:.0f} s)"))
            if args.stage in ("all", "episodes"):
                done = set()
                if rec_f.exists():
                    done = {(d["object"], d["scene"]) for d in map(json.loads, rec_f.read_text().splitlines())}
                ready = [n for n in names if (GRASP_DIR / f"{n}.npz").exists()]
                jobs = [(n, k, objs[n], args.scanner_scale, hooks_dir) for n in ready for k in range(args.scenes)
                        if (n, k) not in done]
                log(f"episodes: {len(jobs)} scenes to run ({len(done)} done before)")
                t0 = time.time()
                with open(rec_f, "a", encoding="utf-8") as fh:
                    for i, (job, r) in enumerate(pool.run(episode, jobs), 1):
                        if r is None:
                            log(f"  {job[0]} scene {job[1]}: lost (worker crashed or hung)")
                            continue
                        fh.write(json.dumps(r, default=float) + "\n")
                        fh.flush()
                        o_ = lambda s, v, r=r: "ok" if r[f"{s}_success_{v}"] else r[f"{s}_reason_{v}"][:18]
                        log(f"  [{i}/{len(jobs)}] {r['object'][:30]:30s} mu {r['mu']:.2f} {r['mass_kg']:.2f} kg: pinch "
                            f"{o_('pinch', 'score')} / {o_('pinch', 'robust')}, hook {o_('hook', 'score')} / "
                            f"{o_('hook', 'robust')} ({r['seconds']:.0f} s)")
                log(f"episodes done ({(time.time() - t0) / 60:.0f} min)")
        finally:
            pool.close()
    recs = [json.loads(x) for x in rec_f.read_text().splitlines()] if rec_f.exists() else []
    if len({r["object"] for r in recs}) < 5:
        log(f"{len(recs)} scenes: too few objects to cross-validate a chooser")
        return 0
    results = {"ranking": ranking_table(recs)}
    for s_, t in results["ranking"].items():
        log(f"{s_:5s} ranking, success of the candidate it would run: score {t['score']:.0%}, robust {t['robust']:.0%} "
            f"({t['scenes_with_a_candidate']} scenes; they picked differently in {t['scenes_where_they_differ']}: "
            f"score {t['score_where_differ']:.0%} vs robust {t['robust_where_differ']:.0%}; robustness of the "
            f"chosen candidate {t['r_of_choice_score']:.2f} vs {t['r_of_choice_robust']:.2f})")
    preds = {}
    for v in VARIANTS:
        results[v], preds[v] = evaluate(recs, v)
        for g, x in results[v].items():
            log(f"[{v}] {g:4s} ({x['scenes']} scenes, {x['objects']} objects): always pinch {x['always_pinch']:.0%}, "
                f"always hook {x['always_hook']:.0%}, rule {x['rule']:.0%}, learned {x['learned']:.0%} (hook chosen "
                f"{x['learned_hook_share']:.0%}), oracle {x['oracle']:.0%}; only the hook worked in "
                f"{x['only_hook_works']:.0%}, only the pinch in {x['only_pinch_works']:.0%}; any of the top 3: "
                f"pinch {x['pinch_any_of_top3']:.0%}, hook {x['hook_any_of_top3']:.0%}")
    hooks_run = [r for r in recs if r["hook_tries_robust"]]
    if hooks_run:
        results["hooks_run"] = {"n": len(hooks_run), "success": float(np.mean([r["hook_success_robust"] for r in hooks_run])),
                                "dropped": float(np.mean([r["hook_reason_robust"] == "dropped" for r in hooks_run]))}
        log(f"hooks that ran (robust ranking): {len(hooks_run)}, succeeded {results['hooks_run']['success']:.0%}, "
            f"dropped {results['hooks_run']['dropped']:.0%}")
    (out / "results.json").write_text(json.dumps(results, indent=1))
    best = max(VARIANTS, key=lambda v: results[v]["all"]["learned"])
    for v in VARIANTS:
        plot(recs, results[v], preds[v], out / f"chooser_{v}.png")
    X = np.stack([vector(r, best) for r in recs])
    ch = Chooser.fit(X, np.array([r[f"pinch_success_{best}"] for r in recs], float),
                     np.array([r[f"hook_success_{best}"] for r in recs], float))
    ch.save(out / "chooser.pt")
    if not args.smoke:
        shutil.copy(out / "chooser.pt", ROOT / "data" / "models" / "chooser.pt")
    log(f"better ranking: {best}; wrote {out / 'chooser.pt'}" + ("" if args.smoke else " and data/models/chooser.pt")
        + f", {out}/chooser_score.png, chooser_robust.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
