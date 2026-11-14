"""Write the command list for each phase (one command per line; every command is idempotent).

    python scripts/run_matrix.py pilot   > cmds_pilot.txt
    python scripts/run_matrix.py A       > cmds_A.txt
    python scripts/run_matrix.py B1      > cmds_B1.txt     # LR selection (needs A)
    python scripts/run_matrix.py B2      > cmds_B2.txt     # final seeds (needs B1 results)
    python scripts/run_matrix.py C       > cmds_C.txt      # e-SNLI train + eval (needs B1 for MNLI LR)
    python scripts/run_matrix.py D       > cmds_D.txt      # data-size test (needs B1)
    python scripts/run_matrix.py F       > cmds_F.txt      # benchmark

Run a list with, e.g.:  while read -r c; do eval "$c" || echo "FAILED: $c" >> failed.txt; done < cmds_A.txt
Add --synthetic to every command with:  python scripts/run_matrix.py A --synthetic
"""
import argparse
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = yaml.safe_load(open(os.environ.get("GAM_PROTOCOL", os.path.join(HERE, "configs", "protocol.yaml"))))
R = P["runs_dir"]


def adapt_dir(v):
    return f"{R}/A/{v}"


def ft_dir(phase, task, v, lr, seed, extra=""):
    return f"{R}/{phase}/{task}/{v}{extra}/lr{lr:g}_s{seed}"


def selected_lr(task, variant, raw=False):
    """Best LR on the held-out split (selection seed only). Ties -> smaller LR."""
    g = P["glue"]
    best = None
    for lr in g["lr_grid"]:
        path = os.path.join(ft_dir("B1", task, variant, lr, g["selection_seed"], "_raw" if raw else ""), "results.json")
        if not os.path.exists(path):
            sys.exit(f"missing LR-selection result: {path} (run phase B1 first)")
        r = json.load(open(path))
        if best is None or r["best_heldout"] > best[0]:
            best = (r["best_heldout"], lr)
    return best[1]


def ft_cmd(task, v, lr, seed, out, init, extra, syn):
    g = P["glue"]
    c = (f"python scripts/finetune.py --task {task} --variant {v} --lr {lr:g} --seed {seed} "
         f"--epochs {g['epochs']} --batch_size {g['batch_size']} --max_len {g['max_len']} "
         f"--gam_lr {g['gam_lr']:g} --out {out} --amp")
    if task in g["eval_every"]:
        c += f" --eval_every {g['eval_every'][task]}"
    if init:
        c += f" --init {init}"
    if task in g["record_energy_tasks"]:
        c += " --record_energy"
    return c + extra + syn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["pilot", "A", "B1", "B2", "C", "D", "F"])
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    syn = " --synthetic" if a.synthetic else ""
    A, g, out = P["adapt"], P["glue"], []

    if a.phase == "pilot":
        pl = P["pilot"]
        for v in pl["variants"]:
            d = f"{R}/pilot/A/{v}"
            out.append(f"python scripts/adapt_mlm.py --variant {v} --out {d} --steps {pl['adapt_steps']} "
                       f"--batch_size {A['batch_size']} --micro_batch {A['micro_batch']} --warmup 30 "
                       f"--eval_every 100 --seed {pl['seed']} --amp{syn}")
            out.append(ft_cmd(pl["task"], v, 2e-5, pl["seed"], f"{R}/pilot/{pl['task']}/{v}", d, "", syn))
        out.append(f"python scripts/benchmark.py --out {R}/pilot/benchmark --lengths 128,512 --iters 20 --amp")
    elif a.phase == "A":
        for v in A["variants"]:
            out.append(f"python scripts/adapt_mlm.py --variant {v} --out {adapt_dir(v)} --steps {A['steps']} "
                       f"--batch_size {A['batch_size']} --micro_batch {A['micro_batch']} --seq_len {A['seq_len']} "
                       f"--lr {A['lr']:g} --gam_lr {A['gam_lr']:g} --warmup {A['warmup']} --seed {A['seed']} --amp{syn}")
    elif a.phase == "B1":
        for t in g["tasks"]:
            for v in g["main_variants"]:
                for lr in g["lr_grid"]:
                    out.append(ft_cmd(t, v, lr, g["selection_seed"], ft_dir("B1", t, v, lr, g["selection_seed"]),
                                      adapt_dir(v), "", syn))
            if g.get("reference_unadapted"):
                for lr in g["lr_grid"]:
                    out.append(ft_cmd(t, "standard", lr, g["selection_seed"],
                                      ft_dir("B1", t, "standard", lr, g["selection_seed"], "_raw"), None, "", syn))
    elif a.phase == "B2":
        for t in g["tasks"]:
            for v in g["main_variants"]:
                lr = selected_lr(t, v)
                for s in g["final_seeds"]:
                    out.append(ft_cmd(t, v, lr, s, ft_dir("B2", t, v, lr, s), adapt_dir(v), "", syn))
            if g.get("reference_unadapted"):
                lr = selected_lr(t, "standard", raw=True)
                for s in g["final_seeds"]:
                    out.append(ft_cmd(t, "standard", lr, s, ft_dir("B2", t, "standard", lr, s, "_raw"), None, "", syn))
        ab = P["ablation"]
        lr = selected_lr(ab["task"], ab["lr_from"])
        for v in ab["variants"]:
            for s in g["final_seeds"]:
                out.append(ft_cmd(ab["task"], v, lr, s, ft_dir("B2", ab["task"], v, lr, s), adapt_dir(v), "", syn))
    elif a.phase == "C":
        e = P["esnli"]
        for v in e["variants"]:
            lr = selected_lr(e["lr_from_task"], v)
            for s in e["seeds"]:
                d = ft_dir("C", "esnli", v, lr, s)
                c = ft_cmd("esnli", v, lr, s, d, adapt_dir(v), " --save_best", syn)
                c = c.replace(f"--epochs {g['epochs']}", f"--epochs {e['epochs']}") + " --eval_every 4000"
                out.append(c)
                ig = v in e["ig_variants_all_seeds"] or (v in e["ig_variants_seed1"] and s == 1)
                out.append(f"python scripts/eval_esnli.py --ckpt {d} --out {d}/eval" + (" --ig" if ig else "") + syn)
    elif a.phase == "D":
        ds = P["data_size"]
        for n in ds["sizes"]:
            for v in ds["variants"]:
                lr = selected_lr(ds["task"], v)
                for s in ds["seeds"]:
                    out.append(ft_cmd(ds["task"], v, lr, s, ft_dir("D", ds["task"], v, lr, s, f"_n{n}"),
                                      adapt_dir(v), f" --train_subsample {n} --eval_every 100000", syn))
    elif a.phase == "F":
        out.append(f"python scripts/benchmark.py --out {R}/F/benchmark_fp32 --lengths {P['benchmark']['lengths']}")
        out.append(f"python scripts/benchmark.py --out {R}/F/benchmark_amp --lengths {P['benchmark']['lengths']} --amp")
    print("\n".join(out))


if __name__ == "__main__":
    main()
