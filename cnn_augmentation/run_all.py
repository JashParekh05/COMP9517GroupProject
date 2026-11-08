"""Run the whole augmentation study: train every config x seed, evaluate, summarise.

    python run_all.py --data <ROOT>                                # all configs, seed 0, val
    python run_all.py --data <ROOT> --configs baseline geo_photo full --seeds 0 1 2
    python run_all.py --data <ROOT> --skip-train --split test      # final test numbers only

Already-finished runs (runs/<name>/best.pt exists) are skipped, so the script can be
re-started after an interruption (e.g. a Colab disconnect).
"""
import argparse
import os
import subprocess
import sys

from experiments import CONFIGS

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(args):
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("BIRD_DATA"))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--extra-ckpt", nargs="*", default=[],
                    help="additional checkpoints to evaluate, e.g. the teammate's tune-512-clean.pt")
    args, train_extra = ap.parse_known_args()   # unknown args are forwarded to train.py
    if not args.data:
        ap.error("--data is required (or set BIRD_DATA)")
    data = os.path.abspath(args.data)

    names = [f"{c}_s{s}" for c in args.configs for s in args.seeds]
    if not args.skip_train:
        for c in args.configs:
            for s in args.seeds:
                if os.path.exists(os.path.join(HERE, "runs", f"{c}_s{s}", "last.pt")):
                    print(f"skip {c}_s{s} (already trained)")
                    continue
                sh(["train.py", "--data", data, "--config", c, "--seed", str(s),
                    "--epochs", str(args.epochs), *train_extra])

    ckpts = [os.path.join("runs", n, "best.pt") for n in names
             if os.path.exists(os.path.join(HERE, "runs", n, "best.pt"))]
    ckpts += [os.path.abspath(p) for p in args.extra_ckpt]
    sh(["evaluate.py", "--data", data, "--split", args.split, "--ckpt", *ckpts])
    sh(["summarize.py", "--split", args.split] + (["--group-seeds"] if len(args.seeds) > 1 else []))


if __name__ == "__main__":
    main()
