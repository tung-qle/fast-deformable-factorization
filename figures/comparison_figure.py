import argparse
import pandas as pd
from pathlib import Path

# results_dir="results/fact_vs_iter_vs_als/anchiale1"
# lr=0.1
# n_adam_epochs=100
# n_lbfgs_epochs=20
# n_als_epochs=5

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default="results/fact_vs_iter_vs_als")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n_adam_epochs", type=int, default=100)
    parser.add_argument("--n_lbfgs_epochs", type=int, default=20)
    parser.add_argument("--n_als_epochs", type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    hierarchical_with_ortho = pd.read_csv(args.results_dir / f"n-factors={args.k}/hierarchical-orthonormalize=True.csv")
    hierarchical_without_ortho = pd.read_csv(args.results_dir / f"n-factors={args.k}/hierarchical-orthonormalize=False.csv")
    gradient = pd.read_csv(args.results_dir / f"n-factors={args.k}/gradient-lr={args.lr}-n_adam_epochs={args.n_adam_epochs}-n_lbfgs_epochs={args.n_lbfgs_epochs}.csv")
    als = pd.read_csv(args.results_dir / f"n-factors={args.k}/als-n_als_epochs={args.n_als_epochs}.csv")

