import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# results_dir="results/fact_vs_iter_vs_als/anchiale1"
# lr=0.1
# n_adam_epochs=100
# n_lbfgs_epochs=20
# n_als_epochs=5

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default="results/fact_vs_iter_vs_als/anchiale6/2024-04-17")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hierachical-order", type=str, default="balanced", choices=["left-to-right", "balanced"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n_adam_epochs", type=int, default=100)
    parser.add_argument("--n_lbfgs_epochs", type=int, default=20)
    parser.add_argument("--n_als_epochs", type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    hierarchical_with_ortho = pd.read_csv(
        args.results_dir / f"n-factors={args.k}/hierarchical-orthonormalize=True-hierarchical_order={args.hierachical_order}.csv")
    hierarchical_without_ortho = pd.read_csv(
        args.results_dir / f"n-factors={args.k}/hierarchical-orthonormalize=False-hierarchical_order={args.hierachical_order}.csv")
    gradient = pd.read_csv(
        args.results_dir / f"n-factors={args.k}/gradient-lr={args.lr}-n-adam-epochs={args.n_adam_epochs}-n-lbfgs-epochs={args.n_lbfgs_epochs}.csv")
    als = pd.read_csv(args.results_dir / f"n-factors={args.k}/als-n-als-epochs={args.n_als_epochs}.csv")
    square_dyadic_gradient = pd.read_csv(
        args.results_dir / f"n-factors={args.k}/square-dyadic-gradient-lr={args.lr}-n-adam-epochs={args.n_adam_epochs}-n-lbfgs-epochs={args.n_lbfgs_epochs}.csv")

    # for each of the dataframes, plot relative-error vs running-time
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(hierarchical_with_ortho["running-time"], hierarchical_with_ortho["relative-error"],
               label="Hierarchical algorithm, with orthonormalization", color="tab:blue", marker="x", s=100)
    ax.scatter(hierarchical_without_ortho["running-time"], hierarchical_without_ortho["relative-error"],
               label="Hierarchical algorithm, without orthonormalization", color="tab:orange", marker="x", s=100)
    ax.plot(gradient["running-time"], gradient["relative-error"], label="Gradient-based [3], our implementation",
            color="tab:green")
    ax.plot(square_dyadic_gradient["running-time"], square_dyadic_gradient["relative-error"],
            label="Gradient-based [3], original implementation", color="tab:purple")

    ax.plot(als["running-time"], als["relative-error"], label="ALS [21]", color="tab:red", marker=".")

    ax.set_xlabel("Running time (s)")
    ax.set_ylabel("Relative error")

    # log scale
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid()

    # move a bit the legend on the right
    ax.legend(fontsize="small", loc="best")
    plt.tight_layout()

    plt.savefig(args.results_dir / f"n-factors={args.k}" / "comparison.pdf")
    plt.show()
