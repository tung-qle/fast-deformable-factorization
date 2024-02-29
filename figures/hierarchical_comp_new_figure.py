import pandas as pd
import argparse
from pathlib import Path
import itertools
import matplotlib.pyplot as plt


def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument("--results-dir", type=Path)
    args.add_argument("--n-factors", type=int, default=4)
    args.add_argument("--noise", type=float, default=0.1)
    args.add_argument("--rank", type=int, default=4)
    args.add_argument("--hierarchical-order", type=str, default="balanced", choices=["left-to-right", "balanced"])
    args.add_argument("--matrix-size", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192])
    args.add_argument("--seed-target-matrix", type=int, nargs="+", default=list(range(10)))
    args.add_argument("--graph", choices=["time", "error"])
    return args.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    results_dir = args.results_dir

    # extract df
    df = pd.DataFrame()
    for matrix_size, seed, orthonormalize in itertools.product(args.matrix_size, args.seed_target_matrix,
                                                               [True, False]):
        result_file = results_dir / f"n_factors={args.n_factors}-rank={args.rank}-noise={args.noise}-matrix_size={matrix_size}-seed={seed}-orthonormalize={orthonormalize}-hierarchical-order={args.hierarchical_order}.csv"
        csv = pd.read_csv(result_file)
        df = pd.concat([df, csv])

    print(df)

    if args.graph == "time":
        # Time comparison
        time_mean = df[["orthonormalize", "matrix-size", "time"]].groupby(["matrix-size", "orthonormalize"]).mean()
        time_mean = time_mean.rename(columns={"time": "time_mean"})
        time_std = df[["orthonormalize", "matrix-size", "time"]].groupby(["matrix-size", "orthonormalize"]).std()
        time_std = time_std.rename(columns={"time": "time_std"})
        time_df = time_mean.join(time_std)
        print(time_df)

        fig, ax = plt.subplots(figsize=(5, 4))
        # plot time vs matrix size with error bars following std
        for orthonormalize in [True, False]:
            time_df_local = time_df.loc[(slice(None), orthonormalize), :]
            matrix_sizes = time_df_local.index.levels[0]
            print(time_df_local)
            print(matrix_sizes)
            ax.errorbar(matrix_sizes, time_df_local["time_mean"], yerr=time_df_local["time_std"], marker="x", label="With orthonormalization" if orthonormalize else "Without orthonormalization")

        ax.set_yscale("log")
        # set log scale in base 2
        ax.set_xscale("log", base=2)
        # show the ticks of xscale in integers instead of scientific notation
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.set_xticks(matrix_sizes)
        ax.set_xlabel("Matrix size $n$")
        ax.set_ylabel("Running time (s)")
        ax.grid()
        ax.legend()

        params = f"n-factors={args.n_factors}-rank={args.rank}-noise={args.noise}-hierarchical-order={args.hierarchical_order}"
        # ax.set_title(params)

        plt.tight_layout()
        plt.savefig(args.results_dir / f"time_vs_matrix_size_{params}.pdf")
        plt.show()

    elif args.graph == "error":
        # Error comparison
        error_mean = df[["orthonormalize", "matrix-size", "error-relative"]].groupby(["matrix-size", "orthonormalize"]).mean()
        error_mean = error_mean.rename(columns={"error-relative": "error_mean"})
        error_std = df[["orthonormalize", "matrix-size", "error-relative"]].groupby(["matrix-size", "orthonormalize"]).std()
        error_std = error_std.rename(columns={"error-relative": "error_std"})
        error_df = error_mean.join(error_std)

        fig, ax = plt.subplots(figsize=(5, 4))
        # plot time vs matrix size with error bars following std
        for orthonormalize in [True, False]:
            error_df_local = error_df.loc[(slice(None), orthonormalize), :]
            matrix_sizes = error_df_local.index.levels[0]
            ax.errorbar(matrix_sizes, error_df_local["error_mean"], yerr=error_df_local["error_std"], marker="x", label="With orthonormalization" if orthonormalize else "Without orthonormalization")

        # show as horizontal bar the noise level
        noise_level = args.noise
        ax.axhline(y=noise_level, color='r', linestyle='--', label="Noise level")

        # ax.set_yscale("log")
        # set log scale in base 2
        ax.set_xscale("log", base=2)
        # show the ticks of xscale in integers instead of scientific notation
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.set_xticks(matrix_sizes)

        ax.set_xlabel("Matrix size $n$")
        ax.set_ylabel("Relative error")
        ax.grid()
        ax.legend()

        params = f"n-factors={args.n_factors}-rank={args.rank}-noise={args.noise}-hierarchical-order={args.hierarchical_order}"
        # ax.set_title(params)

        plt.tight_layout()
        plt.savefig(args.results_dir / f"error_vs_matrix_size_{params}.pdf")
        plt.show()

    else:
        raise NotImplementedError