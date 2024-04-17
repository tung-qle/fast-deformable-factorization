import time
import argparse
import src.GB_gradient as gradient
import src.square_dyadic_gradient as square_dyadic_gradient
import src.GB_factorization as fact
import src.GB_ALS as als
import src.GB_operators as operator
import scipy.linalg
import torch

import pandas as pd

from pathlib import Path


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Experiment to compare the hierarchical factorization methods (normalization or not)"
                    "vs. first-order optimization method. "
                    "vs. alternating least square method"
                    "Loss is the Froebenius norm between the target matrix and the computed factorization."
                    "We use our own implementation of a first-order optimization method. "
                    "The target matrix is the Hadamard matrix.")

    parser.add_argument("--results-dir", type=Path, default="results/fact_vs_iter_vs_als")
    parser.add_argument("--k", type=int, default=9, help="Number of factors.")
    parser.add_argument("--method", type=str, choices=["hierarchical", "gradient", "als", "square-dyadic-gradient-dao"])

    # parameters for hierarchical factorization
    parser.add_argument("--orthonormalize", choices=["True", "False"], default="True")
    parser.add_argument("--hierarchical-order", choices=["left-to-right", "balanced"], default="left-to-right")

    # parameters for iterative method
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate of Adam for iterative method")
    parser.add_argument("--n_adam_epochs", type=int, default=100, help="Number of ADAM iterations")
    parser.add_argument("--n_lbfgs_epochs", type=int, default=20, help="Number of LBFGS iterations")

    # parameters for ALS
    parser.add_argument("--n_als_epochs", type=int, default=5, help="Number of ALS iterations")

    return parser.parse_args()


def balanced_permutation(k):
    if k == 1:
        return [1]
    if k == 2:
        return [1, 2]
    if k % 2 == 0:
        left_perm = balanced_permutation((k // 2) - 1)
        right_perm = [i + (k + 1) // 2 for i in balanced_permutation(k // 2)]
        return [k // 2] + left_perm + right_perm
    if k % 2 == 1:
        left_perm = balanced_permutation(k // 2)
        right_perm = [i + (k + 1) // 2 for i in balanced_permutation(k // 2)]
        return [k // 2 + 1] + left_perm + right_perm


if __name__ == "__main__":
    args = parse_arg()
    size = 2 ** args.k

    # architecture parameters for square dyadic butterfly factorization
    param_format_six = [(2 ** i, 2, 2, 2 ** (args.k - 1 - i), 1, 1) for i in range(args.k)]
    param_format_five = [(size, size, 2, 2, 2 ** (args.k - 1 - i)) for i in range(args.k)]

    # save directory
    save_dir = args.results_dir / f"n-factors={args.k}"
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "hierarchical":
        matrix = torch.from_numpy(scipy.linalg.hadamard(size) * 1.0)
        matrix_twiddle = torch.reshape(matrix, (1, 1, matrix.size()[0], matrix.size()[1]))

        orthonormalize = (args.orthonormalize == "True")
        if args.hierarchical_order == "left-to-right":
            orders = [i for i in range(args.k - 1)]
        elif args.hierarchical_order == "balanced":
            orders = [i - 1 for i in balanced_permutation(args.k - 1)]
        else:
            raise NotImplementedError

        begin = time.time()
        factor_list = fact.GBfactorize(matrix_twiddle, param_format_six, orders, orthonormalize)
        factor_list = [f.factor for f in factor_list]
        end = time.time()

        rel_err = (torch.norm(matrix_twiddle - operator.densification(factor_list, param_format_six)) / torch.norm(
            matrix)).item()
        running_time = end - begin

        metrics = {
            "n-factors": args.k,
            "method": args.method,
            "orthonormalize": orthonormalize,
            "hierarchical-order": args.hierarchical_order,
            "relative-error": rel_err,
            "running-time": running_time
        }
        df = pd.DataFrame([metrics])
        df.to_csv(
            save_dir / f"hierarchical-orthonormalize={orthonormalize}-hierarchical_order={args.hierarchical_order}.csv",
            index=False)

    elif args.method == "gradient":
        matrix = torch.from_numpy(scipy.linalg.hadamard(size) * 1.0)
        factorization = gradient.GBGradient(param_format_six)
        loss_evol, time_evol = factorization.approximate_matrix(matrix.float(), lr=args.lr, epochs=args.n_adam_epochs,
                                                                device="cpu", num_iter_refined=args.n_lbfgs_epochs)
        df = pd.DataFrame()
        df["relative-error"] = loss_evol
        df["running-time"] = time_evol
        df.to_csv(
            save_dir / f"gradient-lr={args.lr}-n-adam-epochs={args.n_adam_epochs}-n-lbfgs-epochs={args.n_lbfgs_epochs}.csv",
            index=False)

    elif args.method == "square-dyadic-gradient-dao":
        matrix = torch.from_numpy(scipy.linalg.hadamard(size) * 1.0)
        factorization = square_dyadic_gradient.SquareDyadicGradient(size)
        loss_evol, time_evol = factorization.approximate_matrix(matrix.float(), lr=args.lr, epochs=args.n_adam_epochs,
                                                                device="cpu", num_iter_refined=args.n_lbfgs_epochs)
        df = pd.DataFrame()
        df["relative-error"] = loss_evol
        df["running-time"] = time_evol
        df.to_csv(
            save_dir / f"square-dyadic-gradient-lr={args.lr}-n-adam-epochs={args.n_adam_epochs}-n-lbfgs-epochs={args.n_lbfgs_epochs}.csv",
            index=False)

    elif args.method == "als":
        matrix = torch.from_numpy(scipy.linalg.hadamard(size) * 1.0)

        loss_evol, time_evol = als.gb_als(matrix.float(), [[t[0], t[1]] for t in param_format_five],
                                          [[t[2], t[3], t[4]] for t in param_format_five], MaxItr=args.n_als_epochs)
        loss_evol = [t.item() for t in loss_evol]
        df = pd.DataFrame()
        df["relative-error"] = loss_evol
        df["running-time"] = time_evol
        df.to_csv(
            save_dir / f"als-n-als-epochs={args.n_als_epochs}.csv", index=False)

    else:
        raise NotImplementedError

    print(df)
