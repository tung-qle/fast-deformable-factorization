import time
import argparse
import src.GB_factorization as fact
import src.GB_operators as operator
from warnings import warn
try:
    import torch
    found_pytorch = True
except ImportError:
    warn("Did not find PyTorch, therefore use NumPy/SciPy")
    found_pytorch = False
import numpy as np
from pathlib import Path
import pandas as pd


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Experiment to compare the hierarchical factorization"
        "methods (normalization or not)"
        "with different permutations and"
        "with/without normalization"
        "Loss is the Froebenius norm between the target matrix and"
        "the computed factorization."
        "We use our own implementation of a first-order optimization method. "
        "The target matrix is the Hadamard matrix."
    )

    parser.add_argument(
        "--n-factors", type=int, default=4, help="Number of factors."
    )
    parser.add_argument(
        "--rank", type=int, default=2, help="Rank of subblocks"
    )
    parser.add_argument(
        "--noise", type=float, default=0.1, help="Std of gaussian noise"
    )
    parser.add_argument("--matrix-size", type=int, default=128)
    parser.add_argument(
        "--seed-target-matrix",
        type=int,
        default=1,
        help="Seed for generating the target matrix",
    )
    parser.add_argument(
        "--orthonormalize", choices=["True", "False"], default="True"
    )
    parser.add_argument(
        "--hierarchical-order",
        choices=["left-to-right", "balanced"],
        default="left-to-right",
    )
    parser.add_argument(
        "--save-dir", type=Path, default="./results/hierarchical_comp_new"
    )
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
    arg = parse_arg()
    n_factors = arg.n_factors
    rank = arg.rank
    noise = arg.noise
    matrix_size = arg.matrix_size
    seed = arg.seed_target_matrix
    orthonormalize = arg.orthonormalize == "True"
    hierarchical_order = arg.hierarchical_order
    save_dir = arg.save_dir

    # set architecture for butterfly factorization
    test = operator.DebflyGen(matrix_size, matrix_size, rank)
    m, min_param = test.smallest_monotone_debfly_chain(
        n_factors, format="abcdpq"
    )

    # generate target matrix
    if found_pytorch:
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)
    twiddle_list = [operator.random_generate(param) for param in min_param]
    matrix = operator.densification(twiddle_list, min_param)
    if found_pytorch:
        noise_matrix = torch.randn(matrix_size, matrix_size)
        noise_matrix = (
            noise_matrix
            / torch.linalg.norm(noise_matrix)
            * torch.linalg.norm(matrix)
            * noise
        )
        matrix = matrix + noise_matrix
        noise_level = (
            torch.linalg.norm(noise_matrix) / torch.linalg.norm(matrix)
        ).item()
    else:
        noise_matrix = np.random.randn(matrix_size, matrix_size)
        noise_matrix = (
            noise_matrix
            / np.linalg.norm(noise_matrix)
            * np.linalg.norm(matrix)
            * noise
        )
        matrix = matrix + noise_matrix
        noise_level = np.linalg.norm(noise_matrix) / np.linalg.norm(matrix)
    print('M', matrix.shape)

    # permutation
    if hierarchical_order == "left-to-right":
        perm = [i for i in range(n_factors - 1)]
    elif hierarchical_order == "balanced":
        perm = [i - 1 for i in balanced_permutation(n_factors - 1)]
    else:
        raise NotImplementedError

    # measure the performance of the hierarchical factorization
    begin = time.time()
    factor_list = fact.GBfactorize(matrix, min_param, perm, orthonormalize)
    factor_list = [f.factor for f in factor_list]
    end = time.time()
    if found_pytorch:
        error = (
            torch.norm(matrix - operator.densification(factor_list, min_param))
            / torch.norm(matrix)
        ).item()
    else:
        error = np.linalg.norm(
            matrix - operator.densification(factor_list, min_param)
        ) / np.linalg.norm(matrix)

    # save results
    results = {
        "n-factors": n_factors,
        "rank": rank,
        "noise": noise,
        "matrix-size": matrix_size,
        "seed-target-matrix": seed,
        "orthonormalize": orthonormalize,
        "hierarchical-order": hierarchical_order,
        "target-matrix-norm": (
            torch.linalg.norm(matrix).item() if found_pytorch
            else np.linalg.norm(matrix)
        ),
        "noise-level-relative": noise_level,
        "time": end - begin,
        "error-relative": error,
    }

    # save in dataframe save_dir / results.csv
    save_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([results])
    df.to_csv(
        save_dir
        / f"n_factors={n_factors}-rank={rank}-noise={noise}-matrix_size={matrix_size}-seed={seed}-orthonormalize={orthonormalize}-hierarchical-order={hierarchical_order}.csv"
    )
    print(df)
