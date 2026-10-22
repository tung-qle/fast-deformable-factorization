"""
Testing the butterfly factorization of the Hadamard transform, with a square
dyadic architecture of arbitrary inner rank (redundant when rank >= 2).
"""

import argparse
import time

import numpy as np
import torch
from scipy.linalg import hadamard

from src.GB_factorization import *
from src.GB_operators import *


def hadamard_chain(n, rank=1):
    # Square dyadic architecture of depth n for the Hadamard matrix of size
    # 2 ** n, with all inner ranks equal to rank
    weight = [1] + [rank] * (n - 1) + [1]
    return [(2 ** i, 2, 2, 2 ** (n - i - 1), weight[i], weight[i + 1]) for i in range(n)]


def run(matrix, gb_params, orders, normalize):
    begin = time.time()
    factor_list = GBfactorize_auto(matrix, gb_params, orders, normalize)
    running_time = time.time() - begin
    factor_list = [f.factor for f in factor_list]
    error = torch.norm(matrix - densification(factor_list, gb_params)) / torch.norm(matrix)
    print(f"normalize={normalize}: relative error {error:.2e}, running time {running_time:.2f}s")
    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=13, help="factorize the Hadamard matrix of size 2**n")
    parser.add_argument("--rank", type=int, default=1, help="inner rank of the architecture (redundant when >= 2)")
    parser.add_argument("--random-order", action="store_true", help="factorize in a random order instead of left-to-right")
    args = parser.parse_args()

    N = 2 ** args.n
    matrix = torch.from_numpy(hadamard(N)).reshape(1, 1, N, N).float()
    gb_params = hadamard_chain(args.n, args.rank)
    orders = [int(i) for i in np.random.permutation(args.n - 1)] if args.random_order else list(range(args.n - 1))
    print("gb_params:", gb_params)
    print("orders:", orders)

    for normalize in [False, True]:
        run(matrix, gb_params, orders, normalize)
