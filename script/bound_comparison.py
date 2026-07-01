import torch
import argparse
import src.GB_operators as operator
import src.GB_factorization as fact
import numpy as np
import scipy

import matplotlib.pyplot as plt


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_level", type=float, nargs='+', default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--k", type=int)
    return parser.parse_args()


def multiply_pattern(pattern_left, pattern_right):
    a1, b1, c1, d1 = pattern_left
    a2, b2, c2, d2 = pattern_right
    assert a1 * c1 // a2 == b2 * d2 // d1
    assert a2 % a1 == 0 and d1 % d2 == 0
    pattern = (a1, b1 * d1 // d2, a2 * c2 // a1, d2)
    return pattern


def get_monarch_pattern(list_patterns, split):
    assert 0 <= split < len(list_patterns) - 1
    left_pattern = list_patterns[0]
    for i in range(1, split + 1):
        left_pattern = multiply_pattern(left_pattern, list_patterns[i])
    right_pattern = list_patterns[split + 1]
    for i in range(split + 2, len(list_patterns)):
        right_pattern = multiply_pattern(right_pattern, list_patterns[i])
    return [left_pattern, right_pattern]


def convert_to_six_params(list_patterns):
    # assert len(list_patterns) == 6
    return [(pattern[0], pattern[1], pattern[2], pattern[3], 1, 1) for pattern in list_patterns]


def monarch_error(matrix, permutation, list_patterns, split_index):
    monarch_arch = convert_to_six_params(get_monarch_pattern(list_patterns, split_index))
    factor_list = fact.GBfactorize(matrix, monarch_arch, [0], True)
    factor_list = [f.factor for f in factor_list]
    return torch.norm(matrix - operator.densification(factor_list, monarch_arch))


if __name__ == '__main__':
    args = get_arguments()

    k = args.k
    n = 2 ** k
    orders = [i for i in range(k - 1)]

    architecture = [(2 ** i, 2, 2, 2 ** (k - 1 - i)) for i in range(k)]
    print(architecture)

    torch.manual_seed(args.seed)
    random_noise = torch.randn(2, 2 ** (k - 1))
    zero_matrix = torch.zeros_like(random_noise)

    our_bound_list = []
    bound_liu_list = []
    epsilon_list = []
    approximation_error_list = []
    matrix_norm_list = []
    relative_energy_list = []

    for noise_level in args.noise_level:
        # twiddle_list = [operator.random_generate(param) for param in architecture]
        # matrix = args.scaling * operator.densification(twiddle_list, architecture).squeeze().squeeze()
        matrix = torch.from_numpy(scipy.linalg.hadamard(n) * 1.0)
        noise = torch.zeros_like(matrix)
        for i in range(2):
            matrix[2 ** (k - 1) * i, :2 ** (k - 1)] = zero_matrix[i]
            noise[2 ** (k - 1) * i, :2 ** (k - 1)] = random_noise[i]

        matrix_norm = torch.norm(matrix).item()
        noise_norm = torch.norm(noise).item()

        matrix = matrix + noise_level * matrix_norm / noise_norm * noise

        matrix = matrix.unsqueeze(0).unsqueeze(0)
        matrix_norm = torch.norm(matrix).item()
        print(f"Matrix norm: {torch.norm(matrix)}")

        energy = torch.norm(random_noise).item() / torch.norm(matrix).item()

        factor_list, epsilon = fact.GBfactorize(matrix, convert_to_six_params(architecture), orders, True,
                                                track_epsilon=True)
        factor_list = [f.factor for f in factor_list]
        approximation_error = torch.norm(matrix - operator.densification(factor_list, convert_to_six_params(architecture))).item()
        print("Approximation error: ", approximation_error)
        print(f"epsilon = {epsilon}")

        bound_liu = np.sqrt(k) * epsilon * torch.norm(matrix).item()
        print(f"Bound Liu et al.: {bound_liu}")

        our_bound = 0
        for split in orders:
            our_bound += 2 ** (k - split - 2) * monarch_error(matrix, orders, architecture, split)
        # our_bound = 2 * monarch_error(matrix, 1) + monarch_error(matrix, 2)
        print(f"Our bound (Theorem 7.2): {our_bound}")

        our_bound_list.append(our_bound / matrix_norm)
        bound_liu_list.append(bound_liu / matrix_norm)
        epsilon_list.append(epsilon)
        approximation_error_list.append(approximation_error / matrix_norm)
        relative_energy_list.append(energy)

        # approximation (torch.norm(matrix - operator.densification(factor_list, architecture)) / torch.norm(matrix)).item()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(args.noise_level, our_bound_list, marker='x', label="Our bound from Theorem 7.2")
    ax.plot(args.noise_level, bound_liu_list, marker='x', label="Bound (2.1) from (Liu et al., 2021)")
    ax.plot(args.noise_level, approximation_error_list, marker='x', label="Approximation error by Algorithm 6.2")
    ax.set_xlabel("Noise level $\epsilon$")
    ax.set_ylabel("Relative error")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()
    ax.grid()

    plt.tight_layout()
    plt.savefig("bound_comparison_noise_level.pdf")
    plt.show()
