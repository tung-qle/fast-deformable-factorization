import time
import argparse
import src.generalized_fly_utils as generalized_fly_utils
import src.GB_factorization as fact
import src.GB_operators as operator
import src.utils as utils
import scipy.linalg
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Experiment to compare the hierarchical factorization methods (normalization or not)"
                    "with different permutations and"
                    "with/without normalization"
                    "Loss is the Froebenius norm between the target matrix and the computed factorization."
                    "We use our own implementation of a first-order optimization method. "
                    "The target matrix is the Hadamard matrix.")

    parser.add_argument("--k", type=int, default=4, help="Number of factors.")
    parser.add_argument("--rank", type=list, default=[1, 2, 4], help="Rank of subblocks")
    parser.add_argument("--noise", type=float, default=0.1, help="Std of gaussian noise")
    parser.add_argument("--n_times", type=int, default=10, help="Number of trials")
    parser.add_argument("--results_path", type=str, default="./data/hierarchical_fact_n_factors_")
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
    n_factors = arg.k
    result_path = arg.results_path + str(n_factors) + ".pkl"
    noise = arg.noise
    n_trials = arg.n_times
    matrix_size = [128, 256, 512, 1024, 2048, 4096, 8192]
    matrix_rank = arg.rank
    mean_error_list = []
    mean_running_time_list = []
    std_error_list = []
    std_running_time_list = []
    mean_noise = []

    for rank in matrix_rank:
        mean_error_dict = {'perm1_norm': [], 'perm2_norm': [], "perm1_vanilla": [], "perm2_vanilla": []}
        mean_running_time_dict = {'perm1_norm': [], 'perm2_norm': [], "perm1_vanilla": [], "perm2_vanilla": []}
        std_error_dict = {'perm1_norm': [], 'perm2_norm': [], "perm1_vanilla": [], "perm2_vanilla": []}
        std_running_time_dict = {'perm1_norm': [], 'perm2_norm': [], "perm1_vanilla": [], "perm2_vanilla": []}
        m_noise = []
        for size in matrix_size:
            time_perm1_normalize = []
            time_perm1_vanilla = []
            time_perm2_normalize = []
            time_perm2_vanilla = []
            error_perm1_normalize = []
            error_perm1_vanilla = []
            error_perm2_normalize = []
            error_perm2_vanilla = []

            test = operator.DebflyGen(size, size, rank)
            m, min_param = test.smallest_monotone_debfly_chain(n_factors, format='abcdpq')
            perm1 = [i for i in range(n_factors - 1)]
            perm2 = [i - 1 for i in balanced_permutation(n_factors - 1)]
            for _ in range(n_trials):
                twiddle_list = [operator.random_generate(param) for param in min_param]
                # matrix = operator.densification(twiddle_list, min_param) + noise * torch.randn(size, size)
                matrix = operator.densification(twiddle_list, min_param)
                noise_matrix = torch.randn(size, size)
                noise_matrix = noise_matrix / torch.linalg.norm(noise_matrix) * torch.linalg.norm(matrix) * noise
                matrix = matrix + noise_matrix

                m_noise.append(
                    (torch.norm(matrix - operator.densification(twiddle_list, min_param)) / torch.norm(matrix)).item())
                print(m_noise[-1])
                # left-to-right factorization, no orthonormalization
                begin = time.time()
                factor_list = fact.GBfactorize(matrix, min_param, perm1, False)
                factor_list = [f.factor for f in factor_list]
                end = time.time()
                error_perm1_vanilla.append(
                    (torch.norm(matrix - operator.densification(factor_list, min_param)) / torch.norm(matrix)).item())
                time_perm1_vanilla.append(end - begin)

                # left-to-right factorization, with orthonormalization
                begin = time.time()
                factor_list = fact.GBfactorize(matrix, min_param, perm1, True)
                factor_list = [f.factor for f in factor_list]
                end = time.time()
                error_perm1_normalize.append(
                    (torch.norm(matrix - operator.densification(factor_list, min_param)) / torch.norm(matrix)).item())
                time_perm1_normalize.append(end - begin)

                # balanced factorization, no orthonormalization
                begin = time.time()
                factor_list = fact.GBfactorize(matrix, min_param, perm2, False)
                factor_list = [f.factor for f in factor_list]
                end = time.time()
                error_perm2_vanilla.append(
                    (torch.norm(matrix - operator.densification(factor_list, min_param)) / torch.norm(matrix)).item())
                time_perm2_vanilla.append(end - begin)

                # balanced factorization, with orthonormalization
                begin = time.time()
                factor_list = fact.GBfactorize(matrix, min_param, perm2, True)
                factor_list = [f.factor for f in factor_list]
                end = time.time()
                error_perm2_normalize.append(
                    (torch.norm(matrix - operator.densification(factor_list, min_param)) / torch.norm(matrix)).item())
                time_perm2_normalize.append(end - begin)

            mean_error_dict["perm1_vanilla"].append(np.mean(np.array(error_perm1_vanilla)))
            std_error_dict["perm1_vanilla"].append(np.std(np.array(error_perm1_vanilla)))
            mean_error_dict["perm2_vanilla"].append(np.mean(np.array(error_perm2_vanilla)))
            std_error_dict["perm2_vanilla"].append(np.std(np.array(error_perm2_vanilla)))
            mean_error_dict["perm1_norm"].append(np.mean(np.array(error_perm1_normalize)))
            std_error_dict["perm1_norm"].append(np.std(np.array(error_perm1_normalize)))
            mean_error_dict["perm2_norm"].append(np.mean(np.array(error_perm2_normalize)))
            std_error_dict["perm2_norm"].append(np.std(np.array(error_perm2_normalize)))

            mean_running_time_dict["perm1_vanilla"].append(np.mean(np.array(time_perm1_vanilla)))
            std_running_time_dict["perm1_vanilla"].append(np.std(np.array(time_perm1_vanilla)))
            mean_running_time_dict["perm2_vanilla"].append(np.mean(np.array(time_perm2_vanilla)))
            std_running_time_dict["perm2_vanilla"].append(np.std(np.array(time_perm2_vanilla)))
            mean_running_time_dict["perm1_norm"].append(np.mean(np.array(time_perm1_normalize)))
            std_running_time_dict["perm1_norm"].append(np.std(np.array(time_perm1_normalize)))
            mean_running_time_dict["perm2_norm"].append(np.mean(np.array(time_perm2_normalize)))
            std_running_time_dict["perm2_norm"].append(np.std(np.array(time_perm2_normalize)))

        mean_error_list.append(mean_error_dict)
        mean_running_time_list.append(mean_running_time_dict)
        std_error_list.append(std_error_dict)
        std_running_time_list.append(std_running_time_dict)
        mean_noise.append(np.mean(np.array(m_noise)))

    with open(result_path, "wb") as handle:
        pickle.dump([mean_error_list, mean_running_time_list, std_error_list, std_running_time_list, mean_noise],
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(result_path, "rb") as handle:
        mean_error_list, mean_running_time_list, std_error_list, std_running_time_list, mean_noise = pickle.load(handle)

    fontsize = 15
    fontsize2 = 12

    fig, ax = plt.subplots(len(matrix_rank), 2, figsize=(15, 15))
    for index, (ax1, ax2) in enumerate(ax):
        ax1.errorbar(matrix_size, mean_running_time_list[index]["perm1_vanilla"],
                     std_running_time_list[index]["perm1_vanilla"], label=r"$\sigma_1$, Algorithm 7", color='red',
                     marker='o')
        ax1.errorbar(matrix_size, mean_running_time_list[index]["perm1_norm"],
                     std_running_time_list[index]["perm1_norm"], label=r"$\sigma_1$, Algorithm 8", color='blue',
                     marker='o')
        ax1.errorbar(matrix_size, mean_running_time_list[index]["perm2_vanilla"],
                     std_running_time_list[index]["perm2_vanilla"], label=r"$\sigma_2$, Algorithm 7", color='red',
                     marker='x')
        ax1.errorbar(matrix_size, mean_running_time_list[index]["perm2_norm"],
                     std_running_time_list[index]["perm2_norm"], label=r"$\sigma_2$, Algorithm 8", color='blue',
                     marker='x')

        ax1.tick_params(axis='both', which='major', labelsize=fontsize)
        ax1.set_ylabel("running time (s)", fontsize=fontsize)
        ax1.set_xlabel("size", fontsize=fontsize)
        ax1.set_xticks(matrix_size)
        # ax1.set_ylabel(r'$\log_{10} \|A - XY^\top\|_F$', fontsize = fontsize)
        # ax1.set_yticks([-12, -10, -8, -6, -4, -2, 0, 2])
        # ax1.set_title("a)", fontsize = fontsize)
        ax1.set_xscale("log")
        # ax1.set_yscale("log")
        ax1.set_xticks(matrix_size)
        ax1.set_xticklabels([str(i) for i in matrix_size])
        ax1.legend(fontsize=fontsize2)
        ax1.grid()

        print(mean_noise[index])
        ax2.errorbar(matrix_size, mean_error_list[index]["perm1_vanilla"], std_error_list[index]["perm1_vanilla"],
                     label=r"$\sigma_1$, Algorithm 7", color='red', marker='o')
        ax2.errorbar(matrix_size, mean_error_list[index]["perm1_norm"], std_error_list[index]["perm1_norm"],
                     label=r"$\sigma_1$, Algorithm 8", color='blue', marker='o')
        ax2.errorbar(matrix_size, mean_error_list[index]["perm2_vanilla"], std_error_list[index]["perm2_vanilla"],
                     label=r"$\sigma_2$, Algorithm 7", color='red', marker='x')
        ax2.errorbar(matrix_size, mean_error_list[index]["perm2_norm"], std_error_list[index]["perm2_norm"],
                     label=r"$\sigma_2$, Algorithm 8", color='blue', marker='x')
        ax2.plot(matrix_size, [mean_noise[index]] * len(matrix_size), label="noise level", color="black")
        ax1.tick_params(axis='both', which='major', labelsize=fontsize)
        ax2.set_ylabel("approximation error", fontsize=fontsize)
        ax2.set_xlabel("size", fontsize=fontsize)
        # ax1.set_ylabel(r'$\log_{10} \|A - XY^\top\|_F$', fontsize = fontsize)
        # ax1.set_yticks([-12, -10, -8, -6, -4, -2, 0, 2])
        # ax1.set_title("a)", fontsize = fontsize)
        ax2.set_xscale("log")
        # ax2.set_yscale("log")
        ax2.set_xticks(matrix_size)
        ax2.set_xticklabels([str(i) for i in matrix_size])
        if index == 0:
            ax2.legend(loc="lower right", fontsize=fontsize2)
        else:
            ax2.legend(loc="upper right", fontsize=fontsize2)
        ax2.grid()

    fig.savefig("comparison_hierarchy.png", dpi=200)
