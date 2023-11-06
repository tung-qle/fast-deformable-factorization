import time
import argparse
import src.generalized_fly_utils as generalized_fly_utils
import src.GB_gradient as gradient
import src.GB_factorization as fact
import src.GB_ALS as als
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
                    "vs. first-order optimization method. "
                    "vs. alternating least square method"
                    "Loss is the Froebenius norm between the target matrix and the computed factorization."
                    "We use our own implementation of a first-order optimization method. "
                    "The target matrix is the Hadamard matrix.")

    parser.add_argument("--k", type=int, default=10, help="Number of factors.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate of Adam for iterative method")
    parser.add_argument("--n_adam_epochs", default=100, help="Number of ADAM iterations")
    parser.add_argument("--n_lbfgs_epochs", default=20, help="Number of LBFGS iterations")
    parser.add_argument("--n_als_epochs", default=5, help="Number of ALS iterations")
    parser.add_argument("--results_path", type=str, default="./data/fact_vs_iter_vs_als_")
    return parser.parse_args()

if __name__ == "__main__":
    arg = parse_arg()
    size = 2 ** arg.k 
    lr = arg.lr
    n_adam_epochs = arg.n_adam_epochs
    n_lbfgs_epochs = arg.n_lbfgs_epochs
    n_als_epochs = arg.n_als_epochs
    result_path = arg.results_path + str(arg.k) + ".pkl"
    param_format_six = [(2 ** i, 2, 2, 2 ** (arg.k - 1 - i), 1, 1) for i in range(arg.k)]
    param_format_five = [(size, size, 2, 2, 2 ** (arg.k - 1 - i)) for i in range(arg.k)]
    matrix = torch.from_numpy(scipy.linalg.hadamard(size) * 1.0)
    matrix_twiddle = torch.reshape(matrix, (1,1,matrix.size()[0], matrix.size()[1]))
    error = dict()
    running_time = dict()
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    # # Running the hierarchical factorization
    # # Orthonormalized butterfly factorization
    # orders = [i for i in range(arg.k - 1)]
    # begin = time.time()
    # print(param_format_six)
    # factor_list = fact.GBfactorize(matrix_twiddle, param_format_six, orders, True)
    # factor_list = [f.factor for f in factor_list]
    # end = time.time()
    # error["ortho_butterfly"] = (torch.norm(matrix_twiddle - operator.densification(factor_list, param_format_six)) / torch.norm(matrix)).item()
    # running_time["ortho_butterfly"] = end - begin
    # print("Error: ", error["ortho_butterfly"])
    # print("Time: ", running_time["ortho_butterfly"])

    # # Normal butterfly factorization
    # orders = [i for i in range(arg.k - 1)]
    # begin = time.time()
    # print(param_format_six)
    # factor_list = fact.GBfactorize(matrix_twiddle, param_format_six, orders, False)
    # factor_list = [f.factor for f in factor_list]
    # end = time.time()
    # error["butterfly"] = (torch.norm(matrix_twiddle - operator.densification(factor_list, param_format_six)) / torch.norm(matrix)).item()
    # running_time["butterfly"] = end - begin
    # print("Error: ", error["butterfly"])
    # print("Time: ", running_time["butterfly"])

    # # Iterative methods
    # factorization = gradient.GBGradient(param_format_six)
    # loss_evol, time_evol = factorization.approximate_matrix(matrix.float(), lr = lr, epochs = n_adam_epochs, device = device, num_iter_refined = n_lbfgs_epochs)
    # error["iterative"] = np.sqrt(np.array(loss_evol) * size) / size
    # running_time["iterative"] = time_evol
    # print(error["iterative"][-1])
    # print(running_time["iterative"][-1])

    # # Alternating least square
    # loss_evol, time_evol = als.gb_als(matrix.float(), [[t[0], t[1]] for t in param_format_five], [[t[2], t[3], t[4]] for t in param_format_five], MaxItr=n_als_epochs)
    # error["als"] = [t.item() for t in loss_evol]
    # running_time["als"] = time_evol

    # with open(result_path, "wb") as handle:
    #     pickle.dump([error, running_time], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(result_path,"rb") as handle:
        error, running_time = pickle.load(handle)
    
    print(error, running_time)

    fontsize = 15
    
    fig, ax1 = plt.subplots(1,1, figsize = (15, 5))
    ax1.plot(running_time['als'], np.array(error['als']), label = "ALS", color = 'green')
    ax1.plot(running_time['iterative'], np.array(error['iterative']), label = "gradient", color = 'gray')
    ax1.scatter(running_time['butterfly'], error['butterfly'], label='butterfly', color = 'red', marker = '*', s = 80)
    ax1.scatter(running_time['ortho_butterfly'], error['ortho_butterfly'], label='ortho_butterfly', color = 'blue', marker = 's', s = 80)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.set_xlabel("running time (s)", fontsize = fontsize)
    ax1.set_ylabel(r'$\log_{10} \|A - XY^\top\|_F / \|A\|_F$', fontsize = fontsize)
    ax1.set_yticks([-12, -10, -8, -6, -4, -2, 0, 2])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(fontsize=fontsize, loc="upper right")
    ax1.grid()
    
    fig.savefig("comparison_methods.png", dpi = 200)

    


     