# BSD 3-Clause License
#
# Copyright (c) 2023, Le Quoc Tung,  ZHENG Leon
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from GB_factorization import GBfactorize
from GB_ALS import gb_als
from GB_gradient import GBGradient
from generalized_fly_utils import DebflyGen
from GB_operators import twiddle_mul_twiddle, param_mul_param, random_generate, densification

import numpy as np
import scipy
import argparse
import math
import time
import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment to compare the hierarchical factorization methods"
                    "vs. first-order optimization method and alternating least square method "
                    "Loss is the Froebenius norm between the target matrix and the computed factorization."
                    "We use our own implementation of a first-order optimization method. "
                    "The target matrix is a matrix admitting butterfly factorization."
                    "It can be noisy, i.e. we add iid Gaussian white noise on the entries. ")
    parser.add_argument("--k", type=int, default=3, help="Number of factors.")
    parser.add_argument("--input_size", type=int, default=512, help="Number of rows of matrix")
    parser.add_argument("--output_size", type=int, default=512, help="Number of columns of matrix")
    parser.add_argument("--trial", type=int, default=10, help="Number of times we repeat the experiments, by sampling "
                                                              "different noisy version of the target matrix.")
    parser.add_argument("--sigma", type=float, default=0.01, help="Standard deviation on the Gaussian white noise to "
                                                                  "sample the noisy version of the target matrix.")
    parser.add_argument("--results_prefix_path", type=str, default="results")
    parser.add_argument("--optim", type=str, default="Adam", help="First order optimizer.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate of first order optimizer.")
    parser.add_argument("--epochs", type=int, default=100, help = "Number of epochs for first order methods")
    parser.add_argument("--max_iter", type=int, default=20, help = "Number of iterations of ALS")
    return parser.parse_args()

if __name__ == "__main__":
    arg = parse_args()
    input_size = arg.input_size
    output_size = arg.output_size
    num_fts = arg.k 
    results_path = "./" + arg.results_prefix_path + "_" + str(input_size) + "_" + str(output_size)
    optimizer = arg.optim
    lr = arg.lr
    epochs = arg.epochs
    max_iter = arg.max_iter

    time_hfact = []
    error_hfact = []

    time_first_order = []
    error_first_order = []

    time_als = []
    error_als = []

    for trial in range(arg.trial):
        print(f"\nEXP_ID: {trial+1}/{arg.trial}")

        # Generate butterfly matrix and butterfly chains
        _, gb_params_6 = DebflyGen(input_size, output_size, 1).smallest_monotone_debfly_chain(num_fts, format = 'abcdpq')
        _, gb_params_5 = DebflyGen(input_size, output_size, 1).smallest_monotone_debfly_chain(num_fts, format = 'pqrst')
        print(gb_params_5)
        twiddle_list = [random_generate(param) for param in gb_params_6]
        matrix = densification(twiddle_list, gb_params_6)

        # Hierarchical factorization
        begin = time.time()
        orders = [i for i in range(num_fts)]
        factor_list = GBfactorize(matrix, gb_params_6, orders, True)
        factor_list = [f.factor for f in factor_list]
        end = time.time()
        error_hfact.append(torch.norm(matrix - densification(factor_list, gb_params_6)) / torch.norm(matrix))
        time_hfact.append(end - begin)

        # First order optimization
        factorization = GBGradient(gb_params_6)
        error, running_time = factorization.approximate_matrix(matrix, lr = 0.1, epochs=epochs)
        time_first_order.append(np.array(running_time))
        error_first_order.append(np.array(error))

        # Alternating least squares
        error, running_time = gb_als(matrix.squeeze(), [[t[0], t[1]] for t in gb_params_5], [[t[2], t[3], t[4]] for t in gb_params_5], MaxItr=max_iter)
        time_als.append(np.array(running_time))
        error_als.append(np.array(error))

    time_hfact = np.array(time_hfact)
    error_hfact = np.log10(np.array(error_hfact))

    time_first_order = np.array(time_first_order)
    error_first_order = np.log10(np.array(error_first_order))

    time_als = np.array(time_als)
    error_als = np.log10(np.array(error_als))

    np.savez(
        results_path,
        size = (input_size, output_size),
        time_hfact = time_hfact, error_hfact = error_hfact,
        time_first_order=time_first_order, error_first_order=error_first_order,
        time_als=time_als, error_als=error_als
    )

    plt.plot(np.mean(time_first_order, axis = 0), np.mean(error_first_order, axis = 0), color='red', label="Adam")
    plt.plot(np.mean(time_als, axis = 0), np.mean(error_als, axis = 0), color='blue', label="ALS")
    plt.scatter(np.mean(time_hfact), np.mean(error_hfact), color="black", label="Hierarchical")
    plt.legend()
    plt.xlabel("Running time (seconds)")
    plt.ylabel("Logarithm of loss function")
    plt.show()