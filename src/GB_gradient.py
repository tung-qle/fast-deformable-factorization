import torch
from torch import nn, optim
from torch import functional as F
from src.generalized_fly_utils import count_parameters, count_gb_parameters, DebflyGen
from src.GB_operators import twiddle_mul_twiddle, param_mul_param, random_generate, densification
from src.utils import *
import math
import time


class GBGradient():
    def __init__(self, gb_params):
        self.gb_params = gb_params
        self.total_params = count_parameters(gb_params)
        parameters = []

        for param in self.gb_params:
            count_param = count_gb_parameters(param)
            scaling = 1.0 / math.sqrt(param[1] * param[4])
            parameters.append(torch.randn(count_param) * scaling)

        self.twiddle = nn.Parameter(torch.cat(parameters, dim=0), requires_grad=True)
        self.start = None
        self.loss = nn.MSELoss(reduction="sum")

    def twiddle_to_matrix(self, device):
        a, b, _, d, p, _ = self.gb_params[0]
        n = a * b * d * p
        output = torch.ones(1, n, 1, 1)
        output = output.to(device)
        current_param = [1, 1, 1, n, 1, 1]
        count = 0
        for param in self.gb_params:
            a, b, c, d, p, q = param
            parameters = self.twiddle[count:count + count_gb_parameters(param)]
            parameters = parameters.view(a, d, b * p, c * q).to(device)
            output = twiddle_mul_twiddle(output, parameters, current_param, param)
            current_param = param_mul_param(current_param, param)
            count += count_gb_parameters(param)
        return output

    def approximate_matrix(self, matrix, lr=0.05, optimizer="Adam", momentum=0.9, epochs=100, epsilon=1e-10,
                           device='cpu', num_iter_refined=20):
        """
        Return the relative error || A - \hat{A} ||_F / || A ||_F at each iteration and the running time. In numpy array.
        """
        # Reshape the target matrix
        if len(matrix.size()) == 2:
            matrix = matrix.view(1, 1, matrix.size()[0], matrix.size()[1])

        matrix = matrix.to(device)
        if optimizer == 'SGD':
            optalg = optim.SGD(params=[self.twiddle], lr=lr)
        elif optimizer == 'Momentum':
            optalg = optim.SGD(params=[self.twiddle], lr=lr, momentum=momentum)
        elif optimizer == 'Adam':
            optalg = optim.Adam(params=[self.twiddle], lr=lr)
        else:
            raise NotImplementedError

        error = []  # saving || A - \hat{A} ||_F^2 at each iteration
        running_time = []
        self.start = time.time()
        for index in range(epochs):
            optalg.zero_grad()
            output = self.loss(self.twiddle_to_matrix(device=device),
                               matrix)  # squared Frobenius norm of the difference
            output.backward()
            optalg.step()

            with torch.no_grad():
                error.append(output.item())
            end = time.time()
            running_time.append(end - self.start)
            # if index % 100 == 0:
                # print(index, ": ", error[-1])

        _, refined_loss, refined_time = self.refining(matrix, num_iter=num_iter_refined, device=device)

        total_squared_error = np.array(error + refined_loss)
        total_rel_error = np.sqrt(total_squared_error) / torch.linalg.norm(matrix)
        total_time = np.array(running_time + refined_time)

        return total_rel_error, total_time

    def refining(self, matrix, num_iter=20, device='cpu'):
        loss_array = []
        running_time = []

        def closure():
            optimizer.zero_grad()
            output = self.loss(self.twiddle_to_matrix(device=device),
                               matrix)  # squared Frobenius norm of the difference
            output.backward()
            return output

        # best = np.inf
        optimizer = optim.LBFGS([self.twiddle], line_search_fn='strong_wolfe')
        for index in range(num_iter):
            optimizer.step(closure)
            end = time.time()
            running_time.append(end - self.start)
            loss_array.append(closure().data.numpy())
        return closure().data.numpy(), loss_array, running_time


if __name__ == "__main__":
    rank = 3
    num_mat = 5
    input_size = 512
    output_size = 4096
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    test = DebflyGen(input_size, output_size, rank)
    m, min_param = test.smallest_monotone_debfly_chain(num_mat, format='abcdpq')
    twiddle_list = [random_generate(param) for param in min_param]
    matrix = densification(twiddle_list, min_param)

    epoch = 200
    factorization = GBGradient(min_param)

    factorization.approximate_matrix(matrix, lr=0.1, epochs=epoch, device=device, num_iter_refined=50)
    print(torch.linalg.norm(factorization.twiddle_to_matrix(device) - matrix) / torch.linalg.norm(matrix))
