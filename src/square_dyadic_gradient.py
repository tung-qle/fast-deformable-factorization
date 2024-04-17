import torch
from torch import nn, optim
from torch import functional as F
from src.utils import *
import math
import time

from src.butterfly.butterfly import Butterfly


class SquareDyadicGradient():
    def __init__(self, size):
        self.butterfly = Butterfly(size, size, bias=False)
        self.start = None
        self.loss = nn.MSELoss(reduction="sum")

    def to_dense_matrix(self):
        x = torch.eye(matrix.size()[1], matrix.size()[1], device=device, requires_grad=False)
        return self.butterfly(x).t()

    def approximate_matrix(self, matrix, lr=0.05, optimizer="Adam", momentum=0.9, epochs=100, epsilon=1e-10,
                           device='cpu', num_iter_refined=20):
        """
        Return the relative error || A - \hat{A} ||_F / || A ||_F at each iteration and the running time. In numpy array.
        """
        matrix = matrix.to(device)
        if optimizer == 'SGD':
            optalg = optim.SGD(params=[self.butterfly.twiddle], lr=lr)
        elif optimizer == 'Momentum':
            optalg = optim.SGD(params=[self.butterfly.twiddle], lr=lr, momentum=momentum)
        elif optimizer == 'Adam':
            optalg = optim.Adam(params=[self.butterfly.twiddle], lr=lr)
        else:
            raise NotImplementedError

        error = []  # saving || A - \hat{A} ||_F^2 at each iteration
        running_time = []
        self.start = time.time()
        for index in range(epochs):
            optalg.zero_grad()
            output = self.loss(self.to_dense_matrix(), matrix)  # squared Frobenius norm of the difference
            output.backward()
            optalg.step()

            with torch.no_grad():
                error.append(output.item())
            end = time.time()
            running_time.append(end - self.start)
            if index % 10 == 0:
                print(index, ": ", error[-1])

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
            output = self.loss(self.to_dense_matrix(),
                               matrix)  # squared Frobenius norm of the difference
            output.backward()
            return output

        # best = np.inf
        optimizer = optim.LBFGS([self.butterfly.twiddle], line_search_fn='strong_wolfe')
        for index in range(num_iter):
            optimizer.step(closure)
            end = time.time()
            running_time.append(end - self.start)
            loss_array.append(closure().data.numpy())
        return closure().data.numpy(), loss_array, running_time


if __name__ == "__main__":
    import scipy
    import src.GB_operators as operator
    import src.GB_factorization as fact

    L = 7
    size = 2 ** L
    device = "cpu"

    epoch = 200
    factorization = SquareDyadicGradient(size)

    matrix = torch.from_numpy(scipy.linalg.hadamard(size) * 1.0).float()
    matrix_twiddle = matrix.reshape(1, 1, matrix.size()[0], matrix.size()[1])

    architecture = [(2 ** (i - 1), 2, 2, 2 ** (L - i), 1, 1) for i in range(1, L + 1)]
    # twiddle_list = [operator.random_generate(param) for param in architecture]
    # matrix_twiddle = operator.densification(twiddle_list, architecture)

    factor_list = fact.GBfactorize(matrix_twiddle, architecture, list(range(L-1)), True)
    factor_list = [f.factor for f in factor_list]

    rel_err = (torch.norm(matrix_twiddle - operator.densification(factor_list, architecture)) / torch.norm(
        matrix_twiddle)).item()
    print(rel_err)

    matrix = matrix_twiddle.squeeze().squeeze()
    factorization.approximate_matrix(matrix, lr=0.1, epochs=epoch, device=device, num_iter_refined=50)
    print(torch.linalg.norm(factorization.to_dense_matrix() - matrix) / torch.linalg.norm(matrix))
