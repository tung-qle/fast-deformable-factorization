import torch
from einops import rearrange
from src.GB_param_generate import *
from src.utils import partial_prod_deformable_butterfly_params


def param_mul_param(param1, param2):
    return [param1[0], param1[1] * param2[1], param1[2] * param2[2], param2[3], param1[4], param2[5]]


def random_generate(param, seed=None):
    """
    Random a set of butterfly factors
    Input:
    A list of params
    """
    if seed:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None
    if len(param) == 6:
        return torch.rand(param[0], param[3], param[1] * param[4], param[2] * param[5], generator=gen)
    return torch.rand(param[0], param[3], param[1], param[2], generator=gen)


def twiddle_mul_twiddle(l_twiddle, r_twiddle, l_param, r_param):
    """
    Compute the product of two compatible twiddles
    Input:
    l_twiddle, r_twiddle: two tensors of order 4
    Output: a tensor of order 4 (a twiddle)
    """
    a1, b1, c1, d1, p1, q1 = l_param
    a2, b2, c2, d2, p2, q2 = r_param
    l_twiddle = rearrange(l_twiddle, 'a1 d1 b1 (c1 q1) -> (a1 c1) d1 b1 q1', c1=c1)
    r_twiddle = rearrange(r_twiddle, 'a2 d2 (p2 b2) c2 -> a2 (b2 d2) p2 c2', b2=b2)
    result = torch.matmul(l_twiddle.float(), r_twiddle.float())
    result = rearrange(result, '(a c1) (b2 d) b1 c2 -> a d (b1 b2) (c1 c2)', c1=c1, b2=b2)
    return result


def twiddle_to_dense(twiddle):
    """
    Write twiddle to the dense form
    """
    a, d, b, c = twiddle.size()
    n = a * d * c
    output = torch.eye(n)
    t = twiddle.view(a * d, b, c)  # (a * d, b, c)
    output = output.reshape(a, c, d, n).permute(0, 2, 1, 3).reshape(a * d, c, n)  # (a * d, c, n)
    output = torch.bmm(t, output)  # (a * d, b, n)
    return output.reshape(a, d, b, n).permute(0, 2, 1, 3).reshape(a * d * b, n)


def dense_to_twiddle(matrix, gb_params):
    """
    Write the target of a factorization with gb_params in twiddle format, by
    restricting a dense matrix to the support of the product of gb_params
    (inverse of twiddle_to_dense on this support). When a_1 > 1 or d_L > 1,
    the product of the chain is not a full matrix but has a fixed support S.
    For any factors X_1, ..., X_L,
        ||A - X_1 ... X_L||^2 = ||A_S - X_1 ... X_L||^2 + ||A - A_S||^2,
    so the entries of A outside S play no role in the factorization: the best
    factors for A are those for A_S, and ||A - A_S|| is a lower bound on the
    approximation error of this architecture.
    Input:
    matrix: a dense matrix of size (M, N) or (1, 1, M, N), or a tensor already
    in twiddle format, which is returned unchanged
    gb_params: a chainable architecture (list of 6-tuples)
    Output: a twiddle of size (a_1, d_L, b * p, c * q), where (a_1, b, c, d_L, p, q)
    is the product of gb_params
    """
    a, b, c, d, p, q = partial_prod_deformable_butterfly_params(gb_params, 0, len(gb_params) - 1)
    b, c = b * p, c * q
    twiddle_size = (a, d, b, c)
    dense_size = (a * b * d, a * c * d)
    if matrix.size() == twiddle_size:
        return matrix
    if matrix.size() == (1, 1) + dense_size:
        matrix = matrix[0, 0]
    assert matrix.size() == dense_size, \
        f"Target of size {tuple(matrix.size())} matches neither the dense size {dense_size} " \
        f"nor the twiddle size {twiddle_size} expected from the product of gb_params"

    twiddle = rearrange(matrix, '(a1 b d1) (a2 c d2) -> a1 a2 d1 d2 b c', a1=a, a2=a, d1=d, d2=d, b=b, c=c)
    twiddle = twiddle.diagonal(dim1=0, dim2=1).diagonal(dim1=0, dim2=1)
    return rearrange(twiddle, 'b c a d -> a d b c')


def product_of_twiddles(twiddles, params):
    """
    Product of a (sub-)chain of twiddles, in twiddle format: the output has
    size (a, d, b * p, c * q) where (a, b, c, d, p, q) is the product of params,
    which is (1, 1, M, N) for a chain with a_1 = d_L = 1 and an M x N product.
    Use twiddle_to_dense to write the result in dense form otherwise.
    """
    output = twiddles[0]
    current_param = list(params[0])
    for twiddle, param in zip(twiddles[1:], params[1:]):
        output = twiddle_mul_twiddle(output, twiddle, current_param, param)
        current_param = param_mul_param(current_param, param)
    return output


def densification(twiddle_list, param_list):
    """
    Compute product of twiddle
    Input:
    twiddle_list: List of twiddles
    param_list: List of params
    Output: product of twiddles, in twiddle format (see product_of_twiddles)
    """
    return product_of_twiddles(twiddle_list, param_list)


if __name__ == "__main__":
    param1 = [1, 2, 2, 2, 1, 2]
    param2 = [2, 2, 2, 1, 2, 1]
    twiddle1 = torch.rand(param1[0], param1[3], param1[1] * param1[4], param1[2] * param1[5])
    twiddle2 = torch.rand(param2[0], param2[3], param2[1] * param2[4], param2[2] * param2[5])
    # param = [[8, 4, 4, 2, 1], [4, 8, 2, 4, 2]]
    # R_parameters = torch.cat([twiddle2.reshape(-1), twiddle1.reshape(-1)])
    # print(twiddle_mul_twiddle(twiddle1, twiddle2, param1, param2))
    matrix1 = twiddle_to_dense(twiddle1)
    matrix2 = twiddle_to_dense(twiddle2)
    # print(matrix1)
    # print(matrix2)
    # print(matrix1 @ matrix2)
    # print(twiddle_mul_twiddle(twiddle1, twiddle2, param1, param2))

    rank = 2
    num_mat = 4
    input_size = 2304
    output_size = 512
    test = DebflyGen(512, 2304, rank)
    m, min_param = test.smallest_monotone_debfly_chain(num_mat, format='abcdpq')
    twiddle_list = [random_generate(param) for param in min_param]
    print(densification(twiddle_list, min_param))
    matrix = [twiddle_to_dense(bf) for bf in twiddle_list]
    print(matrix[0] @ matrix[1] @ matrix[2] @ matrix[3])
