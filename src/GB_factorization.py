import torch
from einops import rearrange
from src.utils import *
from src.GB_operators import *
import time
import numpy as np


def low_rank_project(M, rank=1, normalized_type='L'):
    """
    Return low rank approximation by batch SVD
    Input:
    M: a tensor of order 4, performing svd on the two last axis
    rank: desired rank
    """
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    S_sqrt = S[..., :rank].sqrt()
    U = U[..., :rank] * rearrange(S_sqrt, '... rank -> ... 1 rank')
    Vt = rearrange(S_sqrt, '... rank -> ... rank 1') * Vt[..., :rank, :]

    # print(torch.linalg.norm(torch.matmul(U, Vt) - M))
    return U, Vt


def torch_svd(A, rank):
    """
    Return low rank approximation by finding eigenvalues of a symmetric matrix
    Good when one size of a matrix is small
    Input:
    A: a tensor of order 4, performing svd on the two last axis
    rank: desired rank
    """
    if A.dtype == torch.complex64 or A.dtype == torch.complex128:
        B = torch.matmul(A, A.mH)
    else:
        B = torch.matmul(A, A.transpose(-1, -2))

    sq_S, U = torch.linalg.eigh(B)
    # print(sq_S[..., -(rank+1):])
    U = U[..., -rank:]
    if A.dtype == torch.complex64 or A.dtype == torch.complex128:
        Vh = torch.matmul(U.mH, A)
    else:
        Vh = torch.matmul(U.transpose(-1, -2), A)

    # print(torch.linalg.norm(torch.matmul(U, Vh) - A))
    return U, Vh


def dense_to_pre_low_rank_projection(matrix, b2, c1):
    """
    Reshape a twiddle be ready to factorized
    """
    return rearrange(matrix, 'a d (b1 b2) (c1 c2) -> (a c1) (b2 d) b1 c2', b2=b2, c1=c1)


def left_to_twiddle(left, c1):
    """
    Reshape left twiddle
    """
    return rearrange(left, '(a c1) d b q -> a d b (c1 q)', c1=c1)


def right_to_twiddle(right, b2):
    """
    Reshape right twiddle
    """
    return rearrange(right, 'a (b2 d) b c -> a d (b b2) c', b2=b2)


def gbf_normalization(l_twiddle, r_twiddle, l_param, r_param, type='left'):
    """
    Performing pairwise normalization using QR factorization
    Input:
    l_twiddle: left factor
    r_twiddle: right factor
    l_param: left GB parameter
    r_twiddle: right GB parameter
    type: left -> normalized column left factor, right -> normalized row right factor
    Output: two new factors with one of them being column (row) normalized
    """
    a1, b1, c1, d1, p1, q1 = l_param
    a2, b2, c2, d2, p2, q2 = r_param
    l_twiddle = rearrange(l_twiddle, 'a1 d1 b1 (c1 q1) -> (a1 c1) d1 b1 q1', c1=c1)
    r_twiddle = rearrange(r_twiddle, 'a2 d2 (p2 b2) c2 -> a2 (b2 d2) p2 c2', b2=b2)
    if type == 'left':
        l_twiddle, m_twiddle = torch.linalg.qr(l_twiddle)
        r_twiddle = torch.matmul(m_twiddle, r_twiddle)
        l_twiddle = rearrange(l_twiddle, '(a1 c1) d1 b1 q1 -> a1 d1 b1 (c1 q1)', c1=c1)
        r_twiddle = rearrange(r_twiddle, 'a2 (b2 d2) p2 c2 -> a2 d2 (p2 b2) c2', b2=b2)
    else:
        l_twiddle_tp = r_twiddle.permute(0, 1, 3, 2)
        r_twiddle_tp = l_twiddle.permute(0, 1, 3, 2)
        l_twiddle_tp, m_twiddle_tp = torch.linalg.qr(l_twiddle_tp)
        r_twiddle_tp = torch.matmul(m_twiddle_tp, r_twiddle_tp)
        l_twiddle = rearrange(r_twiddle_tp, '(a1 c1) d1 q1 b1 -> a1 d1 b1 (c1 q1)', c1=c1)
        r_twiddle = rearrange(l_twiddle_tp, 'a2 (b2 d2) c2 p2 -> a2 d2 (p2 b2) c2', b2=b2)
    return l_twiddle, r_twiddle


def intermediate_factorization(start, middle, end, gb_params, target, normalized_type='L', track_epsilon=False):
    """
    Performing one level of hierarchical factorization
    Input: 
    start - end: the initial interval
    middle: the separation of the interval start - end
    gb_params: parameters for butterfly factorization
    target: the target factors
    normalized_type: not important for now
    Output: two factors respecting the supports (start - mid) and (mid + 1 - end) 
    """
    param = partial_prod_deformable_butterfly_params(gb_params, start, end)
    param_left = partial_prod_deformable_butterfly_params(gb_params, start, middle)
    param_right = partial_prod_deformable_butterfly_params(gb_params, middle + 1, end)
    assert target.size() == (param[0], param[3], param[1] * param[4], param[2] * param[5])

    # Reshape the target twiddle 
    target = dense_to_pre_low_rank_projection(target, param_right[1], param_left[2])

    # Compute batch SVD
    l_factor, r_factor = low_rank_project(target, rank=param_left[-1], normalized_type=normalized_type)
    if track_epsilon:
        # ...
        low_rank_errors = torch.linalg.norm(target - torch.matmul(l_factor, r_factor), dim=(-1, -2))
        norms = torch.linalg.norm(target, dim=(-1, -2))
        relative_error = low_rank_errors / norms
        epsilon = torch.max(relative_error)
    else:
        epsilon = None

        # return l_factor, r_factor, low_rank_errors
    # l_factor, r_factor = torch_svd(target, rank = param_left[-1])

    # print("Size l_factor: ", l_factor.size())
    # print("Size r_factor: ", r_factor.size())
    # Reshape the factor twiddle 
    l_factor = left_to_twiddle(l_factor, param_left[2])

    # print(r_factor.size())
    r_factor = right_to_twiddle(r_factor, param_right[1])

    # if not track_epsilon:
    #     return l_factor, r_factor
    return l_factor, r_factor, epsilon


def GBfactorize(matrix, gb_params, orders, normalize=True, normalized_type='L', track_epsilon=False):
    """
    Input: 
    matrix: target matrix that will be factorized
    gb_params: the set of parameters describing the parameterization of GB factors
    orders: a permutation describing the order of factorization
    Output:
    A list of GB factors approximating the target matrix
    """
    result = [Factor(0, len(gb_params) - 1, matrix)]
    max_epsilon = 0
    for i in orders:
        # Search for the corresponding intermediate factors
        if normalize:
            for index in range(len(result)):
                f = result[index]
                if i > f.end:
                    l_factor, r_factor = gbf_normalization(result[index].factor, result[index + 1].factor,
                                                           result[index].param_cal(gb_params),
                                                           result[index + 1].param_cal(gb_params), "left")
                    result[index].factor = l_factor
                    result[index + 1].factor = r_factor
                    continue
                break
            for index in range(len(result))[::-1]:
                f = result[index]
                if i < f.start:
                    l_factor, r_factor = gbf_normalization(result[index - 1].factor, result[index].factor,
                                                           result[index - 1].param_cal(gb_params),
                                                           result[index].param_cal(gb_params), "right")
                    result[index - 1].factor = l_factor
                    result[index].factor = r_factor
                    continue
                break
        for index in range(len(result)):
            f = result[index]
            if f.start <= i and i < f.end:
                l_factor, r_factor, epsilon = intermediate_factorization(f.start, i, f.end, gb_params, f.factor,
                                                                         normalized_type=normalized_type,
                                                                         track_epsilon=track_epsilon)
                if track_epsilon and epsilon.item() > max_epsilon:
                    max_epsilon = epsilon.item()

                l_element = Factor(f.start, i, l_factor)
                r_element = Factor(i + 1, f.end, r_factor)
                del result[index]
                result.insert(index, l_element)
                result.insert(index + 1, r_element)
                break
    if track_epsilon:
        return result, max_epsilon
    return result


if __name__ == "__main__":
    rank = 3
    num_mat = 5
    input_size = 4096
    output_size = 4096
    test = DebflyGen(input_size, output_size, rank)
    m, min_param = test.smallest_monotone_debfly_chain(num_mat, format='abcdpq')

    print(min_param)
    twiddle_list = [random_generate(param) for param in min_param]
    matrix = densification(twiddle_list, min_param)
    orders = [i for i in range(num_mat - 1)]
    # orders = np.random.permutation(num_mat - 1)

    begin = time.time()
    factor_list = GBfactorize(matrix, min_param, orders, False)
    factor_list = [f.factor for f in factor_list]
    end = time.time()
    print("Error: ", torch.norm(matrix - densification(factor_list, min_param)))
    print("Time: ", end - begin)

    begin = time.time()
    factor_list = GBfactorize(matrix, min_param, orders, True)
    factor_list = [f.factor for f in factor_list]
    end = time.time()
    print("Error: ", torch.norm(matrix - densification(factor_list, min_param)))
    print("Time: ", end - begin)
