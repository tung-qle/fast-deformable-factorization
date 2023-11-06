import torch
from einops import rearrange
from src.GB_param_generate import *

def param_mul_param(param1, param2):
    return [param1[0], param1[1] * param2[1], param1[2] * param2[2], param2[3], param1[4], param2[5]]

def random_generate(param):
    """
    Random a set of butterfly factors
    Input:
    A list of params
    """
    if len(param) == 6:
        return torch.rand(param[0], param[3], param[1] * param[4], param[2] * param[5])
    return torch.rand(param[0], param[3], param[1], param[2])

def twiddle_mul_twiddle(l_twiddle, r_twiddle, l_param, r_param):
    """
    Compute the product of two compatible twiddles
    Input:
    l_twiddle, r_twiddle: two tensors of order 4
    Output: a tensor of order 4 (a twiddle)
    """
    a1, b1, c1, d1, p1, q1 = l_param
    a2, b2, c2, d2, p2, q2 = r_param
    l_twiddle = rearrange(l_twiddle, 'a1 d1 b1 (c1 q1) -> (a1 c1) d1 b1 q1', c1 = c1)
    r_twiddle = rearrange(r_twiddle, 'a2 d2 (p2 b2) c2 -> a2 (b2 d2) p2 c2', b2 = b2)
    result = torch.matmul(l_twiddle.float(), r_twiddle.float())
    result = rearrange(result, '(a c1) (b2 d) b1 c2 -> a d (b1 b2) (c1 c2)', c1 = c1, b2 = b2)
    return result

def twiddle_to_dense(twiddle):
    """
    Write twiddle to the dense form
    """
    a, d, b, c = twiddle.size()
    n = a * d * c
    output = torch.eye(n)
    t = twiddle.view(a * d, b, c)  # (a * d, b, c)
    output = output.reshape(a, c, d, n).permute(0, 2, 1, 3).reshape(a * d, c, n) # (a * d, c, n)
    output = torch.bmm(t, output) # (a * d, b, n)
    return output.reshape(a, d, b, n).permute(0, 2, 1, 3).reshape(a * d * b, n)

def densification(twiddle_list, param_list):
    """
    Compute product of twiddle
    Input:
    twiddle_list: List of twiddles
    param_list: List of params
    Output: product of twiddles
    """
    a, b, c, d, p, q = param_list[0]
    n = a * b * d * p
    output = torch.ones(1, n, 1, 1)
    current_param = [1, 1, 1, n, 1, 1]
    for twiddle, param in zip(twiddle_list, param_list):
        output = twiddle_mul_twiddle(output, twiddle, current_param, param)
        current_param = param_mul_param(current_param, param)
    return output

if __name__ == "__main__":
    param1 = [1,2,2,2,1,2]
    param2 = [2,2,2,1,2,1]
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
    m, min_param =  test.smallest_monotone_debfly_chain(num_mat, format='abcdpq')
    twiddle_list = [random_generate(param) for param in min_param]
    print(densification(twiddle_list, min_param))
    matrix = [twiddle_to_dense(bf) for bf in twiddle_list]
    print(matrix[0] @ matrix[1] @ matrix[2] @ matrix[3])