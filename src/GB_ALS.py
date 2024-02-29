
import torch
import time
from tqdm import tqdm
import shutil
import os
from src.generalized_fly_utils import DebflyGen
import numpy as np
from src.GB_operators import twiddle_mul_twiddle, param_mul_param, random_generate, densification


def Generate_Block(num_block, subscript):
    '''
    Input:
    - num_block: an interger, which describe the number of
    blocks that needed to build the butterfly matrix.
    - subscript: a vector, which contains 3 elements (r, s, t),
    r and c means the number of unit matrix the row and column contain,
    respectively. The third element b is the size of the unit diagonal
    matrix. Therefore, the size of each block is [r*b, c*b].

    return:
    - blocks: a list, each element is a block for the butterfly matrix.

    30/03/2021
    '''
    blocks = []
    for i in range(0, int(num_block)):
        block = torch.zeros([subscript[0] * subscript[2], subscript[1] * subscript[2]])
        block_unit = []
        for j in range(0, subscript[0] * subscript[1]):
            diag_element = torch.rand([1, subscript[2]])
            block_unit.append(torch.diag_embed(diag_element))
        for m in range(0, subscript[0]):
            for n in range(0, subscript[1]):
                block[m * subscript[2]:(m + 1) * subscript[2], n * subscript[2]:(n + 1) * subscript[2]] \
                    = block_unit[m * subscript[1] + n]
        blocks.append(block)
    return blocks


def Generate_Chain(superscript, subscript):
    '''
    Input:
    - superscript: a list, each of its element is a 2D vector, which describes the size of the
    Butterfly matrix.
    - subscript: a list, each of its element is a 3D vector, which describes the block in the
    Butterfly matrix.

    return:
    - Buttefly: a list, each of its element is a butterfly matrix.

    30/03/2021
    '''
    num_mat = len(superscript)
    Butterfly = []
    for i in range(0, int(num_mat)):
        super = superscript[i]
        sub = subscript[i]

        mat = torch.zeros(super)
        num_block = super[0] / (sub[0] * sub[2])
        blocks = Generate_Block(num_block, sub)

        for j in range(0, int(num_block)):
            mat[j * sub[0] * sub[2]:(j + 1) * sub[0] * sub[2], j * sub[1] * sub[2]:(j + 1) * sub[1] * sub[2]] = blocks[
                j]

        Butterfly.append(mat)
    return Butterfly


def Find_Blocks(Butterfly, superscript, subscript, k, rsvd=0, T1=None, T2=None):
    '''
    Input:
    - Butterfly: a list, each of its element is a buttefly matrix.
    - superscript: a list, each of its element is a 2D vector, which
    describes the size of the Butterfly matrix.
    - subscript: a list, each of its element is a 3D vector, which
    describes the block in the Butterfly matrix.
    - k: an interger, which describes the position of the matrix in the
    Butterfly chain.

    Output:
    - A_cell: a list, each of its elements is a block matrix.
    - C_cell: a list, each of its elements is a block matrix.

    30/03/2021
    '''
    num_mat = len(Butterfly)
    A_cell = []
    C_cell = []

    if k == 0:
        super = superscript[k]
        A = torch.eye(super[0])
        C = Butterfly[k + 1]
        for i in range(k + 2, num_mat):
            C = torch.mm(C, Butterfly[i])
    elif k == (num_mat - 1):
        super = superscript[k]
        C = torch.eye(super[1])
        A = Butterfly[0]
        for i in range(1, num_mat - 1):
            A = torch.mm(A, Butterfly[i])
    else:
        A = Butterfly[0]
        for i in range(1, k):
            A = torch.mm(A, Butterfly[i])
        C = Butterfly[k + 1]
        if (k + 1 + 1) <= num_mat:
            for j in range(k + 1 + 1, num_mat):
                C = torch.mm(C, Butterfly[j])

    if rsvd == 1:
        A = torch.mm(T1, A)
        C = torch.mm(C, T2)

    sub = subscript[k]
    sup = superscript[k]
    num_block = sup[0] / (sub[0] * sub[2])

    for m in range(0, int(num_block)):
        A_cell.append(A[:, m * sub[0] * sub[2]:(m + 1) * sub[0] * sub[2]])
        C_cell.append(C[m * sub[1] * sub[2]:(m + 1) * sub[1] * sub[2], :])
    return A_cell, C_cell


def Get_Vector_Form(A_cell, C_cell, sup, sub):
    '''
    Input:
    - A_cell: a list, each of its elements is a block matrix.
    - C_cell: a list, each of its elements is a block matrix.
    - sup: a vector, which contains 2 lements and describes the size of the Butterfly matrix.
    - sub: a vector, which contains 3 lements (r, s, t), r and s means the number of unit
    matrix the row and column contain, respectively. The third element t is the size of the
    unit diagonal matrix. Therefore, the size of each block is [r*b, s*b].

    Output:
    - AC_mat: a matrix, which satisfies inv(AC_mat) * vec(D) = vec(B)

    30/03/2021
    '''
    num_block = sup[0] / (sub[0] * sub[2])
    w, _ = A_cell[0].shape
    _, h = C_cell[0].shape
    AC_mat = torch.tensor([])
    for i in range(0, int(num_block)):
        # print('i={}'.format(i))
        A_unit = A_cell[i]
        C_unit = C_cell[i]
        for r in range(0, sub[0]):
            # print('r={}'.format(r))
            for s in range(0, sub[1]):
                # print('s={}'.format(s))
                A_unit2 = A_unit[:, r * sub[2]:(r + 1) * sub[2]]
                C_unit2 = C_unit[s * sub[2]:(s + 1) * sub[2], :]
                for t in range(0, sub[2]):
                    # print('t={}'.format(t))
                    AC_mat = torch.cat((AC_mat,
                                        torch.reshape(torch.mm(A_unit2[:, t].reshape([-1, 1]),
                                                               C_unit2[t, :].reshape([1, -1])),
                                                      [w * h, 1])), 1)
                    # print(AC_mat.shape)
    return AC_mat

def cal_A_C(Butterfly, superscript, k):
    num_mat = len(Butterfly)

    if k == 0:
        super = superscript[k]
        A = torch.eye(super[0])
        C = Butterfly[k + 1]
        for i in range(k + 2, num_mat):
            C = torch.mm(C, Butterfly[i])
    elif k == (num_mat - 1):
        super = superscript[k]
        C = torch.eye(super[1])
        A = Butterfly[0]
        for i in range(1, num_mat - 1):
            A = torch.mm(A, Butterfly[i])
    else:
        A = Butterfly[0]
        for i in range(1, k):
            A = torch.mm(A, Butterfly[i])
        C = Butterfly[k + 1]
        if (k + 1 + 1) <= num_mat:
            for j in range(k + 1 + 1, num_mat):
                C = torch.mm(C, Butterfly[j])
    return A, C

def Vec_to_Mat(B_vec, sup, sub):
    '''
    Input:
    - B_vec: a vector, which store the elements in matrix B.
    - sup: a vector, which contains 2 elements and describes the size of the butterfly matrix.
    - sub: a vector, which contains 3 elements (r, s, t), r and s means the number of unit
    matrix the row and column contain, respectively. The third element t is the size of the
    unit diagonal matrix. Therefore, the size of each block is [r*t, c*t].

    Output:
    - B: a matrix, which is in the butterflyl form.

    30/03/2021
    '''
    blocks = []
    num_block = sup[0] / (sub[0] * sub[2])
    for i in range(0, int(num_block)):
        block = torch.zeros([sub[0] * sub[2], sub[1] * sub[2]])
        block_unit = []
        element_block = B_vec[i * sub[0] * sub[1] * sub[2]:(i + 1) * sub[0] * sub[1] * sub[2]].t()
        for j in range(0, sub[0] * sub[1]):
            block_unit.append(torch.diag_embed(element_block[:, j * sub[2]:(j + 1) * sub[2]]))
        for m in range(0, sub[0]):
            for n in range(0, sub[1]):
                block[m * sub[2]:(m + 1) * sub[2], n * sub[2]:(n + 1) * sub[2]] = block_unit[m * sub[1] + n]
        blocks.append(block)

    B = torch.zeros(sup)
    for m in range(0, int(num_block)):
        B[m * sub[0] * sub[2]:(m + 1) * sub[0] * sub[2], m * sub[1] * sub[2]:(m + 1) * sub[1] * sub[2]] = blocks[m]

    return B

def segment(device, D, Butterfly, k, sup, sub):
    '''
    Input:
    - device: CPU or GPU
    - D: a matrix, which needs to be approximated
    - Butterfly: a list, each of its elements is a DeBut factor
    - k: int, default = 0, which means the first DeBut factor
    - sup: a list, each of its element is a superscript of a DeBut factor
    - sub: a list, each of its elements is a subscript of a DeBut factor

    Return:
    - B_vec: a vector, which contains all the nonzero elements in the first DeBut factors.

    Other functions called:
    - cal_A_C: calculate the multiplication results of DeBut factors on the left and right
    sides of the selected DeBut factor.
    - subblocks_LS: solve the LS problem of the subblock matrices.

    08/04/2021
    '''
    # to store the blocks
    D_blocks = []
    C_blocks = []
    B_vec = torch.tensor([])

    # calculate A and C
    A, C = cal_A_C(Butterfly, sup, k)

    # the pattern info in B
    sub_cur = sub[k]
    w_k = sub_cur[0] * sub_cur[2]

    # new sub
    sub_cur_new = sub_cur.copy()
    sub_cur_new[1] = 1

    # calculate the size of blocks in C
    num_factors = len(Butterfly)
    w = sup[num_factors - 1][1]
    h = sup[k + 1][0]
    num_patterns = int(h / (sub[k + 1][0] * sub[k + 1][2]))
    # print('num_patterns: {}'.format(num_patterns))
    w_block = int(w / num_patterns)

    # to store the blocks for A,B,C
    for i in range(0, int(num_patterns)):
        D_blocks.append(D[:, i * w_block:(i + 1) * w_block])
        C_blocks.append(C[i * sub_cur[2]:(i + 1) * sub_cur[2], i * w_block:(i + 1) * w_block])

    # solve subblock LS and contact the long B_vec
    for i in range(0, num_patterns):
        # print('no. patters: {}'.format(i))
        b_vec = subblocks_LS(device, D_blocks[i].to(device), A.to(device), C_blocks[i].to(device), sub_cur_new)
        B_vec = torch.cat((B_vec.to(device), b_vec.to(device)), 0)

    return B_vec

def subblocks_LS(device, D_block, A_block, C_block, sub_cur):
    '''
    Input:
    - device: CPU or GPU.
    - D_block: a matrix, a subblock of D.
    - A_block: a matrix, a subblock of A, which has the same #rows as D_block
    - C_block: a matrix, a subblock of C, which has the same #columns as D_block
    - sub_cur: a vector, which describe the format of the pattern

    Return:
    - B_vec: a vector, which is the nonzero elements in one patter

    08/04/2021
    '''
    # load block matrices
    A_unit = A_block
    C_unit = C_block
    AC_mat = torch.tensor([])

    # size
    w, _ = A_unit.shape
    _, h = C_unit.shape

    # print('shape of A_unit, C_unit: {}, {}.'.format(A_unit.shape, C_unit.shape))

    # print('r, s, t: {}.'.format(sub_cur))
    # small AC_mat
    for r in range(0, sub_cur[0]):  # row in the pattern
        for s in range(0, sub_cur[1]):  # column in the pattern
            A_unit2 = A_unit[:, r * sub_cur[2]:(r + 1) * sub_cur[2]].to(device)
            C_unit2 = C_unit[s * sub_cur[2]:(s + 1) * sub_cur[2], :].to(device)
            # print('shape of A_unit2, and C_unit2: {}, {}.'.format(A_unit2.shape, C_unit2.shape))
            for t in range(0, sub_cur[2]):
                AC_mat = AC_mat.to(device)
                AC_mat = torch.cat((AC_mat,
                                    torch.reshape(torch.mm(A_unit2[:, t].reshape([-1, 1]),
                                                           C_unit2[t, :].reshape([1, -1])),
                                                  [w * h, 1])), 1)
                # print('shape of subblock AC_mat: {}.'.format(AC_mat.shape))

    # solve the block LS
    B_vec = torch.mm(torch.pinverse(AC_mat),
                     torch.reshape(D_block.to(device), [-1, 1]))

    return B_vec

def seg_Vec_to_Mat(B_vec, sup, sub):
    '''
    sup: vector
    sub: vector
    '''
    blocks = []
    num_block = sup[0] / (sub[0] * sub[2])
    for i in range(0, int(num_block)):
        block = torch.zeros([sub[0] * sub[2], sub[1] * sub[2]])
        block_unit = []
        element_block = B_vec[i * sub[0] * sub[1] * sub[2]:(i + 1) * sub[0] * sub[1] * sub[2]].t()
        for j in range(0, sub[0] * sub[1]):
            block_unit.append(torch.diag_embed(element_block[:, j * sub[2]:(j + 1) * sub[2]]))
        for n in range(0, sub[1]):
            for m in range(0, sub[0]):
                block[m * sub[2]:(m + 1) * sub[2], n * sub[2]:(n + 1) * sub[2]] = block_unit[m * sub[1] + n]
        blocks.append(block)

    B = torch.tensor([])
    for m in range(0, int(num_block)):
        B = torch.cat((B, blocks[m]), 1)

    return B

def subblocks_2toend(device, D, Butterfly, k, sup, sub):
    '''
    Input:
    - device: CPU or GPU
    - D: a matrix, which needs to be approximated
    - Butterfly: a list, each of its element is a DeBut factor
    - k: int, which describes the order of the DeBut factors
    - sup: a list, each of its elements describes the size of the DeBut factors
    - sub: a list, each of its elements describes the format of the patterns

    Return:
    - B_vec: a vector, formed by all the nonzero elements of the DeBut factor needs to be updated

    Other functions called:
    - cal_A_C: calculate the multiplication results of DeBut factors on the left and right
    sides of the selected DeBut factor.
    - subblocks_LS: solve the LS problem of the subblock matrices.

    08/04/2021
    '''
    # to store the blocks
    D_blocks = []
    A_blocks = []
    C_blocks = []
    B_vec = torch.tensor([])

    # calculate A and C
    A, C = cal_A_C(Butterfly, sup, k)

    # the info of patterns in B (DeBut factor which needs update)
    sub_cur = sub[k]
    sup_cur = sup[k]
    w_k = sub_cur[1] * sub_cur[2]  # width of an patter in B
    h_k = sub_cur[0] * sub_cur[2]  # height of an patter in B

    # calculate the size of blocks in C
    num_factors = len(Butterfly)
    w = C.shape[1] # the width of the final output
    h = C.shape[0]  # the height of C
    num_patterns = int(sup_cur[0] / (sub[k][0] * sub[k][2]))  # number of patterns in C
    w_block = int(w / num_patterns)  # width of patterns in C
    # print('number of patterns: {}'.format(num_patterns))

    # to store the blocks for A,B,C
    for i in range(0, int(num_patterns)):
        D_blocks.append(D[:, i * w_block:(i + 1) * w_block])
        A_blocks.append(A[:, i * h_k:(i + 1) * h_k])
        # print(A.shape)
        # print(A_blocks[i].shape)
        C_blocks.append(C[i * w_k:(i + 1) * w_k, i * w_block:(i + 1) * w_block])

    # print('length of D/A/C_blocks: {}, {}, {}.'.format(len(D_blocks), len(A_blocks), len(C_blocks)))

    # solve subblock LS and contact the long B_vec
    for i in range(0, num_patterns):
        # print('{}-th subblocks.'.format(i + 1))
        b_vec = subblocks_LS(device, D_blocks[i].to(device), A_blocks[i].to(device), C_blocks[i].to(device), sub_cur)
        B_vec = torch.cat((B_vec.to(device), b_vec.to(device)), 0)

    return B_vec

def gb_als(D, sup, sub, MaxItr, rsvd = 1, device = 'cpu'):
    Butterfly = Generate_Chain(sup, sub)
    num_mat = len(Butterfly)
    D_approx = torch.eye(sup[0][0]).to(device)
    error = []

    for i in range(0, num_mat):
        Butterfly[i].to(device)

    for k in range(0, num_mat):
        D_approx = torch.mm(D_approx, Butterfly[k].to(device))

    error_init = torch.norm(D_approx - D)

    running_time = []

    flag_dirc = 1

    start = time.time()
    for i in range(0, MaxItr):
        # flag_dirc = 1: from left to right
        if flag_dirc == 1:
            for k in range(0, num_mat):
                # print('{}-th DeBut factor.'.format(k+1))
                if k == 0:
                    sup_cur = sup[k]
                    sub_cur = sub[k]
                    B_vec = segment(device, D, Butterfly, k, sup, sub)
                    Butterfly[k] = seg_Vec_to_Mat(B_vec, sup_cur, sub_cur)
                else:
                    sup_cur = sup[k]
                    sub_cur = sub[k]
                    B_vec = subblocks_2toend(device, D, Butterfly, k, sup, sub)
                    Butterfly[k] = Vec_to_Mat(B_vec, sup_cur, sub_cur)

            # flag_dirc = 0: from right to left
        if flag_dirc == 0:
            for k in range(num_mat - 1, -1, -1):
                # print('{}-th DeBut factor.'.format(k+1))
                if k == 0:
                    sup_cur = sup[k]
                    sub_cur = sub[k]
                    B_vec = segment(device, D, Butterfly, k, sup, sub)
                    Butterfly[k] = seg_Vec_to_Mat(B_vec, sup_cur, sub_cur)
                else:
                    sup_cur = sup[k]
                    sub_cur = sub[k]
                    B_vec = subblocks_2toend(device, D, Butterfly, k, sup, sub)
                    Butterfly[k] = Vec_to_Mat(B_vec, sup_cur, sub_cur)

        # print(Butterfly)
        # approximate D
        D_approx = torch.eye(sup[0][0])
        for k in range(0, num_mat):
            D_approx = torch.mm(D_approx.to(device), Butterfly[k].to(device))

        # check error when know the real D
        error.append(torch.norm(D_approx - D) / torch.norm(D))

        # change the flag
        if flag_dirc == 0:
            flag_dirc = 1
        else:
            flag_dirc = 0

        end = time.time()
        running_time.append(end - start)
        print('Minimum relative error (L2 norm) of approximated D: {}.'.format(min(error)))
        print('Time used: ' + str(end - start) + 's.')
    
    return error, running_time

if __name__ == '__main__':
    rank = 1
    num_mat = 2
    input_size = 4
    output_size = 4
    test = DebflyGen(input_size, output_size, rank)

    m, min_param =  test.smallest_monotone_debfly_chain(num_mat, format='abcdpq')
    twiddle_list = [random_generate(param) for param in min_param]
    matrix = densification(twiddle_list, min_param)
    matrix = matrix.squeeze()

    m, min_param =  test.smallest_monotone_debfly_chain(num_mat, format='pqrst')
    print(min_param)
    factors = gb_als(matrix, [[t[0], t[1]] for t in min_param], [[t[2], t[3], t[4]] for t in min_param], MaxItr=100)

    factors = gb_als(matrix, [[t[0], t[1]] for t in min_param], [[t[2], t[3], t[4]] for t in min_param], MaxItr=100)
    print(matrix)
    print(factors[0] @ factors[1])
