import numpy as np
from pathlib import Path
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import torch

class Factor:
    """
    Implementation of a tree.
    """
    def __init__(self, start, end, factor):
        """
        The value of a node is a subset of consecutive indices {low, ..., high - 1} included in
        {0, ..., num_factors - 1}.
        :param low: int
        :param high: int
        :param num_factors: int
        """
        self.start = start
        self.end = end
        self.factor = factor
    
    def param_cal(self, gb_params):
        return partial_prod_deformable_butterfly_params(gb_params, self.start, self.end)

def partial_prod_deformable_butterfly_params(gb_params, low, high):
    """
    Closed form expression of partial matrix_product of butterfly supports. We name S_L, ..., S_1 the butterfly supports of
    size 2^L, represented as binary matrices. Then, the method computes the partial matrix_product S_{high-1} ... S_low.
    :param supscript: list of sizes of factors
    :param subscript: list of sizes of blocks
    :param low: int
    :param high: int, excluded
    :return: numpy array, binary matrix
    """
    params = gb_params[low:high+1]
    result = [1] * 6
    result[0] = params[0][0]
    result[3] = params[-1][3]
    result[4] = params[0][4]
    result[5] = params[-1][5]
    size_one_middle_h = 1
    size_one_middle_w = 1
    for i in range(high-low+1):
        b, c = params[i][1:3]
        size_one_middle_h *= b
        size_one_middle_w *= c
    result[1] = size_one_middle_h
    result[2] = size_one_middle_w
    return result

def compatible_gb_params(param1, param2):
    a1, _, c1, d1, _, q1 = param1
    a2, b2, _, d2, p2, _ = param2
    if q1 != p2:
        return False
    if a1 * c1 != a2:
        return False
    if d1 != b2 * d2:
        return False
    return True

def redundant_pair_gb_params(param1, param2):
    # Redundancy of a chainable pair in the sense of Definition 4.18 of
    # arXiv:2411.04506: the pair (pi_1, pi_2) with pi_l = (a_l, b_l*p_l, c_l*q_l, d_l)
    # and rank r = q_1 = p_2 is redundant iff r >= min(b_1*p_1, c_2*q_2).
    _, b1, _, _, p1, q1 = param1
    _, _, c2, _, p2, q2 = param2
    assert q1 == p2
    return q1 >= min(b1 * p1, c2 * q2)

def compatible_chain_gb_params(gb_params):
    if gb_params[0][0] != 1:
        return False
    if gb_params[0][4] != 1:
        return False
    if gb_params[-1][3] != 1:
        return False
    if gb_params[-1][5] != 1:
        return False
    for i in range(len(gb_params) - 1):
        if not compatible_gb_params(gb_params[i], gb_params[i + 1]):
            return False
    return True

def non_redundant_chain_gb_params(gb_params):
    # Return True iff no adjacent pair of the chain is redundant in the sense
    # of Definition 4.18 of arXiv:2411.04506.
    for i in range(len(gb_params) - 1):
        if redundant_pair_gb_params(gb_params[i], gb_params[i + 1]):
            return False
    return True
    
def param_mul_param(param1, param2):
    assert compatible_gb_params(param1, param2)
    return [param1[0], param1[1] * param2[1], param1[2] * param2[2], param2[3], param1[4], param2[5]]

def remove_redundancy(gb_params):
    """
    Architecture redundancy removal (Algorithm 4.1 of arXiv:2411.04506)
    Input:
    gb_params: a chainable architecture (list of 6-tuples)
    Output:
    reduced_params: a chainable non-redundant architecture with the same
    expressivity (same set of products)
    groups: list mapping each reduced factor to the list of consecutive
    indices of the original factors it merges
    """
    params = [list(param) for param in gb_params]
    groups = [[i] for i in range(len(gb_params))]
    merged = True
    while merged and len(params) > 1:
        merged = False
        for i in range(len(params) - 1):
            if redundant_pair_gb_params(params[i], params[i + 1]):
                params[i:i + 2] = [param_mul_param(params[i], params[i + 1])]
                groups[i:i + 2] = [groups[i] + groups[i + 1]]
                merged = True
                break
    return [tuple(param) for param in params], groups