import numpy as np
from sympy import primerange
import torch
import sys

MAX = 1e18

def check_compatibility(b, c, type):
    # Return if the parameters b, c of deformale butterfly ensure the monotonicity of the chain
    # Params:
    # b, c: the parameters b and c
    # type: takes three values {square, expanding, shrinking} corresponding to three type of monotonicity
    if type == "square":
        return b == c
    if type == "expanding":
        return b <= c
    if type == "shrinking":
        return b >= c

def format_conversion(m, n, chainbc, weight, format = 'abcd'):
    # Return a sequence of deformable butterfly factors using the infomation of b and c
    # Params:
    # m, n: size of the matrix
    # chainbc: a sequence of pairs (b,c)
    # format: support 2 formats (a,b,c,d) and (p, q, r, s, t)
    a = 1
    d = m
    result = []
    #print(chainbc)
    for i in range(len(chainbc)):
        (b,c) = chainbc[i]
        d = d // b
        if format == 'abcd':
            result.append((a, b * weight[i], c * weight[i + 1], d))
        elif format == 'pqrst':
            result.append((a * b * d * weight[i], a * c * d * weight[i + 1], b * weight[i], c * weight[i + 1], d))
        elif format == 'abcdpq':
            result.append((a, b, c, d, weight[i], weight[i + 1]))
        a = a * c
    return result

def factorize(n):
    # Return a dictionary storing all prime divisor of n with their corresponding powers
    prime_ints = list(primerange(1, n + 1))
    result = {}
    index = 0
    while n > 1:    
        if n % prime_ints[index] == 0:
            k = 0
            while n % prime_ints[index] == 0:
                n = n // prime_ints[index]
                k = k + 1
            result[prime_ints[index]] = k
        index = index + 1
    return result

def random_Euler_sum(n,k):
    # Return k nonnegative integers whose sum equals to n
    result = [0] * k
    sample = np.random.randint(0, k, n)
    for i in sample:
        result[i] = result[i] + 1
    return result

def enumerate_Euler_sum(n, k):
    if k == 1:
        yield (n,)
        return

    for i in range(n + 1):
        for t in enumerate_Euler_sum(n - i, k - 1):
            yield (i,) + t

class DebflyGen():
    def __init__(self, m, n, r):
        self.m = m
        self.n = n
        self.rank = r
        self.divisor_m = []
        self.divisor_n = []
        # Calculate the set of divisor of m 
        for i in range(1, m + 1):
            if m % i == 0:
                self.divisor_m.append(i)
        
        # Calculate the set of divisor of n
        for i in range(1, n + 1):
            if n % i == 0:
                self.divisor_n.append(i)

        self.dp_table = np.zeros((m + 1, n + 1))
        self.dp_table_temp = np.zeros((m + 1, n + 1))

    def random_debfly_chain(self, n_factors, format = 'abcd'):
        # Return an uniformly random deformable butterfly chain whose product is of size m x n has n_factors factors.
        # TODO: Implement the method
        decomp_m = factorize(self.m)
        decomp_n = factorize(self.n)
        b_chain = [1] * n_factors
        c_chain = [1] * n_factors
        weight = [1] + [self.rank] * (n_factors - 1) + [1]
        for (divisor, powers) in decomp_m.items():
            random_partition = random_Euler_sum(powers, n_factors)
            for i in range(len(b_chain)):
                b_chain[i] = b_chain[i] * (divisor ** random_partition[i])

        for (divisor, powers) in decomp_n.items():
            random_partition = random_Euler_sum(powers, n_factors)
            for i in range(len(c_chain)):
                c_chain[i] = c_chain[i] * (divisor ** random_partition[i])
        chainbc = [(b_chain[i], c_chain[i]) for i in range(n_factors)]
        return format_conversion(self.m, self.n, chainbc, weight, format = format)

    @staticmethod
    def enumeration_inner_chain(m, n_factors):
        if m == 1:
            return [[1] * n_factors]
        f_divisors, f_powers = list(factorize(m).items())[0]
        results = []
        for f1 in enumerate_Euler_sum(f_powers, n_factors):
            for f2 in DebflyGen.enumeration_inner_chain(m // (f_divisors ** f_powers), n_factors):
                results.append([(f_divisors ** a) * b for (a,b) in zip(f1, f2)])
        return results
    
    def enumeration_debfly_chain(self, n_factors, format = 'abcd'):
        results = []
        weight = [1] + [self.rank] * (n_factors-1) + [1]
        chain_b = DebflyGen.enumeration_inner_chain(self.m, n_factors)
        chain_c = DebflyGen.enumeration_inner_chain(self.n, n_factors)
        for f1 in chain_b:
            for f2 in chain_c:
                results.append(format_conversion(self.m, self.n, list(zip(f1, f2)), weight, format = format))
        return results

    def smallest_monotone_debfly_chain(self, n_factors, format = 'abcd'):
        # Return a deformable butterfly chain whose product is of size m x n has n_factors factors.
        # Parameters:
        # n_factors: the number 
        try:
            assert n_factors > 0
        except AssertionError:
            print('Need at least 1 factor in the function')
        memorization = {}

        weight = [self.rank] * (n_factors - 1) + [1]
        # Determine the monotonicity type
        if self.m == self.n:
            type = "square" 
        elif self.m > self.n:
            type = "shrinking"
        else:
            type = "expanding"

        # Initialize the dynamic programming table
        for i in self.divisor_m:
            for j in self.divisor_n:
                if check_compatibility(i,j,type):
                    self.dp_table[i,j] = i * j * self.rank
                else:
                    self.dp_table[i,j] = MAX
        
        # Update the value in the dynamic programming table
        for k in range(n_factors - 1):
            for i in self.divisor_m:
                for j in self.divisor_n:
                    self.dp_table_temp[i,j] = MAX
            
            for i in self.divisor_m:
                for j in self.divisor_n:
                    for ii in self.divisor_m:
                        if i <= ii:
                            break
                        if i % ii != 0:
                            continue
                        for jj in self.divisor_n:
                            if j <= jj:
                                break
                            if j % jj != 0:
                                continue
                            if not check_compatibility(ii,jj,type):
                                continue
                            n_params_factor = i * jj * weight[k] * weight[k + 1]
                            if self.dp_table_temp[i,j] > n_params_factor + jj * self.dp_table[i // ii,j // jj]:
                                self.dp_table_temp[i,j] = n_params_factor + jj * self.dp_table[i // ii,j // jj]
                                memorization[(i,j,k+1)] = (ii, jj)
            self.dp_table = self.dp_table_temp * 1
        
        # Recover the parameterizations (b,c)
        k = n_factors - 1
        current_i = self.m
        current_j = self.n
        chainbc = []
        while k >= 0:
            if k == 0:
                chainbc.append((current_i, current_j))
                break
            i,j = memorization[(current_i, current_j, k)]
            chainbc.append((i,j))
            current_i = current_i // i
            current_j = current_j // j
            k = k - 1
        print(chainbc)
        return self.dp_table[self.m, self.n], format_conversion(self.m, self.n, chainbc, [1] + weight, format = format)

def optimized_deform_butterfly_mult_torch(input, num_mat, R_parameters, R_shapes, return_intermediates=False,
                                          version="bmm"):
    """
    Less reshape than the original version.
    Assume that input is 2D (n, in_size).
    """
    n = input.shape[0]
    output = input.contiguous()
    intermediates = [output]
    temp_p = 0
    for m in range(num_mat):
        R_shape = R_shapes[m]
        output_size, input_size, row, col, diag = R_shape[:]
        num_p = col * output_size
        nb_blocks = input_size // (col * diag)
        if version == "pointwise":
            t = R_parameters[temp_p:temp_p + num_p].view(nb_blocks, diag, row, col).permute(0, 2, 3, 1)
            output = output.view(n, nb_blocks, 1, col, diag)
            output = (t * output).sum(dim=-2)
        elif version == "bmm":
            t = R_parameters[temp_p:temp_p + num_p].view(nb_blocks * diag, row, col)  # (nb_blocks * diag, row, col)
            output = output.reshape(n, nb_blocks, col, diag).transpose(-1, -2).reshape(n, -1, col)
            output = torch.bmm(output.transpose(0, 1), t.transpose(2, 1))
            output = output.reshape(nb_blocks, diag, n, row).permute(2, 0, 3, 1)  # (n, nb_blocks, row, diag)
        elif version == "conv1d":
            t = R_parameters[temp_p:temp_p + num_p].view(-1, col, 1)
            output = output.reshape(n, nb_blocks, col, diag).transpose(-1, -2).reshape(n, -1, 1)
            output = torch.nn.functional.conv1d(output, t, groups=nb_blocks * diag)
            output = output.view(n, nb_blocks, diag, row).transpose(-1, -2)
        else:
            raise NotImplementedError
        temp_p += num_p
        intermediates.append(output)

    return output.reshape(n, output_size) if not return_intermediates else intermediates

#-------------------- Useful function to handle generalized butterfly chains --------------------

def count_gb_parameters(param):
    """
    Input: A generalized butterfly parameter
    Output: Number of parameters
    """
    if len(param) == 4:
        return param[0] * param[1] * param[2] * param[3]
    elif len(param) == 5:
        return param[0] * param[3]
    else:
        return param[0] * param[1] * param[2] * param[3] * param[4] * param[5]
    
def count_parameters(param_chain):
    """
    Input: A generalized butterfly chain
    Output: Number of parameters
    """
    assert len(param_chain) > 0
    count = 0
    for param in param_chain:
        count += count_gb_parameters(param)
    return count

def check_monotone(param_chain, rank):
    """
    Input: A generalized butterfly chain and the intended rank
    Output: Decide if the chain is monotone (defined as in the paper Deformable butterfly)
    """
    assert len(param_chain) > 0
    weight = [1] + [rank] * (len(param_chain) - 1) + [1]
    if len(param_chain[0]) == 4:
        m = param_chain[0][0] * param_chain[0][1] * param_chain[0][3]
        n = param_chain[-1][0] * param_chain[-1][2] * param_chain[-1][3]
    elif len(param_chain[0]) == 5:
        m = param_chain[0][0] 
        n = param_chain[-1][1]
    else:
        m = param_chain[0][0] * param_chain[0][1] * param_chain[0][3] * param_chain[0][4]
        n = param_chain[-1][0] * param_chain[-1][2] * param_chain[-1][3] * param_chain[-1][5]
    
    if m == n:
        type = "square" 
    elif m > n:
        type = "shrinking"
    else:
        type = "expanding"

    for i in range(len(param_chain)):
        if len(param_chain[i]) == 4:
            b = param_chain[i][1] // weight[i]
            c = param_chain[i][2] // weight[i + 1]
            if not check_compatibility(b, c, type):
                return False
        elif len(param_chain[i]) == 5:
            b = param_chain[i][2] // weight[i]
            c = param_chain[i][3] // weight[i + 1]
            if not check_compatibility(b, c, type):
                return False
        else:
            if not check_compatibility(param_chain[i][1], param_chain[i][2], type):
                return False
    return True


if __name__ == '__main__':
    rank = 1
    generator = DebflyGen(m=3072, n=768, r=rank)
    # chain = get_i_th_monotone_chain_min_params(generator, num_mat=2, rank=1, i=0)
    _, chain = generator.smallest_monotone_debfly_chain(n_factors=2, format = 'abcdpq')
    print(chain)
    print(check_monotone(chain, rank=rank))
    """
    rank = 2
    num_mat = 2
    n = 500
    input_size = 2304
    output_size = 512
    test = DebflyGen(512, 2304, rank)
    result = test.enumeration_debfly_chain(num_mat, format = 'pqrst')
    m, min_param =  test.smallest_monotone_debfly_chain(num_mat, format='abcd')

    print(len(result))
    result = list(filter(lambda param_chain: check_monotone(param_chain, rank = rank), result))
    print(len(result))
    print(result)

    result = list(filter(lambda param_chain: count_parameters(param_chain) < 2 * m, result))
    print(len(result))

    n_params, R_shapes = test.smallest_monotone_debfly_chain(num_mat, format='pqrst')
    print(R_shapes)
    n_params = int(n_params)
    R_parameters = torch.randn(n_params)
    input = torch.randn(n, input_size)
    result = optimized_deform_butterfly_mult_torch(input, num_mat, R_parameters, R_shapes[::-1])
    print(result.size())
    #print(test.random_debfly_chain(10))
    """