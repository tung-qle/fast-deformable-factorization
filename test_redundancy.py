"""
Testing the redundancy removal algorithm and the factorization of redundant
butterfly architectures (Algorithm 4.1, Lemma 4.23 and Remark 6.4 of
arXiv:2411.04506)
"""

import torch
from scipy.linalg import hadamard
from src.GB_factorization import *
from src.GB_operators import *
from test_fac import hadamard_chain


def test_remove_redundancy():
    # Rank 3 with 2x2 blocks: first and last pairs are redundant
    gb_params = hadamard_chain(5, 3)
    reduced, groups = remove_redundancy(gb_params)
    assert compatible_chain_gb_params(reduced)
    assert non_redundant_chain_gb_params(reduced)
    assert [i for g in groups for i in g] == list(range(len(gb_params)))
    for pm, group in zip(reduced, groups):
        expected = partial_prod_deformable_butterfly_params(gb_params, group[0], group[-1])
        assert list(pm) == list(expected)
    # The equality case q = b * p is redundant (Definition 4.18)
    gb_params = hadamard_chain(5, 2)
    reduced, _ = remove_redundancy(gb_params)
    assert non_redundant_chain_gb_params(reduced)
    assert len(reduced) < len(gb_params)
    # A strictly non-redundant chain is left untouched
    gb_params = hadamard_chain(5, 1)
    reduced, groups = remove_redundancy(gb_params)
    assert reduced == gb_params
    assert all(len(g) == 1 for g in groups)
    print("test_remove_redundancy: OK")


def product_of_twiddles(twiddles, params):
    # Product of a (sub-)chain of twiddles, kept in twiddle format. Unlike
    # densification, this does not require the chain to start with a = p = 1.
    output = twiddles[0]
    current_param = list(params[0])
    for twiddle, param in zip(twiddles[1:], params[1:]):
        output = twiddle_mul_twiddle(output, twiddle, current_param, param)
        current_param = param_mul_param(current_param, param)
    return output


def test_expand_factor():
    torch.manual_seed(0)
    gb_params = hadamard_chain(4, 3)
    _, groups = remove_redundancy(gb_params)
    for group in groups:
        sub_params = [gb_params[i] for i in group]
        merged = partial_prod_deformable_butterfly_params(sub_params, 0, len(sub_params) - 1)
        twiddle = random_generate(merged)
        expanded = expand_factor(twiddle, sub_params)
        for t, pm in zip(expanded, sub_params):
            assert t.size() == (pm[0], pm[3], pm[1] * pm[4], pm[2] * pm[5])
        product = product_of_twiddles(expanded, sub_params)
        error = torch.norm(product - twiddle) / torch.norm(twiddle)
        assert error < 1e-5, f"expansion of group {group} is not exact: {error}"
    print("test_expand_factor: OK")


def test_factorization_redundant_chain():
    torch.manual_seed(0)
    n = 5
    matrix = torch.from_numpy(hadamard(2 ** n)).reshape(1, 1, 2 ** n, 2 ** n).float()
    for r in [1, 2, 3]:
        gb_params = hadamard_chain(n, r)
        for normalize in [False, True]:
            factor_list = GBfactorize_auto(matrix, gb_params, normalize=normalize)
            factor_list = [f.factor for f in factor_list]
            error = torch.norm(matrix - densification(factor_list, gb_params)) / torch.norm(matrix)
            assert error < 1e-5, f"r={r}, normalize={normalize}: error {error}"
    print("test_factorization_redundant_chain: OK")


def test_factorization_noisy_redundant_chain():
    # A redundant architecture must approximate at least as well as its
    # reduced non-redundant counterpart (they are expressively equivalent)
    torch.manual_seed(0)
    gb_params = hadamard_chain(5, 3)
    reduced, _ = remove_redundancy(gb_params)
    target = densification([random_generate(pm, seed=i) for i, pm in enumerate(gb_params)], gb_params)
    target = target + 0.01 * torch.randn(target.size())
    errors = {}
    for name, params in [("redundant", gb_params), ("reduced", reduced)]:
        if name == "redundant":
            factor_list = GBfactorize_auto(target, params)
        else:
            factor_list = GBfactorize(target, params, list(range(len(params) - 1)), True)
        factor_list = [f.factor for f in factor_list]
        errors[name] = torch.norm(target - densification(factor_list, params)).item()
    assert abs(errors["redundant"] - errors["reduced"]) < 1e-4 * (1 + errors["reduced"]), errors
    print(f"test_factorization_noisy_redundant_chain: OK (errors: {errors})")


if __name__ == "__main__":
    test_remove_redundancy()
    test_expand_factor()
    test_factorization_redundant_chain()
    test_factorization_noisy_redundant_chain()
