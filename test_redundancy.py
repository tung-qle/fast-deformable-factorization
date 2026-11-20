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


def outer_chain(gb_params, a, d):
    # Chain whose product is a block diagonal of a blocks, each a direct sum over
    # d (up to a permutation) of products of gb_params (a_1 = a, d_L = d)
    return [(pm[0] * a, pm[1], pm[2], pm[3] * d, pm[4], pm[5]) for pm in gb_params]


def test_factorization_boundary_chain():
    # Chainable architectures with a_1 > 1 or d_L > 1, redundant or not
    torch.manual_seed(0)
    gb_params = hadamard_chain(4, 2)
    twiddles = [random_generate(pm) for pm in gb_params]
    assert torch.equal(densification(twiddles, gb_params), product_of_twiddles(twiddles, gb_params))
    chains = [
        [(2, 2, 2, 4, 1, 2), (4, 2, 2, 2, 2, 1)],  # Chain([(2, 2, 4, 4), (4, 4, 2, 2)]) of lazylinop
        [(2, 2, 2, 4, 1, 1), (4, 2, 2, 2, 1, 1)],
        outer_chain(hadamard_chain(3, 1), 2, 1),
        outer_chain(hadamard_chain(3, 3), 1, 2),
        outer_chain(hadamard_chain(4, 2), 2, 4),
    ]
    for gb_params in chains:
        assert compatible_chain_gb_params(gb_params)
        target = product_of_twiddles([random_generate(pm) for pm in gb_params], gb_params)
        for normalize in [False, True]:
            factor_list = GBfactorize_auto(target, gb_params, normalize=normalize)
            factor_list = [f.factor for f in factor_list]
            error = torch.norm(target - densification(factor_list, gb_params)) / torch.norm(target)
            assert error < 1e-5, f"{gb_params}, normalize={normalize}: error {error}"
    # A target matching neither the dense size nor the twiddle size is rejected
    try:
        GBfactorize_auto(torch.randn(1, 1, 8, 8), chains[0])
    except AssertionError as e:
        assert "dense size (16, 16)" in str(e) and "twiddle size (2, 2, 4, 4)" in str(e)
    else:
        raise AssertionError("a target of the wrong size should be rejected")
    print("test_factorization_boundary_chain: OK")


def test_dense_to_twiddle():
    torch.manual_seed(0)
    # Inverse of twiddle_to_dense, including a, d > 1 and rectangular twiddles
    for a, d, b, c in [(2, 1, 8, 8), (3, 2, 4, 5), (1, 4, 2, 2), (2, 3, 6, 2), (1, 1, 16, 16)]:
        gb_params = [(a, b, c, d, 1, 1)]
        twiddle = torch.randn(a, d, b, c)
        dense = twiddle_to_dense(twiddle)
        assert torch.equal(dense_to_twiddle(dense, gb_params), twiddle)
        assert torch.equal(dense_to_twiddle(dense.reshape(1, 1, *dense.size()), gb_params), twiddle)
    # Inner rank q_1 = p_2 = 2
    gb_params = [(2, 2, 2, 4, 1, 2), (4, 2, 2, 2, 2, 1)]
    twiddle = product_of_twiddles([random_generate(pm) for pm in gb_params], gb_params)
    assert torch.equal(dense_to_twiddle(twiddle_to_dense(twiddle), gb_params), twiddle)
    # A target already in twiddle format is returned unchanged
    assert dense_to_twiddle(twiddle, gb_params) is twiddle

    # The size must match the architecture
    try:
        dense_to_twiddle(torch.randn(8, 8), [(2, 2, 2, 4, 1, 1), (4, 2, 2, 2, 1, 1), (8, 2, 2, 1, 1, 1)])
    except AssertionError as e:
        assert "(16, 16)" in str(e)
    else:
        raise AssertionError("a dense matrix of the wrong size should be rejected")
    print("test_dense_to_twiddle: OK")


def test_factorization_dense_target():
    # Any dense matrix can be factorized with any chainable architecture, even
    # when the product of the architecture cannot be a full matrix (a_1 > 1 or
    # d_L > 1): the entries outside the support of the product are ignored
    torch.manual_seed(0)

    def factorize(target, gb_params, normalize=True):
        factor_list = [f.factor for f in GBfactorize_auto(target, gb_params, normalize=normalize)]
        return twiddle_to_dense(product_of_twiddles(factor_list, gb_params))

    def relative_error(dense, gb_params, normalize=True):
        return (torch.norm(dense - factorize(dense, gb_params, normalize)) / torch.norm(dense)).item()

    # Square-dyadic chain for N = 16 without its left factor: the product is diag(B1, B2)
    gb_params = [(2, 2, 2, 4, 1, 1), (4, 2, 2, 2, 1, 1), (8, 2, 2, 1, 1, 1)]
    support = twiddle_to_dense(torch.ones(2, 1, 8, 8)) != 0

    # A dense target in the range of the chain is recovered
    twiddles = [random_generate(pm) for pm in gb_params]
    dense = twiddle_to_dense(product_of_twiddles(twiddles, gb_params))
    for normalize in [False, True]:
        error = relative_error(dense, gb_params, normalize)
        assert error < 1e-5, f"block diagonal target, normalize={normalize}: error {error}"

    # The Hadamard matrix restricted to the support is in the range of the
    # chain: the error is exactly the norm of the entries outside the support
    dense = torch.from_numpy(hadamard(16)).float()
    error = relative_error(dense, gb_params)
    assert abs(error - 0.5 ** 0.5) < 1e-5, f"Hadamard target: error {error}"

    # The formats (M, N), (1, 1, M, N) and twiddle of the same target are equivalent
    dense = torch.randn(16, 16)
    approximation = factorize(dense, gb_params)
    assert torch.equal(approximation, factorize(dense.reshape(1, 1, 16, 16), gb_params))
    assert torch.equal(approximation, factorize(dense_to_twiddle(dense, gb_params), gb_params))
    # ... and the entries outside the support have no influence on the factors
    assert torch.equal(approximation, factorize(dense * support, gb_params))
    assert torch.equal(approximation[~support], torch.zeros(128))

    # Other architectures with a_1 > 1 and/or d_L > 1, redundant or not: a dense
    # target equal to a product of the chain plus entries outside the support
    # is approximated up to these entries
    chains = [
        [(2, 2, 2, 4, 1, 2), (4, 2, 2, 2, 2, 1)],
        outer_chain(hadamard_chain(3, 1), 1, 2),
        outer_chain(hadamard_chain(3, 3), 2, 1),
        outer_chain(hadamard_chain(4, 2), 2, 4),
    ]
    for gb_params in chains:
        product = twiddle_to_dense(product_of_twiddles([random_generate(pm) for pm in gb_params], gb_params))
        outside = torch.randn(product.size()) * (product == 0)
        for normalize in [False, True]:
            approximation = factorize(product + outside, gb_params, normalize)
            error = torch.norm(product - approximation) / torch.norm(product)
            assert error < 1e-5, f"{gb_params}, normalize={normalize}: error {error}"

    # GBfactorize accepts the same formats; a 2D target also works when a_1 = d_L = 1
    gb_params = hadamard_chain(4, 1)
    dense = torch.from_numpy(hadamard(16)).float()
    factor_list = [f.factor for f in GBfactorize(dense, gb_params, [0, 1, 2], True)]
    error = torch.norm(dense - densification(factor_list, gb_params)[0, 0]) / torch.norm(dense)
    assert error < 1e-5, f"2D Hadamard target: error {error}"
    print("test_factorization_dense_target: OK")


if __name__ == "__main__":
    test_remove_redundancy()
    test_expand_factor()
    test_factorization_redundant_chain()
    test_factorization_noisy_redundant_chain()
    test_factorization_boundary_chain()
    test_dense_to_twiddle()
    test_factorization_dense_target()
