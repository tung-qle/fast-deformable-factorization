# Butterfly Factorization with Error Guarantees

Official code to reproduce the results of the paper:

> **Butterfly Factorization with Error Guarantees**
> Quoc-Tung Le, Léon Zheng, Elisa Riccietti, Rémi Gribonval
> *SIAM Journal on Matrix Analysis and Applications*, **46**(4), 2253–2309, 2025.
> DOI: [10.1137/24M1708796](https://doi.org/10.1137/24M1708796)

## Installation

```bash
# (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies (CPU-only torch)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# install the package (exposes the `src` module)
pip install -e .
```

## Get started

`test_fac.py` factorizes a Hadamard transform into deformable butterfly factors and
reports the reconstruction error and running time:

```bash
python test_fac.py
# options: --n (matrix size 2**n), --rank (inner rank of the architecture,
# redundant when >= 2), --random-order (random factorization order)
```

The core call is `GBfactorize`, which takes the target matrix, the butterfly parameters
`gb_params`, an `orders` list defining the factorization tree, and a boolean toggling
the orthonormalization heuristic:

```python
import torch
from scipy.linalg import hadamard
from src.GB_factorization import GBfactorize
from src.GB_operators import densification

n = 13
N = 2 ** n

# Hadamard matrix of size N x N, shaped as (1, 1, N, N)
matrix = torch.from_numpy(hadamard(N)).reshape(1, 1, N, N).float()

# butterfly parameters and factorization order
gb_params = [(2 ** i, 2, 2, 2 ** (n - i - 1), 1, 1) for i in range(n)]
orders = list(range(n - 1))

# factorize, then reconstruct
factor_list = GBfactorize(matrix, gb_params, orders, True)
factor_list = [f.factor for f in factor_list]

error = torch.norm(matrix - densification(factor_list, gb_params))
print("Error:", error)
```

`GBfactorize` requires a non-redundant architecture (Definition 4.18 of the paper).
For an arbitrary chainable architecture, redundant or not, use `GBfactorize_auto`
(same interface, `orders` optional): it removes the redundancy of the architecture
(Algorithm 4.1), factorizes with the reduced non-redundant architecture, and lifts
the factorization back to the original one with the same approximation error
(Remark 6.4). See `test_redundancy.py` for examples.

The target can be any dense matrix, of size `(M, N)` or `(1, 1, M, N)`, whatever the
chainable architecture. The first factor need not have `a = 1` nor the last one `d = 1`:
the product of the chain is then not a full matrix, but a block diagonal matrix with `a`
blocks, each a `b * p` by `c * q` block matrix of diagonal blocks of size `d`, where
`(a, b, c, d, p, q)` is the product of `gb_params`. The entries of the target outside
this support cannot be reached by any factors of this architecture, so they do not
change the best factors: the algorithm ignores them, and their norm is a lower bound on
the approximation error. Use `product_of_twiddles` to reconstruct (`densification`
requires `a_1 = 1`):

```python
from src.GB_factorization import GBfactorize_auto
from src.GB_operators import product_of_twiddles, twiddle_to_dense

# Square-dyadic chain for N = 16 without its left factor: the product is diag(B1, B2)
gb_params = [(2, 2, 2, 4, 1, 1), (4, 2, 2, 2, 1, 1), (8, 2, 2, 1, 1, 1)]

factor_list = [f.factor for f in GBfactorize_auto(matrix, gb_params)]  # matrix: 16 x 16
approximation = twiddle_to_dense(product_of_twiddles(factor_list, gb_params))  # 16 x 16
```

Here the approximation of a full matrix is at best its two diagonal blocks of size 8. To
approximate the whole matrix, use a chain with `a_1 = d_L = 1`, for instance
`[(1, 4, 4, 4, 1, 1), (4, 2, 2, 2, 1, 1), (8, 2, 2, 1, 1, 1)]`. The target can also be
given in twiddle format, of size `(a, d, b * p, c * q)`; `dense_to_twiddle` converts a
dense matrix to this format.

## Scripts to reproduce figures of the paper

* Figure 3: [`script/comparison_new.sh`](https://github.com/tung-qle/fast-deformable-factorization/blob/master/script/comparison_new.sh)
* Figure 4: [`script/hierarchical_comp_new.sh`](https://github.com/tung-qle/fast-deformable-factorization/blob/master/script/hierarchical_comp_new.sh)
* Figure 5: [`script/bound_comparison.py`](https://github.com/tung-qle/fast-deformable-factorization/blob/master/script/bound_comparison.py)

## Citation

If you use this code, please cite:

```bibtex
@article{le2025butterfly,
  title   = {Butterfly Factorization with Error Guarantees},
  author  = {Le, Quoc-Tung and Zheng, L{\'e}on and Riccietti, Elisa and Gribonval, R{\'e}mi},
  journal = {SIAM Journal on Matrix Analysis and Applications},
  volume  = {46},
  number  = {4},
  pages   = {2253--2309},
  year    = {2025},
  doi     = {10.1137/24M1708796}
}
```
