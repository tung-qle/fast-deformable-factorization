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
