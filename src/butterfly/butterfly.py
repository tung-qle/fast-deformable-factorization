"""
Imported from Dao
"""

import torch.nn as nn
from torch.nn import functional as F
import torch
import math
import numbers
import numpy as np
import copy
from typing import Union, Tuple

from .butterfly_multiply import butterfly_multiply_torch, twiddle_base2_to_base4, butterfly_multiply_base4_torch

real_dtype_to_complex = {torch.float32: torch.complex64, torch.float64: torch.complex128}


class Butterfly(nn.Module):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    Compatible with torch.nn.Linear.

    Parameters:
        in_size: size of input
        out_size: size of output
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        complex: whether complex or real
        increasing_stride: whether the first butterfly block will multiply with increasing stride
            (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
        init: a torch.Tensor, or 'randn', 'ortho', 'identity', 'fft_no_br', or 'ifft_no_br'.
            Whether the weight matrix should be initialized to from randn twiddle, or to be
            randomly orthogonal/unitary, or to be the identity matrix, or the normalized FFT/iFFT
            twiddle (without the bit-reversal permutation).
        nblocks: number of B or B^T blocks. The B and B^T will alternate.
    """

    def __init__(self, in_size, out_size, bias=True, complex=False,
                 increasing_stride=True, init='randn', nblocks=1):
        super().__init__()
        self.in_size = in_size
        self.log_n = log_n = int(math.ceil(math.log2(in_size)))
        self.n = n = 1 << log_n
        self.out_size = out_size
        self.nstacks = int(math.ceil(out_size / self.n))
        self.complex = complex
        self.increasing_stride = increasing_stride
        assert nblocks >= 1
        self.nblocks = nblocks
        dtype = torch.get_default_dtype() if not self.complex else real_dtype_to_complex[torch.get_default_dtype()]
        twiddle_shape = (self.nstacks, nblocks, log_n, n // 2, 2, 2)
        if isinstance(init, torch.Tensor):
            self.init = None
            assert init.shape == twiddle_shape
            assert init.dtype == dtype
            self.twiddle = nn.Parameter(init.clone())
        else:
            assert init in ['empty', 'randn', 'ortho', 'identity', 'fft_no_br', 'ifft_no_br']
            self.init = init
            self.twiddle = nn.Parameter(torch.empty(twiddle_shape, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.twiddle._is_structured = True  # Flag to avoid weight decay
        self.saving = self.twiddle.numel() / (self.in_size * self.out_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)
        twiddle = self.twiddle
        if self.init is None or self.init == 'empty':
            return
        elif self.init == 'randn':
            # complex randn already has the correct scaling of stddev=1.0
            scaling = 1.0 / math.sqrt(2)
            with torch.no_grad():
                twiddle.copy_(torch.randn(twiddle.shape, dtype=twiddle.dtype) * scaling)
        elif self.init == 'ortho':
            twiddle_core_shape = twiddle.shape[:-2]
            if not self.complex:
                theta = torch.rand(twiddle_core_shape) * math.pi * 2
                c, s = torch.cos(theta), torch.sin(theta)
                det = torch.randint(0, 2, twiddle_core_shape, dtype=c.dtype) * 2 - 1  # Rotation (+1) or reflection (-1)
                with torch.no_grad():
                    twiddle.copy_(torch.stack((torch.stack((det * c, -det * s), dim=-1),
                                               torch.stack((s, c), dim=-1)), dim=-2))
            else:
                # Sampling from the Haar measure on U(2) is a bit subtle.
                # Using the parameterization here: http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
                phi = torch.asin(torch.sqrt(torch.rand(twiddle_core_shape)))
                c, s = torch.cos(phi), torch.sin(phi)
                alpha, psi, chi = torch.rand((3,) + twiddle_core_shape) * math.pi * 2
                A = torch.exp(1j * (alpha + psi)) * c
                B = torch.exp(1j * (alpha + chi)) * s
                C = -torch.exp(1j * (alpha - chi)) * s
                D = torch.exp(1j * (alpha - psi)) * c
                with torch.no_grad():
                    twiddle.copy_(torch.stack((torch.stack((A, B), dim=-1),
                                               torch.stack((C, D), dim=-1)), dim=-2))
        elif self.init == 'identity':
            twiddle_eye = torch.eye(2, dtype=twiddle.dtype).reshape(1, 1, 1, 1, 2, 2)
            twiddle_eye = twiddle_eye.expand(*twiddle.shape).contiguous()
            with torch.no_grad():
                twiddle.copy_(twiddle_eye)
        elif self.init in ['fft_no_br', 'ifft_no_br']:
            assert self.complex, 'fft_no_br/ifft_no_br init requires Butterfly to be complex'
            special_fn = (fft if self.init == 'fft_no_br'
                          else ifft)
            b_fft = special_fn(self.n, normalized=True, br_first=self.increasing_stride,
                               with_br_perm=False)
            with torch.no_grad():
                twiddle[:, 0] = b_fft.twiddle
            if self.nblocks > 1:
                twiddle_eye = torch.eye(2, dtype=twiddle.dtype).reshape(1, 1, 1, 1, 2, 2)
                twiddle_eye = twiddle_eye.expand(*twiddle[:, 1:].shape).contiguous()
                with torch.no_grad():
                    twiddle[:, 1:] = twiddle_eye

    def load_twiddle_from_butterfly_fact(self, all_twiddles):
        # Adapt my twiddle to Dao's twiddle. This has been tested in tests.models.layers.test_butterfly_linear.py
        assert self.in_size == self.out_size == self.n
        assert self.nstacks == self.nblocks == 1
        twiddle = torch.cat(
            [torch.reshape(twiddle, (self.nstacks, self.nblocks, 1, self.n // 2, 2, 2)) for twiddle in
             reversed(all_twiddles)],
            dim=2
        )
        self.twiddle.data = twiddle

    def forward(self, input, transpose=False, conjugate=False, subtwiddle=False):
        """
        Parameters:
            input: (batch, *, in_size)
            transpose: whether the butterfly matrix should be transposed.
            conjugate: whether the butterfly matrix should be conjugated.
            subtwiddle: allow using only part of the parameters for smaller input.
                Could be useful for weight sharing.
                out_size is set to self.nstacks * self.n in this case
        Return:
            output: (batch, *, out_size)
        """
        twiddle = self.twiddle
        output = self.pre_process(input)
        output_size = self.out_size if self.nstacks == 1 else None
        if subtwiddle:
            log_n = int(math.ceil(math.log2(input.size(-1))))
            n = 1 << log_n
            twiddle = (twiddle[:, :, :log_n, :n // 2] if self.increasing_stride
                       else twiddle[:, :, -log_n:, :n // 2])
            output_size = None
        if conjugate and self.complex:
            twiddle = twiddle.conj()
        if not transpose:
            output = butterfly_multiply_torch(twiddle, output, self.increasing_stride, output_size)
        else:
            twiddle = twiddle.transpose(-1, -2).flip([1, 2])
            last_increasing_stride = self.increasing_stride != ((self.nblocks - 1) % 2 == 1)
            output = butterfly_multiply_torch(twiddle, output, not last_increasing_stride, output_size)
        if not subtwiddle:
            return self.post_process(input, output)
        else:
            return self.post_process(input, output, out_size=output.size(-1))

    def pre_process(self, input):
        # Reshape to (batch, nstacks, in_size)
        input_size = input.size(-1)
        output = complex_reshape(input, -1, input_size)
        batch = output.shape[0]
        output = output.unsqueeze(1).expand(batch, self.nstacks, input_size)
        return output

    def post_process(self, input, output, out_size=None):
        if out_size is None:
            out_size = self.out_size
        batch = output.shape[0]
        output = output.view(batch, self.nstacks * output.size(-1))
        if out_size != output.shape[-1]:  # Take top rows
            output = output[:, :out_size]
        if self.bias is not None:
            output = output + self.bias[:out_size]
        return output.view(*input.size()[:-1], out_size)

    def __imul__(self, scale):
        """In-place multiply the whole butterfly matrix by some scale factor, by multiplying the
        twiddle.
        Scale must be nonnegative
        """
        assert isinstance(scale, numbers.Number)
        assert scale >= 0
        self.twiddle *= scale ** (1.0 / self.twiddle.shape[1] / self.twiddle.shape[2])
        return self

    def diagonal_multiply_(self, diagonal, diag_first):
        """ Combine a Butterfly and a diagonal into another Butterfly.
        Only support nstacks==1 for now.
        Parameters:
            diagonal: size (in_size,) if diag_first, else (out_size,). Should be of type complex
                if butterfly.complex == True.
            diag_first: If True, the map is input -> diagonal -> butterfly.
                If False, the map is input -> butterfly -> diagonal.
        """
        return diagonal_butterfly(self, diagonal, diag_first, inplace=True)

    def to_base4(self):
        with torch.no_grad():
            twiddle4, twiddle2 = twiddle_base2_to_base4(self.twiddle, self.increasing_stride)
        new = ButterflyBase4(self.in_size, self.out_size, self.bias is not None,
                             self.complex, self.increasing_stride,
                             init=(twiddle4, twiddle2),
                             nblocks=self.nblocks).to(self.twiddle.device)
        if new.bias is not None:
            with torch.no_grad():
                new.bias.copy_(self.bias)
        return new

    def extra_repr(self):
        s = 'in_size={}, out_size={}, bias={}, complex={}, increasing_stride={}, init={}, nblocks={}'.format(
            self.in_size, self.out_size, self.bias is not None, self.complex, self.increasing_stride, self.init,
            self.nblocks, )
        return s


def fft(n, normalized=False, br_first=True, with_br_perm=True) -> nn.Module:
    """ Construct an nn.Module based on Butterfly that exactly performs the FFT.
    Parameters:
        n: size of the FFT. Must be a power of 2.
        normalized: if True, corresponds to the unitary FFT (i.e. multiplied by 1/sqrt(n))
        br_first: which decomposition of FFT. br_first=True corresponds to decimation-in-time.
                  br_first=False corresponds to decimation-in-frequency.
        with_br_perm: whether to return both the butterfly and the bit reversal permutation.
    """
    log_n = int(math.ceil(math.log2(n)))
    assert n == 1 << log_n, 'n must be a power of 2'
    factors = []
    for log_size in range(1, log_n + 1):
        size = 1 << log_size
        exp = torch.exp(-2j * math.pi * torch.arange(0.0, size // 2) / size)
        o = torch.ones_like(exp)
        twiddle_factor = torch.stack((torch.stack((o, exp), dim=-1),
                                      torch.stack((o, -exp), dim=-1)), dim=-2)
        factors.append(twiddle_factor.repeat(n // size, 1, 1))
    twiddle = torch.stack(factors, dim=0).unsqueeze(0).unsqueeze(0)
    if not br_first:  # Take conjugate transpose of the BP decomposition of ifft
        twiddle = twiddle.transpose(-1, -2).flip([2])
    # Divide the whole transform by sqrt(n) by dividing each factor by n^(1/2 log_n) = sqrt(2)
    if normalized:
        twiddle /= math.sqrt(2)
    b = Butterfly(n, n, bias=False, complex=True, increasing_stride=br_first, init=twiddle)
    if with_br_perm:
        br_perm = FixedPermutation(bitreversal_permutation(n, pytorch_format=True))
        return nn.Sequential(br_perm, b) if br_first else nn.Sequential(b, br_perm)
    else:
        return b


def ifft(n, normalized=False, br_first=True, with_br_perm=True) -> nn.Module:
    """ Construct an nn.Module based on Butterfly that exactly performs the inverse FFT.
    Parameters:
        n: size of the iFFT. Must be a power of 2.
        normalized: if True, corresponds to unitary iFFT (i.e. multiplied by 1/sqrt(n), not 1/n)
        br_first: which decomposition of iFFT. True corresponds to decimation-in-frequency.
                  False corresponds to decimation-in-time.
        with_br_perm: whether to return both the butterfly and the bit reversal permutation.
    """
    log_n = int(math.ceil(math.log2(n)))
    assert n == 1 << log_n, 'n must be a power of 2'
    factors = []
    for log_size in range(1, log_n + 1):
        size = 1 << log_size
        exp = torch.exp(2j * math.pi * torch.arange(0.0, size // 2) / size)
        o = torch.ones_like(exp)
        twiddle_factor = torch.stack((torch.stack((o, exp), dim=-1),
                                      torch.stack((o, -exp), dim=-1)), dim=-2)
        factors.append(twiddle_factor.repeat(n // size, 1, 1))
    twiddle = torch.stack(factors, dim=0).unsqueeze(0).unsqueeze(0)
    if not br_first:  # Take conjugate transpose of the BP decomposition of fft
        twiddle = twiddle.transpose(-1, -2).flip([2])
    # Divide the whole transform by sqrt(n) by dividing each factor by n^(1/2 log_n) = sqrt(2)
    if normalized:
        twiddle /= math.sqrt(2)
    else:
        twiddle /= 2
    b = Butterfly(n, n, bias=False, complex=True, increasing_stride=br_first, init=twiddle)
    if with_br_perm:
        br_perm = FixedPermutation(bitreversal_permutation(n, pytorch_format=True))
        return nn.Sequential(br_perm, b) if br_first else nn.Sequential(b, br_perm)
    else:
        return b


def complex_reshape(x, *shape):
    if not x.is_complex():
        return x.reshape(*shape)
    else:
        return torch.view_as_complex(torch.view_as_real(x).reshape(*shape, 2))


def diagonal_butterfly(butterfly: Butterfly,
                       diagonal: torch.Tensor,
                       diag_first: bool,
                       inplace: bool = True) -> Butterfly:
    """
    Combine a Butterfly and a diagonal into another Butterfly.
    Only support nstacks==1 for now.
    Parameters:
        butterfly: Butterfly(in_size, out_size)
        diagonal: size (in_size,) if diag_first, else (out_size,). Should be of type complex
            if butterfly.complex == True.
        diag_first: If True, the map is input -> diagonal -> butterfly.
            If False, the map is input -> butterfly -> diagonal.
        inplace: whether to modify the input Butterfly
    """
    assert butterfly.nstacks == 1
    assert butterfly.bias is None
    twiddle = butterfly.twiddle.clone()
    n = 1 << twiddle.shape[2]
    if diagonal.shape[-1] < n:
        diagonal = F.pad(diagonal, (0, n - diagonal.shape[-1]), value=1)
    if diag_first:
        if butterfly.increasing_stride:
            twiddle[:, 0, 0, :, :, 0] *= diagonal[::2].unsqueeze(-1)
            twiddle[:, 0, 0, :, :, 1] *= diagonal[1::2].unsqueeze(-1)
        else:
            n = diagonal.shape[-1]
            twiddle[:, 0, 0, :, :, 0] *= diagonal[:n // 2].unsqueeze(-1)
            twiddle[:, 0, 0, :, :, 1] *= diagonal[n // 2:].unsqueeze(-1)
    else:
        # Whether the last block is increasing or decreasing stride
        increasing_stride = butterfly.increasing_stride != ((butterfly.nblocks - 1) % 2 == 1)
        if increasing_stride:
            n = diagonal.shape[-1]
            twiddle[:, -1, -1, :, 0, :] *= diagonal[:n // 2].unsqueeze(-1)
            twiddle[:, -1, -1, :, 1, :] *= diagonal[n // 2:].unsqueeze(-1)
        else:
            twiddle[:, -1, -1, :, 0, :] *= diagonal[::2].unsqueeze(-1)
            twiddle[:, -1, -1, :, 1, :] *= diagonal[1::2].unsqueeze(-1)
    out_butterfly = butterfly if inplace else copy.deepcopy(butterfly)
    with torch.no_grad():
        out_butterfly.twiddle.copy_(twiddle)
    return out_butterfly


class ButterflyBase4(Butterfly):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    Compatible with torch.nn.Linear.

    Parameters:
        in_size: size of input
        out_size: size of output
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        complex: whether complex or real
        increasing_stride: whether the first butterfly block will multiply with increasing stride
            (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
        init: 'randn', 'ortho', or 'identity'. Whether the weight matrix should be initialized to
            from randn twiddle, or to be randomly orthogonal/unitary, or to be the identity matrix.
        nblocks: number of B or B^T blocks. The B and B^T will alternate.
    """

    def __init__(self, *args, **kwargs):
        init = kwargs.get('init', None)
        if (isinstance(init, tuple) and len(init) == 2 and isinstance(init[0], torch.Tensor)
                and isinstance(init[1], torch.Tensor)):
            twiddle4, twiddle2 = init[0].clone(), init[1].clone()
            kwargs['init'] = 'empty'
            super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
            with torch.no_grad():
                twiddle4, twiddle2 = twiddle_base2_to_base4(self.twiddle, self.increasing_stride)
        del self.twiddle
        self.twiddle4 = nn.Parameter(twiddle4)
        self.twiddle2 = nn.Parameter(twiddle2)
        self.twiddle4._is_structured = True  # Flag to avoid weight decay
        self.twiddle2._is_structured = True  # Flag to avoid weight decay

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, in_size)
        Return:
            output: (batch, *, out_size)
        """
        output = self.pre_process(input)
        output_size = self.out_size if self.nstacks == 1 else None
        output = butterfly_multiply_base4_torch(self.twiddle4, self.twiddle2, output,
                                                self.increasing_stride, output_size)
        return self.post_process(input, output)

    def __imul__(self, scale):
        """In-place multiply the whole butterfly matrix by some scale factor, by multiplying the
        twiddle.
        Scale must be nonnegative
        """
        assert isinstance(scale, numbers.Number)
        assert scale >= 0
        scale_per_entry = scale ** (1.0 / self.nblocks / self.log_n)
        self.twiddle4 *= scale_per_entry ** 2
        self.twiddle2 *= scale_per_entry
        return self


def bitreversal_permutation(n, pytorch_format=False):
    """Return the bit reversal permutation used in FFT.
    By default, the permutation is stored in numpy array.
    Parameter:
        n: integer, must be a power of 2.
        pytorch_format: whether the permutation is stored as numpy array or pytorch tensor.
    Return:
        perm: bit reversal permutation, numpy array of size n
    """
    log_n = int(math.log2(n))
    assert n == 1 << log_n, 'n must be a power of 2'
    perm = np.arange(n).reshape(n, 1)
    for i in range(log_n):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    perm = perm.squeeze(0)
    return perm if not pytorch_format else torch.tensor(perm, dtype=torch.long)


class FixedPermutation(nn.Module):

    def __init__(self, permutation: torch.Tensor) -> None:
        """Fixed permutation.
        Parameter:
            permutation: (n, ) tensor of ints
        """
        super().__init__()
        self.register_buffer('permutation', permutation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            input: (batch, *, size)
        Return:
            output: (batch, *, size)
        """
        # return input[..., self.permutation]
        # Pytorch 1.7 doesn't have indexing_backward for complex.
        # So we use our own backward
        return IndexLastDim.apply(input, self.permutation)

    def to_butterfly(self, complex=False, increasing_stride=False):
        return perm2butterfly(self.permutation, complex, increasing_stride)


class IndexLastDim(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, permutation):
        ctx.save_for_backward(permutation)
        return X[..., permutation]

    @staticmethod
    def backward(ctx, grad):
        permutation, = ctx.saved_tensors
        output = torch.empty_like(grad)
        output[..., permutation] = grad
        return output, None


def perm2butterfly(v: Union[np.ndarray, torch.Tensor],
                   complex: bool = False,
                   increasing_stride: bool = False) -> Butterfly:
    """
    Parameter:
        v: a permutation, stored as a vector, in left-multiplication format.
            (i.e., applying v to a vector x is equivalent to x[p])
        complex: whether the Butterfly is complex or real.
        increasing_stride: whether the returned Butterfly should have increasing_stride=False or
            True. False corresponds to Lemma G.3 and True corresponds to Lemma G.6.
    Return:
        b: a Butterfly that performs the same permutation as v.
    """
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    n = len(v)
    log_n = int(math.ceil(math.log2(n)))
    if n < 1 << log_n:  # Pad permutation to the next power-of-2 size
        v = np.concatenate([v, np.arange(n, 1 << log_n)])
    if increasing_stride:  # Follow proof of Lemma G.6
        br = bitreversal_permutation(1 << log_n)
        b = perm2butterfly(br[v[br]], complex=complex, increasing_stride=False)
        b.increasing_stride = True
        br_half = bitreversal_permutation((1 << log_n) // 2, pytorch_format=True)
        with torch.no_grad():
            b.twiddle.copy_(b.twiddle[:, :, :, br_half])
        b.in_size = b.out_size = n
        return b
    v = v[None]
    twiddle_right_factors, twiddle_left_factors = [], []
    for _ in range(log_n):
        right_factor, left_factor, v = outer_twiddle_factors(v)
        twiddle_right_factors.append(right_factor)
        twiddle_left_factors.append(left_factor)
    twiddle = torch.stack([torch.stack(twiddle_right_factors),
                           torch.stack(twiddle_left_factors).flip([0])]).unsqueeze(0)
    b = Butterfly(n, n, bias=False, complex=complex, increasing_stride=False,
                  init=twiddle if not complex else Real2ComplexFn.apply(twiddle), nblocks=2)
    return b


def outer_twiddle_factors(v: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Decompose the permutations v to get the right/right twiddle factor, and new permutations
    that only permute elements that are size//2 indices apart.
    Parameters:
        v: (batch_size, size), each is a permutation vector of size @size, in left-multiplication
            format.
    Return:
        twiddle_right_factor: (batch_size * size // 2, 2, 2)
        twiddle_left_factor: (batch_size * size // 2, 2, 2)
        new_v: (batch_size * 2, size // 2)
    """
    # Convert to right-multiplication format since that's what half_balance expects
    batch_size, size = v.shape
    assert size >= 2
    v_right = np.vstack([invert(chunk) for chunk in v])
    half_balance_results = [half_balance(chunk, return_swap_locations=True)
                            for chunk in v_right]
    twiddle_right_factor = torch.cat(
        [swap_locations_to_twiddle_factor(size, swap_low_locs)
         for swap_low_locs, _ in half_balance_results]
    )
    v_right = np.vstack([v_permuted for _, v_permuted in half_balance_results])
    v_left = np.vstack([invert(perm) for perm in v_right])
    size_half = size // 2
    swap_low_x, swap_low_y = np.nonzero(v_left[:, :size_half] // size_half == 1)
    swap_low_locs_flat = swap_low_y + swap_low_x * size // 2
    twiddle_left_factor = swap_locations_to_twiddle_factor(batch_size * size, swap_low_locs_flat)
    v_left[swap_low_x, swap_low_y], v_left[swap_low_x, swap_low_y + size_half] = (
        v_left[swap_low_x, swap_low_y + size // 2], v_left[swap_low_x, swap_low_y]
    )
    new_v = (v_left % size_half).reshape(batch_size * 2, size // 2)
    # Check that each new vector is a permutation
    assert np.allclose(np.sort(new_v), np.arange(size // 2))
    return twiddle_right_factor, twiddle_left_factor, new_v


def swap_locations_to_twiddle_factor(n: int, swap_locations: np.ndarray) -> torch.Tensor:
    twiddle = torch.eye(2).expand(n // 2, 2, 2).contiguous()
    swap_matrix = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
    twiddle[swap_locations] = swap_matrix.unsqueeze(0)
    return twiddle


def half_balance(
        v: np.ndarray, return_swap_locations: bool = False
) -> Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]:
    """Return the permutation vector that makes the permutation vector v
    n//2-balanced. Directly follows the proof of Lemma G.2.
    Parameters:
        v: the permutation as a vector, stored in right-multiplication format.
    """
    n = len(v)
    assert n % 2 == 0
    nh = n // 2
    nodes = [Node(i) for i in range(nh)]
    # Build the graph
    for i in range(nh):
        # There is an edge from s to t
        s, t = nodes[v[i] % nh], nodes[v[i + nh] % nh]
        s.out_edges.append((t, i))
        t.in_edges.append((s, i + nh))
    # Each node has undirected degree exactly 2
    assert all(len(node.in_edges) + len(node.out_edges) == 2 for node in nodes)
    swap_low_locs = []
    swap_high_locs = []
    while len(nodes):
        # Pick a random node.
        start_node, start_loc = nodes[-1], n - 1
        next_node = None
        # Follow undirected edges until rereaching start_node.
        # As every node has undirected degree 2, this will find
        # all cycles in the graph. Reverse edges as needed to
        # make the cycle a directed cycle.
        while next_node != start_node:
            if next_node is None:
                next_node, next_loc = start_node, start_loc
            old_node, old_loc = next_node, next_loc
            if old_node.out_edges:
                # If there's an out-edge from old_node, follow it.
                next_node, old_loc = old_node.out_edges.pop()
                next_loc = old_loc + nh
                next_node.in_edges.remove((old_node, next_loc))
            else:
                # If there's no out-edge, there must be an in-edge.
                next_node, old_loc = old_node.in_edges.pop()
                next_loc = old_loc - nh
                next_node.out_edges.remove((old_node, next_loc))
                swap_low_locs.append(next_loc)
                swap_high_locs.append(old_loc)
            nodes.remove(old_node)
    perm = np.arange(n, dtype=int)
    perm[swap_low_locs], perm[swap_high_locs] = swap_high_locs, swap_low_locs
    if not return_swap_locations:
        return perm, v[perm]
    else:
        return swap_low_locs, v[perm]


class Node:
    def __init__(self, value):
        self.value = value
        self.in_edges = []
        self.out_edges = []


def invert(perm: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Get the inverse of a given permutation vector.
    Equivalent to converting a permutation vector from left-multiplication format to right
    multiplication format.
    Work with both numpy array and Pytorch Tensor.
    """
    assert isinstance(perm, (np.ndarray, torch.Tensor))
    n = perm.shape[-1]
    if isinstance(perm, np.ndarray):
        result = np.empty(n, dtype=int)
        result[perm] = np.arange(n, dtype=int)
    else:
        result = torch.empty(n, dtype=int, device=perm.device)
        result[perm] = torch.arange(n, dtype=int)
    return result


class Real2ComplexFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        return X.to(real_dtype_to_complex[X.dtype])

    @staticmethod
    def backward(ctx, grad):
        return grad.real
