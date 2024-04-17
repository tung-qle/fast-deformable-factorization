from torchvision import models
import argparse
import src.GB_factorization as fact
import src.GB_operators as operator
import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["vit-b16"])
    parser.add_argument("--nb-blocks", type=int, default=4)
    return parser.parse_args()


def monarch_architecture(out_size, in_size, nb_blocks):
    assert in_size % nb_blocks == 0
    min_in_out = min(in_size, out_size)
    assert min_in_out % nb_blocks == 0
    assert out_size % nb_blocks == 0
    tuple0 = (nb_blocks, min_in_out // nb_blocks, in_size // nb_blocks, 1)
    tuple1 = (1, out_size // nb_blocks, min_in_out // nb_blocks, nb_blocks)

    assert tuple1[0] * tuple1[2] % tuple0[0] == 0  # a1 * c1 % a2 == 0
    assert tuple0[1] * tuple0[3] % tuple1[3] == 0  # b2 * d2 % d1 == 0
    assert tuple1[0] * tuple1[2] // tuple0[0] == tuple0[1] * tuple0[3] // tuple1[3]
    q = tuple1[0] * tuple1[2] // tuple0[0]

    return [(tuple1[0], tuple1[1], tuple1[2] // q, tuple1[3], 1, q),
            (tuple0[0], tuple0[1] // q, tuple0[2], tuple0[3], q, 1)]


def count_parameters(architecture):
    return sum([a * b * c * d * e * f for a, b, c, d, e, f in architecture])


def project_butterfly(matrix, architecture, orders):
    factor_list = fact.GBfactorize(matrix.unsqueeze(0).unsqueeze(0), architecture, orders, True)
    factor_list = [f.factor for f in factor_list]
    return (torch.norm(matrix - operator.densification(factor_list, architecture)) / torch.norm(matrix)).item()


def compare_projection(matrix, nb_blocks):
    out_size, in_size = matrix.shape
    min_in_out = min(in_size, out_size)
    assert min_in_out % nb_blocks == 0
    rank = min_in_out // nb_blocks

    monarch_arch = monarch_architecture(out_size, in_size, nb_blocks)
    low_rank_arch = [(1, out_size, 1, 1, 1, rank), (1, 1, in_size, 1, rank, 1)]
    assert count_parameters(monarch_arch) == count_parameters(low_rank_arch)

    error_monarch = project_butterfly(matrix, monarch_arch, [0])
    error_low_rank = project_butterfly(matrix, low_rank_arch, [0])
    # print(f"Monarch error: {error_monarch:.3f}. Low-rank error: {error_low_rank:.3f}")
    print(f"${error_monarch:.3f}$ & ${error_low_rank:.3f}$")
    return error_monarch, error_low_rank


if __name__ == '__main__':
    args = parse_arguments()

    if args.arch == "vit-b16":
        model = models.vit_b_16(pretrained=True)
    else:
        raise NotImplementedError

    # extract weight matrices of feedforward layers
    for i, layer in enumerate(model.encoder.layers):
        if i == 0 or i == 11:
            print(f"Layer {i}")
            compare_projection(layer.mlp[0].weight.data, args.nb_blocks)
            compare_projection(layer.mlp[3].weight.data, args.nb_blocks)

            # extract weight matrices of self-attention layers
            q, k, v = layer.self_attention.in_proj_weight.data.chunk(3, dim=0)
            compare_projection(q, args.nb_blocks)
            compare_projection(k, args.nb_blocks)
            compare_projection(v, args.nb_blocks)
