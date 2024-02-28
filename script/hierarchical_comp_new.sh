n_factors=4
rank=2
noise=0.1

save_dir="./results/hierarchical_comp_new"

for matrix_size in 128 256 512 1024 2048 4096 8192
do
  for seed_target_matrix in 0 1 2 3 4 5 6 7 8 9
  do
    for orthonormalize in True False
    do
      for hierarchical_order in left-to-right balanced
      do
        python hierarchical_comp_new.py \
          --n-factors $n_factors \
          --rank $rank \
          --noise $noise \
          --matrix-size $matrix_size \
          --seed-target-matrix $seed_target_matrix \
          --orthonormalize $orthonormalize \
          --hierarchical-order $hierarchical_order \
          --save-dir $save_dir
      done
    done
  done
done