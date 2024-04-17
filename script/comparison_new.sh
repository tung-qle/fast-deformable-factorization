results_dir="results/fact_vs_iter_vs_als/17_04/anchiale1"
lr=0.1
n_adam_epochs=100
n_lbfgs_epochs=20
n_als_epochs=5

for k in 9 10 11 12
do
  # hierarchical algo
  for orthonormalize in True False
  do
    for hierarchical_order in left-to-right balanced
    do
      python comparison_new.py \
        --results-dir $results_dir \
        --k $k \
        --method hierarchical \
        --orthonormalize $orthonormalize \
        --hierarchical-order $hierarchical_order
    done
  done

  # gradient algo
  python comparison_new.py \
    --results-dir $results_dir \
    --k $k \
    --method gradient \
    --lr $lr \
    --n_adam_epochs $n_adam_epochs \
    --n_lbfgs_epochs $n_lbfgs_epochs

  # square dyadic gradient dao
  python comparison_new.py \
    --results-dir $results_dir \
    --k $k \
    --method square-dyadic-gradient-dao \
    --lr $lr \
    --n_adam_epochs $n_adam_epochs \
    --n_lbfgs_epochs $n_lbfgs_epochs

  # als algo
  python comparison_new.py \
    --results-dir $results_dir \
    --k $k \
    --method als \
    --n_als_epochs $n_als_epochs
done