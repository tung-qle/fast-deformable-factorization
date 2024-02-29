results_dir="results/fact_vs_iter_vs_als/anchiale1"
lr=0.1
n_adam_epochs=100
n_lbfgs_epochs=20
n_als_epochs=5

for k in 9 10 11 12 13
do
  # hierarchical algo
  for orthonormalize in True False
  do

    python comparison_new.py \
      --results-dir $results_dir \
      --k $k \
      --method hierarchical \
      --orthonormalize $orthonormalize
  done

  # gradient algo
  python comparison_new.py \
    --results-dir $results_dir \
    --k $k \
    --method gradient \
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