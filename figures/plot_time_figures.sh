for rank in 1 2 4
do
  for hierarchical_order in left-to-right balanced
  do
    python hierarchical_comp_new_figure.py \
      --results-dir results/hierarchical_comp_new/anchiale1 \
      --n-factors 4 \
      --noise 0.1 \
      --rank $rank \
      --hierarchical-order $hierarchical_order
  done
done
