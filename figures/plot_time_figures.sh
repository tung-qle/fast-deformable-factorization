for rank in 1 2 4
do
  for hierarchical_order in left-to-right balanced
  do
    for graph in time error
    do
      for noise in 0.1 0.3 0.01 0.03
      do
        python figures/hierarchical_comp_new_figure.py \
          --results-dir results/hierarchical_comp_new/anchiale1-2024-03-11 \
          --n-factors 4 \
          --noise $noise \
          --rank $rank \
          --hierarchical-order $hierarchical_order \
          --graph $graph
      done
    done
  done
done
