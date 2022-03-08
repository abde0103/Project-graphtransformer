#!/bin/bash
# Command to reproduce Accuracy in terms of SNR  (Batch norm in the attentions layers is used)
STR="p_"
for p1 in 0.11 0.12 0.13 0.135 0.14 0.17 0.3 0.4 0.75
do 
  cd data/SBMs
  python generate_SBM.py --p1 $p1 --q 0.1  
  cd ../../
  python prepare_data.py --p1 $p1
  python main_SBMs_node_classification.py --gpu_id 0 --epochs 50 --dataset "$STR$p1$p1" --config "configs/old config/SBMs_GraphTransformer_PATTERN_500k_sparse_graph_BN.json";
done
