#!/bin/bash
# Command to train
for size in 25 100 200
do
	cd data/SBMs
	python generate_SBM.py --p1 0.3 --q 0.1 --size_max 50 --size_min 50 --size_max_test $size --size_min_test $size
	cd ../../
	python prepare_data.py --p1 0.3
	python main_SBMs_node_classification.py --gpu_id 0 --epochs 50 --dataset "p_0.30.3" --config "configs/old config/SBMs_GraphTransformer_PATTERN_500k_sparse_graph_BN.json";
done

