# Command to reproduce paper result on our generated dataset:
conda activate graph_transformer_gpu

CONFIG1="../../configs/new config/basic"
CONFIG2="../../configs/new config/with_PosEco"
CONFIG3="../../configs/new config/with_posEcoConcat"


for file in "$CONFIG1"/*;
do
  python ../../main_SBMs_node_classification.py --gpu_id 0 --config ../../$file;
done

for file in "$CONFIG2"/*;
do
  python ../../main_SBMs_node_classification.py --gpu_id 0 --config $file;
done

for file in "$CONFIG3"/*;
do
  python ../../main_SBMs_node_classification.py --gpu_id 0 --config $file;
done

read -p "Press any key to resume ..."