# Command to reproduce paper result on our generated dataset:
conda activate graph_transformer_gpu

CONFIG1="../../configs/new config/basic"
CONFIG2="../../configs/new config/with_PosEco"
CONFIG3="../../configs/new config/with_posEcoConcat"
cd ../../data/SBMs
python generate_SBM.py --p1 0.5 --q 0.25

cd ../../

python prepare_data.py --p1 0.5

for file in "$CONFIG1"/*;
do
  python main_SBMs_node_classification.py --gpu_id 0 --config "${file}";
done

for file in "$CONFIG2"/*;
do
  python main_SBMs_node_classification.py --gpu_id 0 --config "${file}";
done

for file in "$CONFIG3"/*;
do
  python main_SBMs_node_classification.py --gpu_id 0 --config "${file}";
done

read -p "Press any key to resume ..."