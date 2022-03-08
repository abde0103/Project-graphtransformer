# Command to download dataset:
#   bash script_download_SBMs.sh


CONFIG="configs/new config/test_params"

for file in "$CONFIG"/*;
do
  python main_SBMs_node_classification.py --gpu_id 0 --config $file;
done



