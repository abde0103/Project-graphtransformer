# Graph Transformer Architecture
## 1. Updates and experiments

For graph transformers described below, we tested (cf presentation.pdf) : 
- Generation of SBM datasets using different intra-communities and inter-communities densities. 
- Model performances (accuracy on test set) for different Signal-Noise-ratio (fixed q and variable p).
- Model performances for different hidden dimension (cf "**[A Generalization of Transformer Networks to Graphs](https://arxiv.org/abs/2012.09699)**"). 
- concatenating the Laplacian positional encoding and the input instead of summing them for full attention architecture and sparse attention architecture.
- Testing different Positional encoding methods : without/ with LaplacianPE/ WL-PE.


## 2. Added scripts
- data/SBMs/generate_SBM.py : generate SBM datasets with custom parameters.
- data/SBMs/utils_generate_SBM.py : Useful functions for data/SBMs/generate_SBM.py.
- prepare_data.py : put the generated data in the right format for the main_scripts.
- data/SBMs/SBMs_generated.py : Useful to build SBMsDataset graphs, only used when training the model.
- nets/SBMs_node_classification/graph_transformer_net_concat.py : Architecture with LaplacianPE concatenation instead of a simple addition.
- All the scripts in scripts/MAP583 : see below.
- All the files in the folder results : basic results for presentation.pdf.
- results/SNR_results/analyse_results.ipynb : notebook to reproduce the plots in presentation.pdf.
- presentation.pdf

## 3. Dataset generation
- To generate a dataset, please run : 
```
cd data/SBMs
python generate_SBM.py --p1 0.5 --p2 0.5 --q 0.2 --size_min 80 --size_max 120 --size_min_test 80 --size_max_test 120
```
- To put it in the right format for the main_script, please run : 
```
cd ../../
python prepare_data.py --p1 0.5   ## in the root directory
```
## 4. New results reproducibility
- First you have to activate graph_transformer env
```
conda activate graph_transformer_gpu ## if gpu instructions were followed
conda activate graph_transformer ## if cpu instructions were followed
```
- To reproduce the results of Accuracy in terms of SNR while using Batch-norm (cf presentation.pdf) in the graph transformer, please run :
```
bash scripts/MAP583/accuracy_SNR_Batch_norm.sh
```

- To reproduce the results of Accuracy in terms of SNR while using Layer-norm (cf presentation.pdf) in the graph transformer, please run :
```
bash scripts/MAP583/accuracy_SNR_Layer_norm.sh
``` 
- To reproduce the results of test generalisation (cf presentation.pdf), please run :
```
bash scripts/MAP583/test_generalisation.sh
```
- To reproduce the comparative table beetween generated dataset and orignal SBM dataset on all tests in the paper (sparse graph, full graph, with PE, without PE, etc...) and between sum laplacien encoding and concat laplacien encoding (cf presentation.pdf) :
```
bash scripts/MAP583/paper_result_replication_&_concatNormalisation.sh
```

# II. Readme content of the source code of the paper

Source code for the paper "**[A Generalization of Transformer Networks to Graphs](https://arxiv.org/abs/2012.09699)**" by _[Vijay Prakash Dwivedi](https://github.com/vijaydwivedi75) and [Xavier Bresson](https://github.com/xbresson)_, at **AAAI'21 Workshop on Deep Learning on Graphs: Methods and Applications (DLG-AAAI'21)**.

We propose a generalization of transformer neural network architecture for arbitrary graphs: **Graph Transformer**. <br>Compared to the [Standard Transformer](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), the highlights of the presented architecture are: 

- The attention mechanism is a function of neighborhood connectivity for each node in the graph.  
- The position encoding is represented by Laplacian eigenvectors, which naturally generalize the sinusoidal positional encodings often used in NLP.  
- The layer normalization is replaced by a batch normalization layer.  
- The architecture is extended to have edge representation, which can be critical to tasks with rich information on the edges, or pairwise interactions (such as bond types in molecules, or relationship type in KGs. etc). 

<br>

<p align="center">
  <img src="./docs/graph_transformer.png" alt="Graph Transformer Architecture" width="800">
  <br>
  <b>Figure</b>: Block Diagram of Graph Transformer Architecture
</p>


## 1. Repo installation

This project is based on the [benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns) repository.

[Follow these instructions](./docs/01_benchmark_installation.md) to install the benchmark and setup the environment.


<br>

## 2. Download datasets

[Proceed as follows](./docs/02_download_datasets.md) to download the datasets used to evaluate Graph Transformer.


<br>

## 3. Reproducibility of the original paper 

[Use this page](./docs/03_run_codes.md) to run the codes and reproduce the published results.


<br>

## 4. Reference 

:page_with_curl: Paper [on arXiv](https://arxiv.org/abs/2012.09699)    
:pencil: Blog [on Towards Data Science](https://towardsdatascience.com/graph-transformer-generalization-of-transformers-to-graphs-ead2448cff8b)    
:movie_camera: Video [on YouTube](https://www.youtube.com/watch?v=h-_HNeBmaaU&t=237s)    
```
@article{dwivedi2021generalization,
  title={A Generalization of Transformer Networks to Graphs},
  author={Dwivedi, Vijay Prakash and Bresson, Xavier},
  journal={AAAI Workshop on Deep Learning on Graphs: Methods and Applications},
  year={2021}
}
```


<br><br><br>

