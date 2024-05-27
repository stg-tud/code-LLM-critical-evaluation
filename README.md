We provide the code to generate all data and results. However, it requires a lot of space and time to generate everything. So, we have also provided intermediate results that the code
generates and a jupyter notebook to generate all plots and results presented in the paper.

The code has been tested on `Ubuntu 22.04` with `python 3.9.16`.

There are two directories - `attention` and `DirectProbe`. The code for `DirectProbe` has been taken from [DirectProbe](https://github.com/utahnlp/DirectProbe). We have only changed the `main.py` to pass different config files. The
two directories needs to be at the same level.

### Getting started
1. Create aconda environment and install the required packages. 

	```
    # In attention/ directory,
	conda create -n attention python=3.9.16
    pip install -r requirements.txt
	```

2. To set up the DirectProbe code, follow the instructions from [DirectProbe](https://github.com/utahnlp/DirectProbe). Note that DirectProbe requires Gurobi but can also run without it. However, running without Gurobi results into unstable results.

	We ran the the experiments with Gurobi and provide all the results generated during our experiments in the `DirectProbe/results` directory.
	We ran DirectProbe only for layers 5,9 and 12 and so provide results only for these layers.

### Generating plots and results
All the results provided in the paper is available in `results_attention_tsne.ipynb` and `results_hidden_repr.ipynb`. Before running the notebooks,
note that 
attention distribution and attention maps both require self-attention values to be saved first and t-SNE plots require 
hidden representations are saved beforehand.  

### Attention Analysis
We provide various splits of randomly sampled 3000 codes in `exp_data/`. We performed our experiments with `exp_0.jsonl`, but any other set of codes should work and give similar results.
In `attention/` directory,

1. To save attention maps, run
 
	```
	python save_graph_info.py --model [model_name] --exp_name exp0
	``` 
	to run the code with `exp_0.jsonl`. For any other set of codes, pass the filename using `--code_file` Replace `model_name` with `codebert`, `graphcodebert`, `unixcoder`,
	`plbart` or `codet5`. This will save the attention maps in `graph_info/exp_0/[model_name]` directory. For smaller model, it requires about 18GB space; larger models require more space.
	
	Note: For CodeRL, the weights need to be downloaded. The link to download the weights is available in CodeRL repo.

2. For exact graph comparision with AST, run 
	
	```
	python graph_comp.py --graph_loc graph_info/exp0/[model_name] --save_dir graph_comparision --exp_name exp_0 --all_layers
	``` 	
	If `model_name` is plbart, also pass `--num_layers 6`. The comparision results are stores in `graph_comparision/ast/exp0/`.

3. For exact graph comparision with DFG, run
	
	```
	python dfg_comp.py --graph_loc graph_info/exp0/[model_name] --save_dir graph_comparision --exp_name exp_0 --all_layers
	``` 
	If `model_name` is plbart, also pass `--num_layers 6`. If the code split used is not `exp_0.jsonl`, pass the one that is used with `--code_file`.
	The comparision results are stores in `graph_comparision/dfg/exp_0`.

4. For similarity analysis with GED, run

	```
	python similarity.py --graphs_dir graph_info/exp0/[model_name] --save_dir graph_comparision --exp_name exp_0 --all_layers
	``` 
	If `model_name` is plbart, also pass `--num_layers 6`. If the code split used is not `exp_0.jsonl`, pass the one that is used with `--code_file`. The results are stores in 
	`graph_comparision/similarity/exp_0`.

	Calculation of Graph Edit Distance can take a lot of time, ~10 hours for each layer of one model.

###  Probing on Hidden Representation
1. First save the hidden representation for all models by running (in `attention/` directory),

	```
	python save_word_embedding.py --model [model_name] --exp_name exp_0
	```
	This will save the hidden representation in the directory `structural_probe/exp_0`. By default, the code uses exp_0.jsonl split. For any other code split, pass it using `--code_file`.

2. Create the dataset for DirectProbe. You need to generate 3 things - directories to save the dataset, config files and finally, the dataset.
	The `DirectProbe/results/` already contains all the generated results. Move these results to a different location before proceeding. 
	
	Run the following in `attention/` directory:
	
	```
	python create_dp_dataset.py --task [task_name] --create_data_dirs
	python create_dp_dataset.py --task [task_name] --create_config_files
	python create_dp_dataset.py --task [task_name] --save_dataset
    ```	
	`task_name` can be `distance`, `siblings` or `dfg`.
	
	Similarly, run `create_dp_id_dataset.py` with `task_name` as `distance_id` or `siblings_id` for dataset with only identifiers as the second token.

	Note: The two files, `create_dp_dataset.py` and `create_dp_id_dataset.py`, has model details in the files. The file has entries for CodeGen and encoder and decoder of CodeT5+2B. To generate the dataset for other models, simply add
	them and their details to the respective lists. The dataset can also be created for one model at a time, but that would result in different data points in the datasets of different models. 

3. In `DirectProbe/` directory and run

	```
	python main.py --config_file [config_file]
	```
	The config_file has the structure `config_files/task_name/config_[model_layer.ini]`. For example, for layer 12 of CodeBERT for tree distance task, 
	it is `config_files/distance/config_codebert_12.ini`.

	Note that, gurobi creates multiple processes and the code takes a lot of time to run. The time taken depends on number of cores available. We ran our experiments on a processor with 32 cores.
	
	The results generated here might vary slightly from those reported in the paper since the data points are sampled randomly.
