from dfg.DFG import DFG_python
from dfg.utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from utils import load_codesearchnet

import pickle
import os
import argparse
import numpy as np
import random
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
from tree_sitter import Language, Parser
from math import ceil



#### Distance dataset
def get_entries_dist(emb_file, selected_token_types):
    with open(emb_file, 'rb') as f:
        embeddings_info = pickle.load(f)

    idxs = list(embeddings_info.keys())
    selected_idxs = sorted(random.sample(idxs, 160))

    selected_entries = []
    for idx in selected_idxs:
        emb_info = embeddings_info[idx]
        tree_dist = emb_info['tree_dist']
        token_info = emb_info['code_token_info']

        sel_rows, sel_cols = np.where(tree_dist<=6)
        for row, col in zip(sel_rows, sel_cols):
            if row == col:
                continue
            if token_info[row]['type'] not in selected_token_types:
                continue
            dist = tree_dist[row, col]
            to_store = [idx, (row, col), dist]
            selected_entries.append(to_store)

    dist_2, dist_3, dist_4, dist_5, dist_6 = [], [], [], [], []

    for entry in selected_entries:
        if entry[-1] == 2:
            dist_2.append(entry)
        elif entry[-1] == 3:
            dist_3.append(entry)
        elif entry[-1] == 4:
            dist_4.append(entry)
        elif entry[-1] == 5:
            dist_5.append(entry)
        elif entry[-1] == 6:
            dist_6.append(entry)

    selected_2 = sorted(random.sample(dist_2, 1300))
    selected_3 = sorted(random.sample(dist_3, 1300))
    selected_4 = sorted(random.sample(dist_4, 1300))
    selected_5 = sorted(random.sample(dist_5, 1300))
    selected_6 = sorted(random.sample(dist_6, 1300))

    all_entries = selected_2 + selected_3 + selected_4 + selected_5 + selected_6
    all_entries.sort(key = lambda x : x[0])

    return all_entries

def get_emb_and_dist(emb_file, entries, layer):
    with open(emb_file, 'rb') as f:
        embeddings_info = pickle.load(f)

    embeddings_diff = []
    token_and_dist = []

    for idx, (row, col), dist in entries:
        emb_info = embeddings_info[idx]
        tree_dist = emb_info['tree_dist']
        embeddings = emb_info['hidden_repr']
        token_info = emb_info['code_token_info']

        assert tree_dist[row, col] == dist

        layer_emb = embeddings[layer]

        emb_a = layer_emb[row]
        emb_b = layer_emb[col]
        diff = emb_a - emb_b

        t_type_a, t_a = token_info[row]['type'], token_info[row]['token']
        t_type_b, t_b = token_info[col]['type'], token_info[col]['token']

        embeddings_diff.append(diff)
        token_and_dist.append((f'{t_type_a, t_type_b}:{t_a}:{t_b}', dist))

    embeddings_diff = np.array(embeddings_diff)

    return embeddings_diff, token_and_dist
    
### Siblings Dataset
def get_entries_sib(emb_file, selected_token_types, graph_info_dir):
    with open(emb_file, 'rb') as f:
        embeddings_info = pickle.load(f)
    
    idxs = list(embeddings_info.keys())
    selected_idxs = sorted(random.sample(idxs, 100))
    
    selected_enteries_0 = []
    selected_enteries_1 = []
    for idx in selected_idxs:
        emb_info = embeddings_info[idx]
        tree_dist = emb_info['tree_dist']
        token_info = emb_info['code_token_info']
        code_file_name = emb_info['code_file']+'.pkl'
        
        with open(os.path.join(graph_info_dir, code_file_name), 'rb') as f:
            graph_info = pickle.load(f)
            ast_graph = graph_info['ast_graph']
        
        assert ast_graph.shape == tree_dist.shape, print(ast_graph.shape, tree_dist.shape)
        
        num_rows = ast_graph.shape[0]
        rows_of_1, cols_of_1 = np.where(ast_graph == 1)
        row_cols = [[] for _ in range(num_rows)]
        
        for idx_of_col, col in enumerate(cols_of_1):
            row = rows_of_1[idx_of_col]
            row_cols[row].append(col)
        
        num_col = num_rows
        all_cols = list(range(num_col))
        
        for row, info in enumerate(token_info):
            if info['type'] not in selected_token_types:
                continue
            cols_1 = row_cols[row]
            num_1s = len(cols_1)
            cols_0 = [x for x in all_cols if x not in cols_1 and x!=row]
            if num_1s < len(cols_0):
                cols_0 = random.sample(cols_0, num_1s)
                
            for col in cols_1:
                to_store = [idx, (row, col), 1]
                selected_enteries_1.append(to_store)

            for col in cols_0:
                to_store = [idx, (row, col), 0]
                selected_enteries_0.append(to_store)
        
    
    selected_0 = sorted(random.sample(selected_enteries_0, 1500))
    selected_1 = sorted(random.sample(selected_enteries_1, 1500))
    
    all_enteries = selected_0 + selected_1
    
    all_enteries.sort(key = lambda x : x[0])
    
    return all_enteries

def get_emb_and_sib(emb_file, enteries, layer):
    with open(emb_file, 'rb') as f:
        embeddings_info = pickle.load(f)
    
    embeddings_concat = []
    sib_or_not = []
    for idx, (row, col), sib in enteries:
        emb_info = embeddings_info[idx]
        embeddings = emb_info['hidden_repr']
        token_info = emb_info['code_token_info']
        
        layer_emb = embeddings[layer]
        emb_a = layer_emb[row]
        emb_b = layer_emb[col]
        concat = np.concatenate((emb_a, emb_b))
        
        t_type_a, t_a = token_info[row]['type'], token_info[row]['token']
        t_type_b, t_b = token_info[col]['type'], token_info[col]['token']
        
        embeddings_concat.append(concat)
        sib_or_not.append((f'{t_type_a, t_type_b}:{t_a}:{t_b}', sib))
    
    embeddings_concat = np.array(embeddings_concat)
    
    return embeddings_concat, sib_or_not
    
## Dataflow dataset
def get_dfg_adj(code_string, parser, lang = 'python'):
    code_string = remove_comments_and_docstrings(code_string, lang)
    tree = parser.parse(bytes(code_string, 'utf-8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)

    code_string = code_string.split('\n')
    code_tokens=[index_to_code_token(x,code_string) for x in tokens_index]

    index_to_code={}
    for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
        index_to_code[index]=(idx,code)

    DFG, _ = DFG_python(root_node, index_to_code, {})
    DFG = sorted(DFG,key=lambda x:x[1])

    n = len(code_tokens)
    dfg_adj = np.zeros((n,n))
    count = 0
    
    for _, row, rel_type, _, cols in DFG:
        if len(cols) != 0:
            rel = -1 if rel_type == 'computedFrom' else 1
            for col in cols:
                count += 1
                dfg_adj[row, col] = rel
    
    r, c = np.where(dfg_adj != 0)
    assert r.shape[0] == count
    
    return dfg_adj, code_tokens
    
    
def get_enteries_df(emb_file, code_file):
    with open(emb_file, 'rb') as f:
        embeddings_info = pickle.load(f)


    codes = load_codesearchnet(code_file)

    code_file_list = []
    for code in codes:
        code_file_list.append(code['code_file'])
    
    idxs = list(embeddings_info.keys())
    selected_idxs = sorted(random.sample(idxs, 130))

    selected_enteries_comes = []
    selected_enteries_comp = []
    selected_enteries_none = []

    for idx in selected_idxs:
        emb_info = embeddings_info[idx]
        token_info = emb_info['code_token_info']
        code_file_name = emb_info['code_file']
        idx_in_list = code_file_list.index(code_file_name)


        code = codes[idx_in_list]
        code_tokens = code['code_tokens']

        assert token_info[1]['token'] == code_tokens[1]

        code_string = code['code']
        dfg_adj = None
        try:
            dfg_adj, _ = get_dfg_adj(code_string, parser)
        except:
            dfg_adj = None
            print('exception')

        gen_length = len(code_tokens)
        if dfg_adj is None:
            continue
        if dfg_adj.shape[0] != gen_length:
            continue

        comes_rows, comes_col = np.where(dfg_adj == 1)
        comp_rows, comp_col = np.where(dfg_adj == -1)

        num_rows = gen_length
        row_cols_comes = [[] for _ in range(num_rows)] 
        row_cols_comp = [[] for _ in range(num_rows)]

        for idx_of_col, col in enumerate(comes_col):
            row = comes_rows[idx_of_col]
            row_cols_comes[row].append(col)

        for idx_of_col, col in enumerate(comp_col):
            row = comp_rows[idx_of_col]
            row_cols_comp[row].append(col)

        num_col = num_rows
        all_cols = list(range(num_col))

        for row, info in enumerate(token_info):
            cols_comes = row_cols_comes[row]
            cols_comp = row_cols_comp[row]
            cols_none = [x for x in all_cols if x not in cols_comes and x not in cols_comp and x!=row]

            num_cols_none = int(ceil(max(len(cols_comes), len(cols_comp))/2))
            if num_cols_none < len(cols_none):
                cols_none = random.sample(cols_none, num_cols_none)


            for col in cols_comes:
                to_store = [idx, (row, col), 1]
                selected_enteries_comes.append(to_store)

            for col in cols_comp:
                to_store = [idx, (row, col), -1]
                selected_enteries_comp.append(to_store)

            for col in cols_none:
                to_store = [idx, (row, col), 0]
                selected_enteries_none.append(to_store)
                
    
    sel_comes = sorted(random.sample(selected_enteries_comes, 1500))
    sel_comp = sorted(random.sample(selected_enteries_comp, 1500))
    sel_none = sorted(random.sample(selected_enteries_none, 1500))
    
    all_enteries = sel_comes + sel_comp + sel_none
    all_enteries.sort(key = lambda x : x[0])
    return all_enteries
    
def get_emb_and_df(emb_file, enteries, layer):
    with open(emb_file, 'rb') as f:
        embeddings_info = pickle.load(f)
    
    embeddings_concat = []
    entities = []
    
    for idx, (row, col), dfg_rel in enteries:
        emb_info = embeddings_info[idx]
        embeddings = emb_info['hidden_repr']
        token_info = emb_info['code_token_info']
        
        layer_emb = embeddings[layer]
        emb_a = layer_emb[row]
        emb_b = layer_emb[col]
        concat = np.concatenate((emb_a, emb_b))
        
        t_type_a, t_a = token_info[row]['type'], token_info[row]['token']
        t_type_b, t_b = token_info[col]['type'], token_info[col]['token']
        
        embeddings_concat.append(concat)
        entities.append((f'{t_type_a, t_type_b}:{t_a}:{t_b}', dfg_rel))
    
    embeddings_concat = np.array(embeddings_concat)
    
    return embeddings_concat, entities

### Save
def save_dataset(train_index, test_index, dp_dir, task, model, embeddings = None, layer = None, entities = None):
    if embeddings is None and entities is None:
        raise Exception('Provide either embeddings or entities')
    if embeddings is not None:
        if layer is None:
            raise Exception('Which layer are the embeddings for?')
        embeddings_train = embeddings[train_index]
        embeddings_test = embeddings[test_index]

        embeddings_dir = os.path.join(dp_dir, 'data', task, model, 'embeddings', 'layers')
        train_file = os.path.join(embeddings_dir, 'train', layer+'.txt')
        test_file = os.path.join(embeddings_dir, 'test', layer+'.txt')

        np.savetxt(train_file, embeddings_train)
        np.savetxt(test_file, embeddings_test)   

    if entities is not None:
        ent_dir = os.path.join(dp_dir, 'data', task, model, 'entities')
        train_file = os.path.join(ent_dir, 'train.txt')
        test_file = os.path.join(ent_dir,'test.txt')

        entities_train = []
        entities_test = []
        for idx in train_index:
            entities_train.append(entities[idx])
        for idx in test_index:
            entities_test.append(entities[idx])

        with open(train_file, 'w') as f:
            for token, dist in entities_train[:-1]:
                f.write(f'{token}\t{dist}\n')
            token, dist = entities_train[-1]
            f.write(f'{token}\t{dist}')

        with open(test_file, 'w') as f:
            for token, dist in entities_test[:-1]:
                f.write(f'{token}\t{dist}\n')
            token, dist = entities_test[-1]
            f.write(f'{token}\t{dist}')

        labels = list(set(sorted(l[1] for l in entities)))
        tags_dir = os.path.join(dp_dir, 'data', task, model, 'labels')
        tags_file = os.path.join(tags_dir, 'tags.txt')
        with open(tags_file, 'w') as f:
            for label in labels[:-1]:
                f.write(f'{label}\n')
            f.write(f'{labels[-1]}')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required = True)
    parser.add_argument('--dp_dir', default='../DirectProbe/')
    parser.add_argument('--create_data_dirs', action = 'store_true')
    parser.add_argument('--create_config_files', action = 'store_true')
    parser.add_argument('--save_dataset', action = 'store_true')

    args = parser.parse_args()

    selected_token_types = ['def', 'for', 'if', 'none', 'else', 'false', 'true' 'or',
                            'and', 'return', 'not', 'elif','with', 'try', 'raise', 
                            'except', 'break', 'while', 'assert', 'print', 'continue', 'class']

    models = ['codegen', 'codet5p_2b', 'codet5p_2b_dec']
    num_layers = [17, 21, 33]     # including the embedding layer as layer 0
    embeddings_file = ['structural_probe/exp_0/codegen.pkl',
    			'structural_probe/exp_0/codet5p_2b.pkl',
    			'structural_probe/exp_0/codet5p_2b_dec.pkl']

    graphs_dir = ['graph_info/exp_0/codegen/',
    		'graph_info/exp_0/codet5p_2b/',
    		'graph_info/exp_0/codet5p_2b_dec/']
    
    # create directories for the task
    if args.create_data_dirs:
        print('creating data directories')
        direc =  os.path.join(args.dp_dir,'data', args.task)
        if not os.path.exists(direc):
            os.mkdir(direc)
        for model in models:
            model_dir = os.path.join(direc, model)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            ent_dir = os.path.join(model_dir, 'entities')
            labels_dir = os.path.join(model_dir, 'labels')
            emb_dir = os.path.join(model_dir, 'embeddings')
            dirs = [ent_dir, labels_dir, emb_dir]

            for dir in dirs:
                if not os.path.exists(dir):
                    os.mkdir(dir)
            emb_dir = os.path.join(emb_dir, 'layers')
            if not os.path.exists(emb_dir):
                os.mkdir(emb_dir)

            dirs = [os.path.join(emb_dir, 'train'), os.path.join(emb_dir, 'test')]
            for dir in dirs:
                if not os.path.exists(dir):
                    os.mkdir(dir)
    
        print('creating results directories')    
        direc =  os.path.join(args.dp_dir,'results', args.task)    
        if not os.path.exists(direc):
            os.mkdir(direc)
        for i, model in enumerate(models):
            model_dir = os.path.join(direc, model)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            for layer in range(num_layers[i]):
                layer_dir = os.path.join(model_dir, str(layer))
                if not os.path.exists(layer_dir):
                    os.mkdir(layer_dir)
                
    
    #save config files
    if args.create_config_files:
        task = args.task
        for i, model in enumerate(models):
            for layer in range(num_layers[i]):
                config = f'''[run]
# The directory of outputs
output_path = results/{task}/{model}/{layer}/

[data]
# the common directory path
common = data/{task}

# the path of label set
label_set_path = ${{common}}/{model}/labels/tags.txt
# the path of training examples
entities_path = ${{common}}/{model}/entities/train.txt
# the path of training embeddings.
embeddings_path = ${{common}}/{model}/embeddings/layers/train/{layer}.txt
# the path of test examples
test_entities_path = ${{common}}/{model}/entities/test.txt
# the path of test emebddings 
test_embeddings_path = ${{common}}/{model}/embeddings/layers/test/{layer}.txt

[clustering]
# If enable gpu
enable_cuda = True

# What is the step size when we check the overlapps
# In most case, this setting does not need to be changed.
rate=0.01

# There are two modes: probing and prediction
# probing mode: Apply the DirectProbe on the given training set and make predictions on the test set.
# prediction mode: Using the given cluters to make predictions using the test set.
# We need a prediction mode because in some case, clustering process is time consuming.
# We want to reuse the cluters.
mode = probing
# mode = prediction

# The path of pre-probed clusters.
probing_cluster_path = results/{task}/{model}/{layer}/'''

                config_file = os.path.join(args.dp_dir, 'config_files', task, f'config_{model}_{layer}.ini')
                with open(config_file, 'w') as f:
                    f.write(config)


    # save the dataset
    if args.save_dataset:
        if args.task == 'distance':
            print(f'Creating dataset for task: {args.task}')
            emb_file = embeddings_file[0]
            all_entries = get_entries_dist(emb_file, selected_token_types)
            split = ShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 0)
            train_index, test_index = next(split.split(all_entries))

            for i, model in enumerate(models):
                print(f'saving data for model {model}')
                emb_file = embeddings_file[i]
                for layer in range(num_layers[i]):
                    embeddings_diff, token_dist = get_emb_and_dist(emb_file, all_entries, layer)
                    save_dataset(train_index, test_index, args.dp_dir, args.task, model, embeddings=embeddings_diff, layer=str(layer))
                save_dataset(train_index, test_index, args.dp_dir, args.task, model, entities=token_dist)


        elif args.task == 'siblings':
            print(f'Creating dataset for task: {args.task}')
            emb_file = embeddings_file[0]
            graph_dir = graphs_dir[0]
            all_entries = get_entries_sib(emb_file, selected_token_types, graph_dir)
            split = ShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 0)
            train_index, test_index = next(split.split(all_entries))
            
            for i, model in enumerate(models):
                print(f'saving data for model {model}')
                emb_file = embeddings_file[i]
                graph_dir = graphs_dir[i]
                for layer in range(num_layers[i]):
                    embeddings_concat, sib_or_not = get_emb_and_sib(emb_file, all_entries, layer)
                    save_dataset(train_index, test_index, args.dp_dir, args.task, model, embeddings=embeddings_concat, layer = str(layer))
                save_dataset(train_index, test_index, args.dp_dir, args.task, model, entities=sib_or_not)
                
                
        elif args.task == 'dfg':
            PY_LANGUAGE = Language('build/my-languages.so', 'python')
            parser = Parser() 
            parser.set_language(PY_LANGUAGE)
            lang = 'python'
            print(f'Creating dataset for task: {args.task}')
            code_file = 'exp_data/exp_0.jsonl'
            emb_file = embeddings_file[0]
            all_enteries = get_enteries_df(emb_file, code_file)
            split = ShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 0)
            train_index, test_index = next(split.split(all_enteries))
            for i, model in enumerate(models):
                print(f'saving data for model {model}')
                emb_file = embeddings_file[i]
                for layer in range(num_layers[i]):
                    embeddings_concat, entities = get_emb_and_df(emb_file, all_enteries, layer)
                    save_dataset(train_index, test_index, args.dp_dir, args.task, model, embeddings=embeddings_concat, layer = str(layer))
                save_dataset(train_index, test_index, args.dp_dir, args.task, model, entities = entities)
            
            
        elif args.task == 'token_type':
            pass
