from dfg.DFG import DFG_python
from dfg.utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from utils import load_codesearchnet, get_max_edges
from graph_comp_utils import *

import argparse
from tree_sitter import Language, Parser
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import json


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
    
    for edges in DFG:
        row = edges[1]
        cols = edges[-1]
        if len(cols) != 0:
            for col in cols:
                count += 1
                dfg_adj[row, col] = 1
                
    assert count == dfg_adj.sum()
    
    return dfg_adj, code_tokens

def save_dfg_stats(code_file, graph_loc, save_dir, layer, exp_name, parser):
    codes = load_codesearchnet(code_file)
    info_files = os.listdir(graph_loc)
    num_codes = len(info_files)
    
    file_0 = info_files[0]
    file_0 = os.path.join(graph_loc, file_0)
    
    with open(file_0, 'rb') as f:
        info = pickle.load(f)
    
    model_graph = info['model_graphs']
    num_layers = model_graph.shape[0]
    num_heads = model_graph.shape[1]
    
    if layer >= num_layers or layer < -1:
        raise f'Wrong layer index provided: {layer}'
    
    thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    f_scores = {}
    recall_values = {}
    precision_values = {}
    
    for i in range(num_heads):
        f_scores[i] = {}
        recall_values[i] = {}
        precision_values[i] = {}
        for threshold in thresholds:
            f_scores[i][threshold] = 0.0
            recall_values[i][threshold] = 0.0
            precision_values[i][threshold] = 0.0
            
    success = 0
    for code in tqdm(codes):
        info_file_name = os.path.join(graph_loc, code['code_file']+'.pkl')
        if os.path.exists(info_file_name):
            with open(info_file_name, 'rb') as f:
                info = pickle.load(f)

            info_tokens = info['code_tokens']
            code_tokens = code['code_tokens']
            assert info_tokens == code_tokens      

            code_string = code['code']
            dfg_graph = None
            try:
                dfg_graph, gcb_ct = get_dfg_adj(code_string, parser)
            except:
                dfg_graph = None
                print('exception')
                pass

            if dfg_graph is not None:
                model_graphs = info['model_graphs']
                layer_graph = model_graphs[layer]

                if layer_graph[0].shape == dfg_graph.shape:
                    success += 1
                    for i, head in enumerate(layer_graph):
                        for threshold in thresholds:
                            thr_head = get_max_edges(head, mode = 'threshold', threshold = threshold)
                            f_scr = f_score(thr_head, dfg_graph)
                            recall_value = recall(thr_head, dfg_graph)
                            precision_value = precision(thr_head, dfg_graph)
                            f_scores[i][threshold] += f_scr
                            recall_values[i][threshold] += recall_value
                            precision_values[i][threshold] += precision_value

                        
    for i in range(num_heads):
        for threshold in thresholds:
            f_scores[i][threshold] /= success
            recall_values[i][threshold] /= success
            precision_values[i][threshold] /= success          
            
    model_name = graph_loc.split('/')[-1]
    if model_name == '':
        model_name = graph_loc.split('/')[-2]
        
    if  not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    output_dir = os.path.join(save_dir, 'dfg')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    if exp_name is not None:
        output_dir = os.path.join(output_dir, exp_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
    data_name = f'{model_name}_layer_{layer}.json'
    data = {
        'fscore' : f_scores,
        'recall' : recall_values,
        'precision': precision_values
    }
    
    with open(os.path.join(output_dir, data_name), 'w') as f:
        json.dump(data,f)
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_file', default = 'exp_data/exp_0.jsonl')
    parser.add_argument('--graph_loc', required = True)
    parser.add_argument('--save_dir', required = True)
    parser.add_argument('--exp_name', required = False)
    parser.add_argument('--layer', default = -1, type=int)
    parser.add_argument('--all_layers', action='store_true')
    parser.add_argument('--num_layers', default=12, type=int)
    args = parser.parse_args()
    
    PY_LANGUAGE = Language('build/my-languages.so', 'python')
    parser = Parser() 
    parser.set_language(PY_LANGUAGE)
    
    if args.all_layers:
        for l in range(args.num_layers):
            print(f'Evaluating layer {l}...')
            save_dfg_stats(args.code_file, args.graph_loc, args.save_dir, l, args.exp_name, parser)
    else:
        save_dfg_stats(args.code_file, args.graph_loc, args.save_dir, args.layer, args.exp_name, parser)        
    
    


