import argparse
import pickle
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from graph_comp_utils import *
from utils import get_max_edges

def save_ast_stats(graph_loc, save_dir, layer, exp_name = None):
    graphs = os.listdir(graph_loc)
    num_codes = len(graphs)
    
    thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    #get the num layer and num head of the model
    graph_0 = graphs[0]
    
    with open(os.path.join(graph_loc, graph_0), 'rb') as file:
        graph_info = pickle.load(file)
        
    model_graphs = graph_info['model_graphs']
    num_layers = model_graphs.shape[0]
    num_heads = model_graphs.shape[1]
    print(num_layers, num_heads)
    
    if layer >= num_layers or layer < -1:
        raise f'Wrong layer index provided: {layer}'
    
    
    f_scores = {}
    recall_vals = {}
    precision_vals = {}
    ious = {}
    
    for i in range(num_heads):
        f_scores[i]  = {}
        recall_vals[i] = {}
        precision_vals[i] = {}
        ious[i] = {}
        for threshold in thresholds:
            f_scores[i][threshold] = 0.0
            recall_vals[i][threshold] = 0.0
            precision_vals[i][threshold] = 0.0
            ious[i][threshold] = 0.0
            
    
    for graph in tqdm(graphs):
        with open(os.path.join(graph_loc, graph), 'rb') as file:
            graph_info = pickle.load(file)
            
        model_graphs = graph_info['model_graphs']
        ast_graph = graph_info['ast_graph']
        
        last_layer = model_graphs[layer]
        
        for i, head in enumerate(last_layer):
            for threshold in thresholds:
                thr_head = get_max_edges(head, mode = 'threshold', threshold = threshold)
                f_scr = f_score(thr_head, ast_graph)
                recall_val = recall(thr_head, ast_graph)
                precision_val = precision(thr_head, ast_graph)
                iou = IoU(thr_head, ast_graph)
                
                f_scores[i][threshold] += f_scr 
                recall_vals[i][threshold] += recall_val
                precision_vals[i][threshold] += precision_val
                ious[i][threshold] += iou
                
    for i in range(num_heads):
        for threshold in thresholds:
            f_scores[i][threshold] /= num_codes
            recall_vals[i][threshold] /= num_codes
            precision_vals[i][threshold] /= num_codes
            ious[i][threshold] /= num_codes
            
            
    #save the data
    model_name = graph_loc.split('/')[-1]
    if model_name == '':
        model_name = graph_loc.split('/')[-2]
        
    
    if  not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    output_dir = os.path.join(save_dir, 'ast')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    if exp_name is not None:
        output_dir = os.path.join(output_dir, exp_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    fig_name = f'{model_name}_layer_{layer}.png'
    data_name = f'{model_name}_layer_{layer}.json'
    
    data = {
    	'fscore' : f_scores,
    	'recall' : recall_vals,
    	'precision': precision_vals,
    	'iou' : ious
    }
    
    with open(os.path.join(output_dir, data_name), 'w') as f:
        json.dump(data, f) 
    
    
    fig, axs = plt.subplots(2,2, sharex = True, sharey = True, figsize=(10,10))
    cmap = plt.get_cmap('jet_r')
    N = num_heads
    for i in range(num_heads):
        color = cmap(float(i/N))
        #### F_score
        f_score_head = f_scores[i]
        keys = list(f_score_head.keys())
        values = list(f_score_head.values())
        
        for j, value in enumerate(values):
            assert f_score_head[keys[j]] == value
        
        axs[0,0].plot(keys, values, marker = 'x', c = color, markersize = 4, label = f'head {i}')
        #axs[0,0].legend() 
        axs[0,0].set_title(f'Plot of F_score for {model_name}')
        axs[0,0].set_xlabel('Threshold')
        axs[0,0].set_ylabel('F_score')
        
        ### Recall
        recall_head = recall_vals[i]
        keys = list(recall_head.keys())
        values = list(recall_head.values())
        
        for j, value in enumerate(values):
            assert recall_head[keys[j]] == value
        
        axs[1,0].plot(keys, values, marker = 'x', c = color, markersize = 4, label = f'head {i}')
        #axs[1,0].legend() 
        axs[1,0].set_title(f'Plot of Recall for {model_name}')
        axs[1,0].set_xlabel('Threshold')
        axs[1,0].set_ylabel('Recall')
        
        ###Precision
        precision_head = precision_vals[i]
        keys = list(precision_head.keys())
        values = list(precision_head.values())
        
        for j, value in enumerate(values):
            assert precision_head[keys[j]] == value
        
        axs[1,1].plot(keys, values, marker = 'x', c = color, markersize = 4, label = f'head {i}')
        #axs[1,1].legend() 
        axs[1,1].set_title(f'Plot of Precision for {model_name}')
        axs[1,1].set_xlabel('Threshold')
        axs[1,1].set_ylabel('Precision')
        
        
        
        ### IoU
        iou_head = ious[i]
        keys = list(iou_head.keys())
        values = list(iou_head.values())
        
        for j, value in enumerate(values):
            assert iou_head[keys[j]] == value
        
        axs[0,1].plot(keys, values, marker = 'x', c = color, markersize = 4, label = f'head {i}')
        #xs[0,1].legend() 
        axs[0,1].set_title(f'Plot of IoU for {model_name}')
        axs[0,1].set_xlabel('Threshold')
        axs[0,1].set_ylabel('IoU')
        
    #handles_and_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #handles, labels = [sum(handle_legend, []) for handle_legend in zip(*handles_and_labels)]
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.savefig(os.path.join(output_dir, fig_name))
    
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_loc', required = True)
    parser.add_argument('--save_dir', required = True)
    parser.add_argument('--exp_name', required = False)
    parser.add_argument('--layer', default = -1, type=int)
    parser.add_argument('--all_layers', action='store_true')
    parser.add_argument('--num_layers', default=12, type=int)
    args = parser.parse_args()
    
    if args.all_layers:
        for l in range(args.num_layers):
           print(f'Evaluating layer {l}...')
           save_ast_stats(args.graph_loc, args.save_dir, l, exp_name = args.exp_name) 
    else:
        save_ast_stats(args.graph_loc, args.save_dir, args.layer, exp_name = args.exp_name)
    
    
    
    
    
    
    
    
    
    
