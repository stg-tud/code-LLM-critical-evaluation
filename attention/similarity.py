from utils import *
from graph_utils import *
from dfg_comp import get_dfg_adj

import os
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
import argparse
import json
from tree_sitter import Language, Parser


def node_match(n1, n2):
    return n1['name'] == n2['name']


def edge_subst_cost(e1, e2):
    return 0


def edge_del_cost(e1):
    return 1


def edge_ins_cost(e1):
    return 1


def ast_graph_wo_identifiers(code_string, code_tokens, tokens, parser):
    byte_code = bytes(code_string, 'utf-8')
    tree = parser.parse(byte_code)
    root_node = tree.root_node

    collected_tokens = []
    traverse_node(root_node, collected_tokens, byte_code)
    ast_info, _, is_error = get_ast_tokens_and_prog_graphs(collected_tokens, code_tokens, tokens, byte_code, (0,0))

    ast_token_id_list = []
    for info in ast_info:
        ast_token_id_list.append(info['id'])

    assert len(set(ast_token_id_list)) == len(ast_token_id_list)

    ast_graph = np.zeros(shape=(len(ast_info), len(ast_info)))

    for row, info in enumerate(ast_info):
        motif_nodes = info['motif_nodes']
        for node in motif_nodes:
            if node in ast_token_id_list:
                col = ast_token_id_list.index(node)
                row_type = info['type']
                col_type = ast_info[col]['type']
                if row_type != 'identifier' and col_type != 'identifier':
                    ast_graph[row, col] = 1

    return ast_graph

def similarity_analysis(args, layer, parser):
    model_name = args.graphs_dir.split('/')[-1]
    if model_name == '':
        model_name = args.graphs_dir.split('/')[-2]

    graphs = os.listdir(args.graphs_dir)
    one_graph = graphs[0]
    with open(os.path.join(args.graphs_dir, one_graph), 'rb') as f:
        graphs_info = pickle.load(f)
        model_graphs = graphs_info['model_graphs']
        num_layers, num_heads, _, _ = model_graphs.shape

    if layer >= num_layers or layer < -1:
        raise f'Wrong layer index provided: {layer}'

    ast_edit_dist_per_head = {}
    dfg_edit_dist_per_head = {}
    com_edit_dist_per_head = {}
    ast_edit_dist_per_head_wo_id = {}

    for k in range(num_heads):
        ast_edit_dist_per_head[k] = 0.0
        dfg_edit_dist_per_head[k] = 0.0
        com_edit_dist_per_head[k] = 0.0
        ast_edit_dist_per_head_wo_id[k] = 0.0

    codes = load_codesearchnet(args.code_file)
    num_codes = len(codes)
    success = 0
    for code in tqdm(codes):
        code_string = code['code']
        info_file_name = os.path.join(args.graphs_dir, code['code_file']+'.pkl')
        if os.path.exists(info_file_name):
            with open(info_file_name, 'rb') as f:
                graph_info = pickle.load(f)

            dfg_adj = None
            try:
                dfg_adj, gcb_ct = get_dfg_adj(code_string, parser)
            except:
                dfg_adj = None
                pass

            if dfg_adj is not None:
                model_adj = graph_info['model_graphs']
                layer_adj = model_adj[layer]
                ast_adj = graph_info['ast_graph']

                code_tokens = graph_info['code_tokens']
                model_tokens = graph_info['model_tokens']
                ast_adj_wo_id = ast_graph_wo_identifiers(code_string, code_tokens, model_tokens, parser)

                tokens = graph_info['code_tokens']
                unique_tokens = make_nodes_unique(tokens)
                num_nodes = len(tokens)
                unique_tokens = {
                    k: {'name': unique_tokens[k]} for k in range(len(unique_tokens))
                }

                if layer_adj[0].shape == dfg_adj.shape:
                    success += 1

                    for head_num, head in enumerate(layer_adj):
                        thr_head = get_max_edges(head, mode = 'threshold', threshold = args.threshold)

                        model_graph = nx.from_numpy_array(thr_head, create_using=nx.DiGraph)
                        ast_graph = nx.from_numpy_array(ast_adj, create_using=nx.DiGraph)
                        dfg_graph = nx.from_numpy_array(dfg_adj, create_using=nx.DiGraph)
                        ast_graph_wo_id = nx.from_numpy_array(ast_adj_wo_id, create_using=nx.DiGraph)

                        nx.set_node_attributes(model_graph, unique_tokens)
                        nx.set_node_attributes(ast_graph, unique_tokens)
                        nx.set_node_attributes(dfg_graph, unique_tokens)
                        nx.set_node_attributes(ast_graph_wo_id, unique_tokens)

                        com_graph = nx.compose(ast_graph, dfg_graph)

                        #AST edit distance cost
                        it = nx.optimize_graph_edit_distance(model_graph, ast_graph,
                                                node_match=node_match, edge_del_cost=edge_del_cost,
                                                edge_ins_cost=edge_ins_cost, edge_subst_cost=edge_subst_cost)
                        ast_cost = it.__next__()

                        # DFG edit distance cost
                        it = nx.optimize_graph_edit_distance(model_graph, dfg_graph, 
                                                node_match=node_match, edge_del_cost=edge_del_cost,
                                                edge_ins_cost=edge_ins_cost, edge_subst_cost=edge_subst_cost)
                        dfg_cost = it.__next__()

                        # edit distance cost for common graph of AST and DFG
                        it = nx.optimize_graph_edit_distance(model_graph, com_graph, 
                                                node_match=node_match, edge_del_cost=edge_del_cost,
                                                edge_ins_cost=edge_ins_cost, edge_subst_cost=edge_subst_cost)
                        com_cost = it.__next__()

                        # AST with identifiers edit distance cost
                        it = nx.optimize_graph_edit_distance(model_graph, ast_graph_wo_id, 
                                                node_match=node_match, edge_del_cost=edge_del_cost,
                                                edge_ins_cost=edge_ins_cost, edge_subst_cost=edge_subst_cost)
                        ast_wo_id_cost = it.__next__()

                        #per node cost
                        ast_cost = ast_cost/num_nodes
                        dfg_cost = dfg_cost/num_nodes
                        com_cost = com_cost/num_nodes
                        ast_wo_id_cost = ast_wo_id_cost/num_nodes

                        ast_edit_dist_per_head[head_num] += ast_cost
                        dfg_edit_dist_per_head[head_num] += dfg_cost
                        com_edit_dist_per_head[head_num] += com_cost
                        ast_edit_dist_per_head_wo_id[head_num] += ast_wo_id_cost

                        model_graph = None
                        ast_graph = None
                        dfg_graph = None
                        com_graph = None
                        ast_graph_wo_id = None

    print(f'Graph similarity calculated for {success} codes out of {num_codes} codes.')
    for k in range(num_heads):
        ast_edit_dist_per_head[k] /= success
        dfg_edit_dist_per_head[k] /= success
        com_edit_dist_per_head[k] /= success
        ast_edit_dist_per_head_wo_id[k] /= success

    cost = {
        'ast' : ast_edit_dist_per_head,
        'dfg' : dfg_edit_dist_per_head,
        'common' : com_edit_dist_per_head,
        'ast_wo_identifiers' : ast_edit_dist_per_head_wo_id
    }

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    save_dir = os.path.join(save_dir, 'similarity')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.exp_name is not None:
        save_dir = os.path.join(save_dir, args.exp_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_file_name = f'layer_{layer}_threshold_{args.threshold}.json'
    with open(os.path.join(save_dir, save_file_name), 'w') as f:
        json.dump(cost, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphs_dir', required=True)
    parser.add_argument('--code_file', default='exp_data/exp_0.jsonl')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--exp_name', required=False)
    parser.add_argument('--layer', default = -1, type=int)
    parser.add_argument('--layers', nargs='+', type=int, required=False)
    parser.add_argument('--all_layers', action='store_true')
    parser.add_argument('--num_layers', default=12, type=int)
    parser.add_argument('--threshold', default=0.05, type=float)
    
    args = parser.parse_args()

    PY_LANGUAGE = Language('build/my-languages.so', 'python')
    parser = Parser()
    parser.set_language(PY_LANGUAGE)

    if args.all_layers:
        for l in range(args.num_layers):
            print(f'Evaluating layer {l}...')
            similarity_analysis(args, l, parser)
    elif args.layers is not None:
        for l in args.layers:
            print(f'Evaluating layer {l}...')
            similarity_analysis(args, l, parser)    
    else:
        similarity_analysis(args, args.layer, parser)


