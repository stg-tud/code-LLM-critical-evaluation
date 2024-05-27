import argparse
import os
import numpy as np
from tqdm import tqdm
import json
from tree_sitter import Language, Parser
import copy
import pickle

from utils import load_codesearchnet
from get_attention import *

def find_list(main_list, sub_list, sub_list_index = 0):
    begin_item = sub_list[0]
    if begin_item not in main_list or len(main_list) < len(sub_list):
        return None
    else:
        index = main_list.index(begin_item)
        if index + len(sub_list) > len(main_list):
            return None
        else:
            main_list_part = main_list[index : index+len(sub_list)]
            if sub_list != main_list_part:
                sub_list_index += index+1
                return find_list(main_list[index+1:], sub_list, sub_list_index)
            else:
                sub_list_index += index
                return sub_list_index


def traverse_node(node, collected_tokens, byte_code):
    exclude_token = ['"""', "'''", "\\'"]
    for child in node.children:
        if child.child_count == 0:
            token = byte_code[child.start_byte : child.end_byte].decode('utf-8')
            
            if token not in exclude_token and child.type != 'comment':
                token_info = {}
                token_info['id'] = child.id 
                token_info['token'] = token
                token_info['type'] = child.type
                
                token_info['start_byte'] = child.start_byte
                token_info['end_byte'] = child.end_byte

                motif_nodes = []
                for other_child in node.children:
                    if other_child.id != child.id:
                        if other_child.child_count == 0:
                            motif_nodes.append(other_child.id)
                        elif other_child.type != 'block':
                            for one_level_down_child in other_child.children:
                                if one_level_down_child.child_count == 0:
                                    motif_nodes.append(one_level_down_child.id)

                token_info['motif_nodes'] = motif_nodes
                collected_tokens.append(token_info)
        else:
            traverse_node(child, collected_tokens, byte_code)
            

def get_ast_tokens_and_prog_graphs(collected_tokens, model_tokens, tokens, byte_code, window):
    """
    collected_tokens : AST info
    model_tokens : from dataset
    tokens: from model
    window : (start of line token in collected, end of line in collected)
    """
    assert len(model_tokens) == len(tokens), 'This really should not have happened'
    for i in range(len(model_tokens)):
        assert model_tokens[i].replace(" ",'') == tokens[i].replace(" ",''), f'{model_tokens[i]} : {tokens[i]}'
    
    ast_tokens = []
    tokens_idx = 0
    to_skip = []
    index_in_code_tokens = []
    
    for i, token_info in enumerate(collected_tokens):
        if i in range(window[0], window[1]):
            index_in_code_tokens.append(len(ast_tokens))
        
        new_token = copy.deepcopy(token_info)
        if i in to_skip:
            continue
        
        #edge case - for some reason a code has ';' at the end.
        # this is not there in the code token in dataset but available in parse tree
        if token_info['token'] == ';':
            continue
        
        if token_info['token'] != model_tokens[tokens_idx]:
            if token_info['token'] not in model_tokens[tokens_idx]:
                continue
            
                
            #edge case: There is a bug in tree-sitter. So it parses r'text' to r' and '
            #but r'''text''' to r'''
            #the conditional handles this case
            if new_token['token'] == "r'''":
                next_info = collected_tokens[i+1]
                new_token['token'] = new_token['token'] + byte_code[new_token['end_byte'] : next_info['start_byte']].decode('utf-8')
            
            j = 0
            while new_token['token'] != model_tokens[tokens_idx]:
                j+=1
                next_info = collected_tokens[i+j]

                new_token['token'] = new_token['token'] + byte_code[new_token['end_byte'] : next_info['start_byte']].decode('utf-8') + next_info['token']
                new_token['end_byte'] = next_info['end_byte']
                
                new_token['motif_nodes'] = new_token['motif_nodes'] + next_info['motif_nodes']
                
                if new_token['id'] in new_token['motif_nodes']:
                    new_token['motif_nodes'].remove(token_info['id'])
                if next_info['id'] in new_token['motif_nodes']:
                    new_token['motif_nodes'].remove(next_info['id'])
                
                to_skip.append(i+j)
                
        ast_tokens.append(new_token)
        tokens_idx+=1
        
    
    is_error= False
    for i, token in enumerate(model_tokens):
        if token != ast_tokens[i]['token']:
            is_error = True
        
    return ast_tokens, index_in_code_tokens, is_error
    
def get_prog_graph_edges(wala_graph_file, code_data, parser):
    code_nodes = []
    with open(wala_graph_file) as f:
        wala_graph = json.load(f)
        
    prog_graph = wala_graph['turtle_analysis']
    
    for node in prog_graph:
        if node is not None and 'sourceLines' in node.keys():
            code_nodes.append(node)
    
    graph_dict = dict()
    if len(code_nodes) > 0:
        code = code_data['code']
        byte_code = bytes(code, 'utf-8')
        code_tree = parser.parse(byte_code)
        collected_code_tokens = []
        traverse_node(code_tree.root_node, collected_code_tokens, byte_code)
        
        code_tokens = []
        for token in collected_code_tokens:
            code_tokens.append(token['token'])
            
        for code_node in code_nodes:
            source_line = ''
            
            # get line tokens
            for line in code_node['sourceLines']:
                source_line += line
            byte_line = bytes(source_line, 'utf-8')
            line_tree = parser.parse(byte_line)
            collected_line_tokens = []
            traverse_node(line_tree.root_node, collected_line_tokens, byte_line)
            if len(collected_line_tokens) == 0:
                continue
            if collected_line_tokens[-1]['type'] == 'block' and collected_line_tokens[-1]['token'] == '':
                del collected_line_tokens[-1]
            line_tokens = []
            for token in collected_line_tokens:
                line_tokens.append(token['token'])
                
            #get_text_tokens
            source_text = code_node['sourceText']
            byte_text = bytes(source_text, 'utf-8')
            text_tree = parser.parse(byte_text)
            collected_text_tokens = []
            traverse_node(text_tree.root_node, collected_text_tokens, byte_text)
            if len(collected_text_tokens) == 0:
                continue
            if collected_text_tokens[-1]['type'] == 'block' and collected_text_tokens[-1]['token'] == '':
                del collected_text_tokens[-1]
            text_tokens = []
            for token in collected_text_tokens:
                text_tokens.append(token['token'])
                
            text_index = find_list(line_tokens, text_tokens)
            span_length = len(text_tokens)
            
            text_tokens_sub = text_tokens[1:]
            while text_index is None and len(text_tokens_sub)>0:
                text_index = find_list(line_tokens, text_tokens_sub)
                span_length = len(text_tokens_sub)
                if len(text_tokens_sub) == 1:
                    text_tokens_sub = []
                else:
                    text_tokens_sub = text_tokens_sub[1:]
                    
            text_tokens_sub = text_tokens[:-1]
            while text_index is None and len(text_tokens_sub)>0:
                text_index = find_list(line_tokens, text_tokens_sub)
                span_length = len(text_tokens_sub)
                if len(text_tokens_sub) == 1:
                    text_tokens_sub = []
                else:
                    text_tokens_sub = text_tokens_sub[:-1]
                
            for line_token in line_tokens:
                if line_token == ' ' or line_token == '\t':
                    line_tokens.remove(line_token)
            
            begin_index = find_list(code_tokens, line_tokens)
            end_index = begin_index + len(line_tokens)
            
            _, index_in_code_token, _ = get_ast_tokens_and_prog_graphs(collected_code_tokens, code_data['code_tokens'],
                                                                      code_data['code_tokens'], byte_code, (begin_index, end_index))
            
            if text_index is not None:
                index_of_source_text = index_in_code_token[text_index : text_index+span_length]
            else:
                index_of_source_text = None
            
            dfg_edges = []
            cfg_edges = []
            if 'flowsTo' in code_node['edges'].keys():
                dfg_edges_dict = code_node['edges']['flowsTo']
                dfg_edges = []
                for edge_type, edge_list in dfg_edges_dict.items():
                    dfg_edges += edge_list
                    
            if 'immediatelyPrecedes' in code_node['edges'].keys():
                cfg_edges = code_node['edges']['immediatelyPrecedes']
                
            graph_dict[code_node['nodeNumber']] = {
                'nodeNumber': code_node['nodeNumber'],
                'index_of_source_text' : index_of_source_text,
                'dfg_edges' : dfg_edges,
                'cfg_edges' : cfg_edges,
                'sourceLines' : source_line,
                'sourceText' : source_text
            }
            
    return graph_dict
    
def tokens_to_graph(ast_info):
    ast_token_id_list = []
    for info in ast_info:
        ast_token_id_list.append(info['id'])
    
    #always ensure this assertion
    #if this assertion fails then the implementation below won't work
    #the assertion assumes each node in tree-sitter parse tree has unique id
    assert len(set(ast_token_id_list)) == len(ast_token_id_list)

    ast_graph = np.zeros(shape = (len(ast_info), len(ast_info)))
    
    for row, info in enumerate(ast_info):
        motif_nodes = info['motif_nodes']
        for node in motif_nodes:
            # this check is necessary since some of the node ids are not
            # present in ast_info due to merging of nodes which retains 
            # node id of only the first node
            if node in ast_token_id_list:
                col = ast_token_id_list.index(node)
                ast_graph[row, col] = 1
    
    return ast_graph
