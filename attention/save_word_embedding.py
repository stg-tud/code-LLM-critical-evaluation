import argparse
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import copy

from transformers import RobertaModel, RobertaTokenizer, T5ForConditionalGeneration, RobertaForMaskedLM
from transformers import PLBartTokenizer, PLBartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from unixcoder import UniXcoder

from tree_sitter import Language, Parser

from utils import load_codesearchnet

def get_model_and_tokenizer(model, model_version = None):
    valid_models = ['codebert', 'graphcodebert', 'unixcoder', 'codet5', 'plbart', 'coderl', 'codet5p_2b', 'codet5p_220', 'codet5p_770', 'codet5_musu', 'codet5_lntp', 'codegen', 'codet5p_2b_dec']
    assert model in valid_models, f'Wrong model name : {model}'

    if model == 'codebert':
        model_version = 'microsoft/codebert-base'
        model = RobertaModel.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char

    if model == 'graphcodebert':
        model_version = 'microsoft/graphcodebert-base'
        model = RobertaForMaskedLM.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char 


    if model == 'plbart':
        model_version = 'uclanlp/plbart-base'
        tokenizer = PLBartTokenizer.from_pretrained(model_version)
        model = PLBartForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        special_char = '▁'
        return model, tokenizer, special_char

    if model == 'codet5':
        model_version = 'Salesforce/codet5-base'
        model = T5ForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char

    if model == 'unixcoder':
        model_version = 'microsoft/unixcoder-base'
        model = UniXcoder(model_version)
        special_char='Ġ'
        return model, None, special_char

    if model == 'coderl':
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-large-ntp-py')
        model_version = 'coderl_weights/coderl/'
        model = T5ForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        special_char='Ġ'
        return model, tokenizer, special_char

    if model == 'codet5p_2b':
        model_version = "Salesforce/codet5p-2b"
        tokenizer = AutoTokenizer.from_pretrained(model_version)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_version,
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True, output_hidden_states=True)

        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
        # model.config.encoder.output_attentions = True
        model.config.encoder.output_hidden_states = True
        special_char='Ġ'
        return model, tokenizer, special_char
        
        
    if model == 'codet5p_2b_dec':
        model_version = "Salesforce/codet5p-2b"
        tokenizer = AutoTokenizer.from_pretrained(model_version)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_version,
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True, output_hidden_states=True)
        
        model.config.add_cross_attention=False
        model.config.output_attentions = True
        model.config.output_hidden_states = True
        model.config.decoder.add_cross_attention = False
        model.config.decoder.output_attentions = True
        model.config.decoder.output_hidden_states = True
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
        
        special_char='Ġ'
        return model, tokenizer, special_char
    
    
    if model == 'codet5p_220':
        model_version = 'Salesforce/codet5p-220m'
        model = T5ForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char
        
    if model == 'codet5p_770':
        model_version = 'Salesforce/codet5p-770m'
        model = T5ForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char
        
    if model == 'codet5_musu':
        model_version = 'Salesforce/codet5-base-multi-sum'
        model = T5ForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char
        
    if model == 'codet5_lntp':
        model_version = 'Salesforce/codet5-large-ntp-py'
        model = T5ForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char
        
    if model == 'codegen':
        model_version = 'Salesforce/codegen2-3_7B'
        model = AutoModelForCausalLM.from_pretrained(model_version, output_hidden_states = True, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char
        

def merge_hidden_repr(hidden_states, tokenized_tokens, code_tokens, start_index = 1, end_index=-1, special_char = 'Ġ'):
    mask = []
    code_idx = 0
    merged_token = ''
    merged_tokens = []

    modified_code_tokens = []

    for token in code_tokens:
        if not " " in token:
            modified_code_tokens.append(token)
        else:
            modified_code_tokens.append(token.replace(" ", ""))

    for i in range(len(tokenized_tokens)):
        token = tokenized_tokens[i]
        while len(token) > 0 and token[0] == special_char:
            token = token[1:]
        merged_token += token
        mask.append(code_idx)

        if merged_token == modified_code_tokens[code_idx]:
            code_idx += 1
            merged_tokens.append(merged_token)
            merged_token = ''

    if code_idx != len(modified_code_tokens):
        print("tokens: ", tokenized_tokens, "\n")
        print("raw_tokens:", code_tokens, "\n")
        print("merged: ", merged_tokens, "\n")
        raise Exception(f'Tokens mismatch: \n {code_idx}, {len(modified_code_tokens)} ')

    mask = torch.tensor(mask)
    num_layers = len(hidden_states)
    
    if end_index == -1:
        all_hidden_states = torch.cat([hidden_states[n][:, start_index:-1, :] for n in range(num_layers)], dim=0)
        seq_len = len(modified_code_tokens)
        all_hidden_states = torch.stack([all_hidden_states[:, mask == m, :].mean(dim = 1) for m in range(seq_len)], dim = 1)
    else:
        all_hidden_states = torch.cat([hidden_states[n][:, start_index:, :] for n in range(num_layers)], dim=0)
        seq_len = len(modified_code_tokens)
        all_hidden_states = torch.stack([all_hidden_states[:, mask == m, :].mean(dim = 1) for m in range(seq_len)], dim = 1)
    return all_hidden_states.cpu().detach().numpy()
        

def get_lca_info(node, walk_path, curr_depth, depth, is_code_token, code_token_info, byte_code):
    if node.child_count == 0:
        token = byte_code[node.start_byte : node.end_byte].decode('utf-8')
        walk_path.append(token)
        depth.append(curr_depth)
        is_code_token.append(True)

        token_info = {}
        token_info['type'] = node.type
        token_info['token'] = token
        token_info['start_byte'] = node.start_byte
        token_info['end_byte'] = node.end_byte
        code_token_info.append(token_info)

    else:
        walk_path.append(node.type)
        depth.append(curr_depth)
        is_code_token.append(False)

    children = node.children
    if node.type == 'string':
        text = ''
        for child in children:
            text += byte_code[child.start_byte:child.end_byte].decode('utf-8')

    for child in children:
        curr_depth += 1
        get_lca_info(child, walk_path, curr_depth, depth, is_code_token, code_token_info, byte_code)
        curr_depth -= 1

        if node.child_count == 0:
            token = byte_code[node.start_byte:node.end_byte].decode('utf-8')
            walk_path.append(token)
            depth.append(curr_depth)
            is_code_token.append(True)
            print('this probebly is unreachable')

        else:
            walk_path.append(node.type)
            depth.append(curr_depth)
            is_code_token.append(False)


# get tree distances
def find_tree_distance(u, v, walk_path, depth):
    if u > v:
        a = v
        b = u
    else:
        a = u
        b = v

    slice = depth[a:b+1]
    min_depth = min(slice)
    index = slice.index(min_depth) + a

    tree_dist = depth[u] + depth[v] - 2*min_depth
    lca = walk_path[index]
    return lca, tree_dist


def rows_to_delete_or_merge(code_token_info, code_tokens, byte_code):
    rows_to_delete = []
    rows_to_merge = []
    exclude_token = ['"""', "'''", "\\'"]
    to_skip = []
    code_token_idx = 0

    for i, token_info in enumerate(code_token_info):
        if i in to_skip:
            continue
        token = token_info['token']
        new_token = copy.deepcopy(token_info)

        if token == ';':
            rows_to_delete.append(i)
            continue
        if token in exclude_token or token_info['type'] == 'comment':
            rows_to_delete.append(i)
            continue
        if token != code_tokens[code_token_idx]:
            if token not in code_tokens[code_token_idx]:
                rows_to_delete.append(i)
                continue

        if token == code_tokens[code_token_idx]:
            code_token_idx += 1
            continue

        merged_rows = []
        merged_rows.append(i)

        if new_token['token'] == "r'''":
            next_info = code_token_info[i+1]
            new_token['token'] += byte_code[new_token['end_byte']:next_info['start_byte']].decode('utf-8') + "'''"
            merged_rows.append(i+1)

        j = 0
        while new_token['token'] != code_tokens[code_token_idx]:
            j += 1
            next_info = code_token_info[i+j]

            new_token['token'] += byte_code[new_token['end_byte']:next_info['start_byte']].decode('utf-8') + next_info['token']

            new_token['end_byte'] = next_info['end_byte']
            merged_rows.append(i+j)
            to_skip.append(i+j)

        rows_to_merge.append(merged_rows)
        code_token_idx += 1

    return rows_to_delete, rows_to_merge


#save required informtation
def save_word_embeddings(args, save_dir):
    model, tokenizer, special_char = get_model_and_tokenizer(args.model)
    model = model.to(args.device)

    codes = load_codesearchnet(args.code_file)
    embeddings_dict = {}

    code_num = 0
    for code in tqdm(codes):
        code_tokens = code['code_tokens']
        code_file_name = code['code_file']
        
        if args.model == 'unixcoder':
            tokenized_tokens, token_ids = model.tokenize([' '.join(code_tokens)])
            source_ids = torch.tensor(token_ids).to(args.device)
            outputs, _, _ = model(source_ids)
            tokenized_tokens = tokenized_tokens[0]

        elif args.model == 'codegen':    
            tokenized_tokens = tokenizer.tokenize(' '.join(code_tokens))
            tokens = tokenized_tokens 
            token_idx = tokenizer.convert_tokens_to_ids(tokens)
            inputs = torch.tensor(token_idx).unsqueeze(0)
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            
        elif args.model == 'codet5p_2b':    
            tokenized_tokens = tokenizer.tokenize(' '.join(code_tokens))
            tokens = tokenized_tokens 
            token_idx = tokenizer.convert_tokens_to_ids(tokens)
            inputs = torch.tensor(token_idx).unsqueeze(0)
            inputs = inputs.to(args.device)
            label = ' '.join(code['docstring_tokens'])
            labels = tokenizer(label, return_tensors='pt').input_ids.to(args.device)
            outputs = model(input_ids=inputs, labels=labels)
            
        elif args.model == 'codet5p_2b_dec':    
            tokenized_tokens = tokenizer.tokenize(' '.join(code_tokens))
            tokens = tokenized_tokens 
            token_idx = tokenizer.convert_tokens_to_ids(tokens)
            inputs = torch.tensor(token_idx).unsqueeze(0)
            inputs = inputs.to(args.device)
            label = ' '.join(code['code_tokens'])
            labels = tokenizer(label, return_tensors='pt').input_ids.to(args.device)
            outputs = model(input_ids=inputs, labels=labels)
              
        else:
            tokenized_tokens = tokenizer.tokenize(' '.join(code_tokens))
            tokens = [tokenizer.cls_token] + tokenized_tokens + [tokenizer.sep_token]
            token_idx = tokenizer.convert_tokens_to_ids(tokens)
            inputs = torch.tensor(token_idx).unsqueeze(0)
            inputs = inputs.to(args.device)
            

            if args.model in ['codet5', 'codet5_large', 'coderl', 'codet5p_220', 'codet5p_770']:
                label = ' '.join(code['docstring_tokens'])
                labels = tokenizer(label, return_tensors='pt').input_ids.to(args.device)
                outputs = model(input_ids=inputs, labels=labels)
            else:
                outputs = model(inputs)

        if args.model in ['plbart', 'codet5', 'codet5_large', 'codet5p_220', 'codet5p_770']:
            all_hidden_states = merge_hidden_repr(outputs.encoder_hidden_states, tokenized_tokens, code_tokens, special_char=special_char)
        elif args.model == 'coderl':
            hidden_states = outputs[-1]
            all_hidden_states = merge_hidden_repr(hidden_states, tokenized_tokens, code_tokens, special_char=special_char)
        elif args.model in ['codebert', 'graphcodebert', 'unixcoder']:
            start_index = 3 if args.model == 'unixcoder' else 1
            all_hidden_states = merge_hidden_repr(outputs.hidden_states, tokenized_tokens, code_tokens, start_index=start_index,  special_char=special_char)
        
        elif args.model in ['codet5p_2b'] :
            start_index = 0
            try:
                all_hidden_states = merge_hidden_repr(outputs.encoder_hidden_states, tokenized_tokens, code_tokens, start_index=start_index, end_index=None, special_char=special_char)
            except:
                all_hidden_states = None
         
        elif args.model=='codet5p_2b_dec':
            start_index = 0
            try:
                all_hidden_states = merge_hidden_repr(outputs.decoder_hidden_states, tokenized_tokens, code_tokens, start_index=start_index, end_index=None, special_char=special_char)
            except:
                all_hidden_states = None
        elif args.model == 'codegen':
            start_index = 0
            try:
                all_hidden_states = merge_hidden_repr(outputs.hidden_states, tokenized_tokens, code_tokens, start_index=start_index, end_index=None, special_char=special_char)
            except Exception as e:
                print(e)
                all_hidden_states = None
            
        
            
        if all_hidden_states is not None:
            walk_path = []
            curr_depth = 0
            depth = []
            is_code_token = []
            code_token_info = []

            code_string = code['code']
            byte_code = bytes(code_string, 'utf-8')
            tree = parser.parse(byte_code)

            get_lca_info(tree.root_node, walk_path, curr_depth, depth, is_code_token, code_token_info, byte_code)
            n = sum(is_code_token)
            tree_dist_matrix = np.ndarray((n, n))
            lca_matrix = np.ndarray((n, n), dtype='str')

            ast_terminal_tokens = []
            for u, is_terminal in enumerate(is_code_token):
                if is_terminal:
                    ast_terminal_tokens.append(walk_path[u])
                    row_idx = sum(is_code_token[:u])
                    dist_matrix_row = []
                    lca_matrix_row = []

                    for v, is_terminal_again in enumerate(is_code_token):
                        if is_terminal_again:
                            row_node = walk_path[u]
                            col_node = walk_path[v]

                            lca, tree_dist = find_tree_distance(u, v, walk_path, depth)
                            dist_matrix_row.append(tree_dist)
                            lca_matrix_row.append(lca)

                    assert len(dist_matrix_row) == n
                    assert len(lca_matrix_row) == n
                    tree_dist_matrix[row_idx] = dist_matrix_row
                    lca_matrix[row_idx] = lca_matrix_row
            try:
                rows_to_delete, rows_to_merge = rows_to_delete_or_merge(code_token_info, code_tokens, byte_code)

                for rows in rows_to_merge:
                    merging_rows = tree_dist_matrix[rows]
                    minm = merging_rows.min(axis = 0)
                    tree_dist_matrix[rows[0]] = minm

                    for other_rows in rows[1:]:
                        rows_to_delete.append(other_rows)

                tree_dist_matrix = np.delete(tree_dist_matrix, rows_to_delete, 0)
                tree_dist_matrix = np.delete(tree_dist_matrix, rows_to_delete, 1)

                code_token_info_new = []
                for i, info in enumerate(code_token_info):
                    if i not in rows_to_delete:
                        code_token_info_new.append(info)

                assert len(code_token_info_new) == len(code_tokens)
                assert tree_dist_matrix.shape[0] == len(code_tokens)
                assert tree_dist_matrix.shape[1] == len(code_tokens)
                assert all_hidden_states.shape[1] == tree_dist_matrix.shape[0]

                embeddings_dict[code_num] = {}
                embeddings_dict[code_num]['hidden_repr'] = all_hidden_states
                embeddings_dict[code_num]['tree_dist'] = tree_dist_matrix
                embeddings_dict[code_num]['code_token_info'] = code_token_info_new
                embeddings_dict[code_num]['code_file'] = code_file_name

                code_num += 1
            except:
                print('There was an issue when trying to get delete and merge rows.')
        #print(len(embeddings_dict.keys()))
        file_name = args.model + '.pkl'

        with open(os.path.join(save_dir, file_name), 'wb') as f:
            pickle.dump(embeddings_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--code_file', default='exp_data/exp_0.jsonl')
    parser.add_argument('--num_codes', default=None, type=int)
    parser.add_argument('--save_dir', default='structural_probe')
    parser.add_argument('--exp_name', required=False)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if not os.path.exists('build/'):
           Language.build_library(
              'build/my-languages.so',
              ['tree-sitter-python']
           )

    PY_LANGUAGE = Language('build/my-languages.so', 'python')
    parser = Parser()
    parser.set_language(PY_LANGUAGE)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if args.exp_name is not None:
        save_dir = os.path.join(save_dir, args.exp_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    save_word_embeddings(args, save_dir)

