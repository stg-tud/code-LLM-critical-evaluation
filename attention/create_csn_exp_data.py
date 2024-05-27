import argparse
import os
import json
from tqdm import tqdm

from utils import load_codesearchnet

def create_csn_exp_data(file = 'csn/csn_subset.jsonl', num_codes = 3000):
    codes = load_codesearchnet(file, num_codes = num_codes)
    save_dir = 'exp_data'
    
    file_num = len(os.listdir('exp_data'))
    exp_file = os.path.join(save_dir, f'exp_{file_num}.jsonl')
    
    with open(exp_file, 'w') as f:
        for code in tqdm(codes):
            remove_comments(code)
            merge_kwargs_stars(code)
            f.write(json.dumps(code) + "\n")  
            

def remove_comments(code):
    tokens = code['code_tokens']
    tokens_to_remove = []
    for token in tokens:
        if token[0] == '#' or (len(token) >= 3 and token[0:3] == '"""') or (len(token) >= 3 and token[0:3] == "'''"):
            tokens_to_remove.append(token)
            
    for token in tokens_to_remove:
        tokens.remove(token) 
        
def merge_kwargs_stars(code):
    code_tokens = code['code_tokens']
    
    new_tokens = []
    to_skip = []

    for i, token in enumerate(code_tokens):
        new_token = token
        if i in to_skip:
            continue

        if token == '*':
            next_token = code_tokens[i+1]
            if next_token == '*':
                new_token += next_token
                to_skip.append(i+1)

        new_tokens.append(new_token)
    
    code['code_tokens'] = new_tokens
  
def save_code(file = 'exp_data/exp_0.jsonl', save_dir = 'outputs/exp_0_py_files'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    all_data = load_codesearchnet(file)
    all_code_files = []
    char_begin = 97
    for data in all_data:
        file_name = data['sha']
        while file_name in all_code_files:
            file_name += chr(char_begin)
            char_begin += 1
            if char_begin >= 123:
                char_begin = 97
        all_code_files.append(file_name)
        
        code_string = data['code']
        with open(os.path.join(save_dir, file_name+'.py'), 'w') as f:
            f.write(code_string)
        data['code_file'] = file_name
        
    with open(file, 'w') as f:
        for data in all_data:
            f.write(json.dumps(data) + "\n")                 
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default = 'create_exp')
    parser.add_argument('--split_loc', default='exp_data/exp_0.jsonl')
    parser.add_argument('--code_output_loc', default='outputs/code')
    args = parser.parse_args()
    
    if args.mode == 'create_exp':
        create_csn_exp_data()
    elif args.mode == 'save_code':
        save_code(file=args.split_loc, save_dir=args.code_output_loc)
    else:
        print('mode should be either create_exp or save_code')
