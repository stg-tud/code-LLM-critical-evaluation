import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
    
def load_codesearchnet(file, num_codes = None, modify_csn=False):
    """
    returns an array with num_codes number of code json data randomly sampled from the file
    modify codes such as remove code comments
    """
    with open(file, 'r') as f:
        data = f.readlines()
        
    assert num_codes is None or type(num_codes) == int ,"Input the number of codes to return"
    
    if num_codes is None:
        code_idxs = range(len(data))
    else: 
        code_idxs = random.sample(range(len(data)), num_codes)
    
    codes = []
    for idx in code_idxs:
        codes.append(json.loads(data[idx]))
        
    return codes
    
def merge_tokens_and_attention(tokenized_tokens, code_tokens, attention, start_index, end_index = -1, break_text_tokens = False, remove_cls_and_sep=True, mode='codebert', special_char='Ġ'):
    """ 
    start_index is 1 for codebert, 3 for unixCoder, 0 for PLBART
    special_char = '▁' for PLBART
    """
    
    assert mode in ["<encoder-only>", "<decoder-only>", "<encoder-decoder>", 'codebert', 'plbart']
    
    mask = []
    code_idx = 0
    merged_token = ''
    merged_tokens = []
    
    modified_code_tokens = []
    # The text within the code (comments, print statements etc.) exists as a single token in dataset
    # This creates an issue for comparision with merges tokens as merged ones don't have spaces and no way
    # to determine where to introduce space. Two ways to resolve this:
    # 1) compare after removing all spaces in between the text
    # 2) break the text into tokens at white_spaces.... This can be beneficial in seeing which part of text is
    # attended to more but, in some cases, can reduce the number of code token attention with a limit of 3 tokens 
    for token in code_tokens:
        if not " " in token:
            modified_code_tokens.append(token)
        else:
            if break_text_tokens:
                modified_code_tokens += token.split()
            else:
                modified_code_tokens.append(token.replace(" ", ""))
    
    
    if not remove_cls_and_sep:
        if mode == 'codebert':
            tokenized_tokens = ['<s>']+tokenized_tokens+['</s>']
            code_tokens = ['<s>']+code_tokens+['</s>']
        elif mode == 'plbart':
            tokenized_tokens = tokenized_tokens+['</s>']
            code_tokens = code_tokens+['</s>']
        else:             # mode is one of unixcoder modes
            tokenized_tokens = ['<s>',mode,'</s>']+tokenized_tokens+['</s>']
            code_tokens = ['<s>',mode,'</s>']+code_tokens+['</s>']
    
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
     	    

    
    mask = torch.tensor(mask)    #mask basically gives the index of the last subword of a token
    
    num_layers = len(attention)
    
    if remove_cls_and_sep:
        if end_index == -1: 
            all_att = torch.cat([attention[n][:, :, start_index:-1, start_index:-1] for n in range(num_layers)], dim = 0)
        else:
            all_att = torch.cat([attention[n][:, :, start_index:, start_index:] for n in range(num_layers)], dim = 0)
    else:
        all_att = torch.cat([attention[n][:, :, :, :] for n in range(num_layers)], dim = 0)
    
    seq_len = len(modified_code_tokens)
    all_att = torch.stack([all_att[:, :, :, mask==i].mean(dim = 3) for i in range(seq_len)], dim = 3)
    all_att = torch.stack([all_att[:,:, mask==i].mean(dim=2) for i in range(seq_len)], dim=2)
    
    return all_att, merged_tokens
    
def draw_map(att_map, tokens, figsize = (3,3), save_loc=None, labelsize=20):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(att_map)
    ax.set_xticks(np.arange(len(tokens)), labels=tokens)
    ax.set_yticks(np.arange(len(tokens)), labels=tokens)
    plt.tick_params(labelsize = labelsize)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    #for i in range(len(tokens)):
     #   for j in range(len(tokens)):
      #      text = ax.text(j, i, att_map[i, j],
       #                ha="center", va="center", color="w")
    
    fig.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc, dpi=300)
    else:
        plt.show()
    
#make all nodes unique
#This is required....otherwise networkx merges the same named nodes.   
def make_nodes_unique(tokens):
    for i, token in enumerate(tokens):
        num_occ = 0
        for j in range(i+1, len(tokens)):
            if token == tokens[j]:
                num_occ += 1
                tokens[j] = tokens[j] + '_' +str(num_occ) 
    return tokens
    
def get_max_edges(head, mode = 'both', **kwargs):
    assert mode in ['max', 'threshold', 'both'], f"{mode} is not a valid mode of masking."
    head = copy.deepcopy(head)
    
    if mode == 'max':
        n_max = kwargs['n_max']
        mask_0 = np.argpartition(head, -n_max, axis = 1) < head.shape[1] - n_max
        #mask_1 = np.argpartition(head, -n_max, axis = 1) >= head.shape[1] - n_max
        head[mask_0] = 0 
        
    elif mode == 'threshold':
       threshold = kwargs['threshold']
       head = head * (head > threshold)
    
    else:
        n_max = kwargs['n_max']
        threshold = kwargs['threshold']
        mask_0 = np.argpartition(head, -n_max, axis = 1) < head.shape[1] - n_max
        head[mask_0] = 0
        head = head * (head > threshold)
        
    return head
    
 
    

