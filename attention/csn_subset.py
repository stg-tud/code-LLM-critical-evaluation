import json
from tqdm import tqdm

from transformers import RobertaTokenizer, PLBartTokenizer

from utils import load_codesearchnet
from unixcoder import UniXcoder

# the input token size during pretrianing of codebert, graphcodebert, and unixcoder is limited
#so only select codes whose token size is within the limit.
def check_csn_token_size(file = 'codesearchnet/python/final/jsonl/test/python_test_0.jsonl'):
    cb_tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    gcb_tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
    plb_tokenizer = PLBartTokenizer.from_pretrained('uclanlp/plbart-base')
    
    model_name = 'microsoft/unixcoder-base'
    uxc_model = UniXcoder(model_name)
    
    codes = load_codesearchnet(file)
    selected_codes = []
    
    with open("codesearchnet/csn_subset.jsonl", 'w') as f:
        for code in tqdm(codes):
            include = True
            code_tokens = code['code_tokens']
            tokenized = cb_tokenizer.tokenize(" ".join(code_tokens))
            include = check_merge_token_size(tokenized, code_tokens)
            if len(tokenized) > 500:
                include = False
	          
            if include is True: 
                tokenized = gcb_tokenizer.tokenize(" ".join(code_tokens))
                include = check_merge_token_size(tokenized, code_tokens)
                
                if len(tokenized) > 500:
                    include = False
	     
            if include is True:
                tokenized ,_ = uxc_model.tokenize([" ".join(code_tokens)]) 
                include = check_merge_token_size(tokenized[0], code_tokens)
                if len(tokenized[0]) > 500:
                    include = False 
                        
            if include is True:
                tokenized = plb_tokenizer.tokenize(" ".join(code_tokens))
                include = check_merge_token_size(tokenized, code_tokens, special_char = '▁')
                          
            if include:
                f.write(json.dumps(code) + "\n")
                
	        
	        
def check_merge_token_size(tokenized_tokens, code_tokens, break_text_tokens=True, special_char='Ġ'):
    code_idx = 0
    merged_token = ''
    
    
    modified_code_tokens = []
    
    for token in code_tokens:
        modified_code_tokens.append(token.replace(" ", ""))
            
    for i in range(len(tokenized_tokens)):
        token = tokenized_tokens[i]
        while len(token) > 0 and token[0] == special_char:
            token = token[1:]
        merged_token += token
        
        if merged_token == modified_code_tokens[code_idx]:
            code_idx += 1
            merged_token = ''
    
    include = code_idx == len(modified_code_tokens)     
    return include
    

if __name__ == "__main__":
    selected = check_csn_token_size()
        
        
  
