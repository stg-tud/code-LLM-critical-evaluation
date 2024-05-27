import torch
import os
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, T5ForConditionalGeneration, RobertaForMaskedLM
from transformers import PLBartTokenizer, PLBartForConditionalGeneration
from transformers import AutoModel, AutoTokenizer

from transformers import RobertaModel, RobertaTokenizer, T5ForConditionalGeneration, RobertaForMaskedLM
from transformers import PLBartTokenizer, PLBartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from utils import merge_tokens_and_attention
from unixcoder import UniXcoder


def get_attention_codebert(data, device='cuda:0', random = False):
    """
    layer and head index starts from 1.
    """
    
    model_version = 'microsoft/codebert-base'
    model = RobertaModel.from_pretrained(model_version, output_attentions = True)
    
    if random: 
        config = model.config
        model = None
        model = RobertaModel(config) 
    
    model.to(device)
    
    tokenizer = RobertaTokenizer.from_pretrained(model_version)
    
    raw_tokens  = data['code_tokens']
    code_tokens = tokenizer.tokenize(' '.join(raw_tokens))
    tokens  = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    token_idx = tokenizer.convert_tokens_to_ids(tokens)

    inputs  = torch.tensor(token_idx).unsqueeze(0)

    inputs = inputs.to(device)

    outputs = model(inputs)

    attention = outputs.attentions

    
    all_att, tokens = merge_tokens_and_attention(code_tokens, raw_tokens, attention, 1, break_text_tokens = False)
              
    all_att = all_att.detach().cpu().numpy()
        
    return all_att, tokens
        
        
def get_attention_uniXcoder(data, device='cuda:0', random = False):
    model_name = 'microsoft/unixcoder-base'
    model = UniXcoder(model_name, random = random)
    model.to(device)
    
     
    
    raw_tokens = data['code_tokens']
    code_tokens, token_ids = model.tokenize([' '.join(raw_tokens)])
    source_ids = torch.tensor(token_ids).to(device)
    outputs, token_embeddings, sentence_embeddings =  model(source_ids)
    attentions = outputs.attentions
    
    all_att, tokens = merge_tokens_and_attention(code_tokens[0],raw_tokens, attentions, 3, break_text_tokens = False)
    all_att = all_att.detach().cpu().numpy()
    
    return all_att, tokens
    
    
def get_attention_codeT5(data, model_version = 'Salesforce/codet5-base', device='cuda:0', random = False):
    if os.path.exists(model_version):
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-large-ntp-py')
    else:
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
    model = T5ForConditionalGeneration.from_pretrained(model_version, output_attentions = True)
    
    if random:
        config = model.config
        model = None
        model = T5ForConditionalGeneration(config) 
    
    model.to(device)
    
    raw_tokens = data['code_tokens']
    code_tokens = tokenizer.tokenize(' '.join(raw_tokens))
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    token_idx = tokenizer.convert_tokens_to_ids(tokens)
    inputs = torch.tensor(token_idx).unsqueeze(0).to(device)
    
    label = ' '.join(data['docstring_tokens'])
    labels = tokenizer(label, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids = inputs, labels = labels)
    
    if model_version == 'coderl_weights/coderl/':
        encoder_attention = outputs[-1]
    else:
        encoder_attention = outputs.encoder_attentions
    
    all_att, tokens = merge_tokens_and_attention(code_tokens, raw_tokens, encoder_attention, 1, break_text_tokens = False)
    all_att = all_att.detach().cpu().numpy()
    
    return all_att, tokens
    
    
def get_attention_graphcodebert(data, device='cuda:0', random = False):
    """
    layer and head index starts from 1.
    """
    
    model_version = 'microsoft/graphcodebert-base'
    model = RobertaForMaskedLM.from_pretrained(model_version, output_attentions = True)
    
    if random:
        config = model.config
        model = None
        model = RobertaForMaskedLM(config) 
    
    model.to(device)
    
    tokenizer = RobertaTokenizer.from_pretrained(model_version)
    
    raw_tokens  = data['code_tokens']
    code_tokens = tokenizer.tokenize(' '.join(raw_tokens))
    tokens  = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    token_idx = tokenizer.convert_tokens_to_ids(tokens)

    inputs  = torch.tensor(token_idx).unsqueeze(0).to(device)

    outputs = model(inputs)

    attention = outputs.attentions

    all_att, tokens = merge_tokens_and_attention(code_tokens, raw_tokens, attention, 1, break_text_tokens = False)
    
    all_att = all_att.detach().cpu().numpy()

    return all_att, tokens
    
def get_attention_plbart(data, device = 'cuda:0', random = False):
    model_version = 'uclanlp/plbart-base'
    tokenizer = PLBartTokenizer.from_pretrained(model_version)
    model = PLBartForConditionalGeneration.from_pretrained(model_version, output_attentions = True)
    
    if random: 
       config = model.config
       model = None
       model = PLBartForConditionalGeneration(config) 
    
    model.to(device)
    
    raw_tokens  = data['code_tokens']
    code_tokens = tokenizer.tokenize(' '.join(raw_tokens))
    tokens  = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    token_idx = tokenizer.convert_tokens_to_ids(tokens)
    
    inputs = torch.tensor(token_idx).unsqueeze(0).to(device)
    
    outputs = model(inputs)
    
    attention = outputs.encoder_attentions   
    
    all_att, tokens = merge_tokens_and_attention(code_tokens, raw_tokens, attention, 1, break_text_tokens = False, special_char = '▁', mode='plbart')
    
    all_att = all_att.detach().cpu().numpy()
    
    return all_att, tokens
    

def get_attention_codeT5p(data, model_version = 'Salesforce/codet5p-220m', device='cuda', random = False):
    
    tokenizer = RobertaTokenizer.from_pretrained(model_version)
    model = T5ForConditionalGeneration.from_pretrained(model_version, output_attentions = True)
    
    if random:
        config = model.config
        model = None
        model = T5ForConditionalGeneration(config) 
    
    model.to(device)
    
    raw_tokens = data['code_tokens']
    code_tokens = tokenizer.tokenize(' '.join(raw_tokens))
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    token_idx = tokenizer.convert_tokens_to_ids(tokens)
    inputs = torch.tensor(token_idx).unsqueeze(0).to(device)
    
    label = ' '.join(data['docstring_tokens'])
    labels = tokenizer(label, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids = inputs, labels = labels)
    
    encoder_attention = outputs.encoder_attentions
    
    all_att, tokens = merge_tokens_and_attention(code_tokens, raw_tokens, encoder_attention, 1, break_text_tokens = False)
    all_att = all_att.detach().cpu().numpy()
    
    return all_att, tokens
        
   
def get_attention_codeT5p_2b(data, model, tokenizer, device='cuda', random = False):
    
    
    
    raw_tokens = data['code_tokens']
    code_tokens = tokenizer.tokenize(' '.join(raw_tokens))
    tokens = code_tokens
    token_idx = tokenizer.convert_tokens_to_ids(tokens)
    inputs = torch.tensor(token_idx).unsqueeze(0).to(device)
    
    label = ' '.join(data['docstring_tokens'])
    labels = tokenizer(label, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids = inputs, labels = labels)
    
    encoder_attention = outputs.encoder_attentions
    
    try:
        all_att, tokens = merge_tokens_and_attention(code_tokens, raw_tokens, encoder_attention ,0, None, break_text_tokens = False)
        all_att = all_att.detach().cpu().numpy()
    
        return all_att, tokens

    except:
        print("invalid code")
        
        
def get_attention_codeT5p_2b_dec(data, model, tokenizer, device='cuda', random = False):
    
    
    
    raw_tokens = data['code_tokens']
    code_tokens = tokenizer.tokenize(' '.join(raw_tokens))
    tokens = code_tokens
    token_idx = tokenizer.convert_tokens_to_ids(tokens)
    inputs = torch.tensor(token_idx).unsqueeze(0).to(device)
    
    label = ' '.join(data['code_tokens'])
    labels = tokenizer(label, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids = inputs, labels = labels)
    
    decoder_attention = outputs.decoder_attentions
    
    try:
        all_att, tokens = merge_tokens_and_attention(code_tokens, raw_tokens, decoder_attention ,0, None, break_text_tokens = False)
        all_att = all_att.detach().cpu().numpy()
    
        return all_att, tokens

    except:
        print("invalid code")

def get_attention_codegen(data, model, tokenizer, device='cuda', random = False):
    
    #tokenizer = AutoTokenizer.from_pretrained(model_version)
    #model = AutoModelForCausalLM.from_pretrained(model_version, output_attentions = True, trust_remote_code=True) 
    
    #model.to(device)
    
    raw_tokens = data['code_tokens']
    code_tokens = tokenizer.tokenize(' '.join(raw_tokens))
    #tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    tokens = code_tokens
    token_idx = tokenizer.convert_tokens_to_ids(tokens)
    inputs = torch.tensor(token_idx).unsqueeze(0).to(device)
    
    outputs = model(inputs)
    
    attention = outputs.attentions
    
    try:
        all_att, tokens = merge_tokens_and_attention(code_tokens, raw_tokens, attention, 0, None, break_text_tokens = False)
        all_att = all_att.detach().cpu().numpy()
        return all_att, tokens
    except:
        print("invalid code")
        
    

