from copy import deepcopy

def intersection(model_graph, program_graph):
    model_graph = deepcopy(model_graph)
    program_graph = deepcopy(program_graph)
    model_graph[model_graph > 0] = 1
    model_graph[model_graph == 0] = -1
    
    return (model_graph == program_graph).sum()

def IoU(model_graph, program_graph):
    i = intersection(model_graph, program_graph)
    u = (model_graph > 0).sum() + (program_graph == 1).sum() - i
    return i/u

def precision(model_graph, program_graph):
    tp = intersection(model_graph, program_graph)
    tp_fp = (model_graph > 0).sum()
    if tp_fp == 0:
        return 0
    return tp/tp_fp

def recall(model_graph, program_graph):
    tp = intersection(model_graph, program_graph)
    tp_fn = (program_graph == 1).sum()
    if tp_fn == 0:
        return 0
    return tp/tp_fn

def f_score(model_graph, program_graph):
    p = precision(model_graph, program_graph)
    r = recall(model_graph, program_graph)
    if p+r == 0:
        return 0
    return 2 * (p*r) / (p+r)
