from torch.nn import Module, BatchNorm2d

def get_trainables(paras):
    trainables = []
    for p in paras:
        if p.requires_grad:
            trainables.append(p)
    return trainables

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn

def set_bn_eval(m):
    if type(m) == BatchNorm2d:
        m.eval()
    