import os
import runpy
import sys
from torch.utils.checkpoint import checkpoint
RAG_PROBE = os.environ.get('RAG_PROBE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rag_probe'))
SCRIPTS = os.path.join(RAG_PROBE, 'scripts')
sys.path.insert(0, SCRIPTS)
import step9_train_csv as base
_current = {}
_real_auto = base.AutoModelForCausalLM
_real_train = base.train_single_config

def _identity(returns_tuple):

    def forward(*args, **kwargs):
        h = args[0] if args else kwargs['hidden_states']
        return (h,) if returns_tuple else h
    return forward

def _checkpointed(forward):

    def wrapped(*args, **kwargs):
        return checkpoint(forward, *args, use_reentrant=False, **kwargs)
    return wrapped

def patch_layers(model, str_layer, cls_layer):
    model.config.use_cache = False
    layers = model.model.layers
    last = len(layers) - 1 if cls_layer == -1 else cls_layer
    for (i, layer) in enumerate(layers):
        if i > last:
            layer.forward = _identity(type(layer).__name__ != 'Qwen3DecoderLayer')
        elif i > str_layer:
            layer.forward = _checkpointed(layer.forward)
    return model

class _AutoModel:

    @staticmethod
    def from_pretrained(*args, **kwargs):
        model = _real_auto.from_pretrained(*args, **kwargs)
        return patch_layers(model, _current['str_layer'], _current['cls_layer'])

def _train_single_config(*args, **kwargs):
    _current.update(str_layer=kwargs['str_layer'], cls_layer=kwargs['cls_layer'])
    return _real_train(*args, **kwargs)
base.AutoModelForCausalLM = _AutoModel
base.train_single_config = _train_single_config
if __name__ == '__main__':
    sys.argv[0] = f'{SCRIPTS}/step9_train_csv_retrieved.py'
    runpy.run_path(sys.argv[0], run_name='__main__')
