from torch import nn
from transformers import PreTrainedModel
_TENSOR_RETURNING_LAYERS = {'Qwen3DecoderLayer', 'Qwen3MoeDecoderLayer'}

class TSVLayer(nn.Module):

    def __init__(self, tsv, lam):
        super().__init__()
        self.tsv = tsv
        self.lam = lam

    def forward(self, x):
        if self.tsv is not None:
            v = self.lam[0] * self.tsv.to(device=x.device, dtype=x.dtype)
            v = v.expand(-1, x.shape[1], -1)
            x = x + v
        return x

class LlamaDecoderLayerWrapper(nn.Module):

    def __init__(self, llama_decoder_layer, tsv_layer, model_name=''):
        super().__init__()
        self.llama_decoder_layer = llama_decoder_layer
        self.tsv_layer = tsv_layer
        self.model_name = model_name
        cls_name = type(llama_decoder_layer).__name__
        self._returns_tensor = cls_name in _TENSOR_RETURNING_LAYERS
        for attr in ['attention_type']:
            if hasattr(llama_decoder_layer, attr):
                setattr(self, attr, getattr(llama_decoder_layer, attr))

    def forward(self, *args, **kwargs):
        outputs = self.llama_decoder_layer(*args, **kwargs)
        if self._returns_tensor:
            return self.tsv_layer(outputs)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            hidden_states = self.tsv_layer(hidden_states)
            return (hidden_states,) + outputs[1:]
        if isinstance(outputs, list):
            hidden_states = outputs[0]
            hidden_states = self.tsv_layer(hidden_states)
            outputs[0] = hidden_states
            return outputs
        return self.tsv_layer(outputs)

def get_nested_attr(obj, attr_path):
    for attr in attr_path.split('.'):
        obj = getattr(obj, attr)
    return obj

def find_longest_modulelist(model, path=''):
    (longest_path, longest_len) = (path, 0)
    for (name, child) in model.named_children():
        child_path = f'{path}.{name}' if path else name
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            (longest_len, longest_path) = (len(child), child_path)
        (cp, cl) = find_longest_modulelist(child, child_path)
        if cl > longest_len:
            (longest_len, longest_path) = (cl, cp)
    return (longest_path, longest_len)

def get_layers(model: PreTrainedModel):
    (longest_path, _) = find_longest_modulelist(model)
    return get_nested_attr(model, longest_path)

def add_tsv_layers(model: PreTrainedModel, tsv, alpha: list, str_layer: int, model_name: str=''):
    layers = get_layers(model)
    assert str_layer < len(layers), f'str_layer={str_layer} >= num_layers={len(layers)}'
    decoder_layer = layers[str_layer]
    layers[str_layer] = LlamaDecoderLayerWrapper(decoder_layer, TSVLayer(tsv, alpha), model_name)
