import warnings
import inspect
from typing import Optional, Callable, List, Union, TYPE_CHECKING
import torch
from torch.nn import functional as F
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.utils import GenerateOutput, GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput, NEED_SETUP_CACHE_CLASSES_MAPPING, QUANT_BACKEND_CLASSES_MAPPING
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.utils import ModelOutput, is_hqq_available, is_quanto_available, is_torchdynamo_compiling, logging
from transformers.cache_utils import DynamicCache, EncoderDecoderCache, QuantizedCacheConfig, OffloadedCache
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer
if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer
logger = logging.get_logger(__name__)

class LMWrapper:

    def __init__(self, model_name_or_path: str, cache_dir: str, device_map: str, max_memory: dict, offload_folder: str=None):
        self.model_name_or_path = model_name_or_path
        self.gen_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device_map, max_memory=max_memory, offload_folder=offload_folder).eval()
        self.device = self.gen_model.device

    @torch.no_grad()
    def generate_contrast(self, inputs: Optional[torch.Tensor]=None, inputs_pos: Optional[torch.Tensor]=None, inputs_neg: Optional[torch.Tensor]=None, attention_mask_pos: Optional[torch.Tensor]=None, attention_mask_neg: Optional[torch.Tensor]=None, generation_config: Optional[GenerationConfig]=None, logits_processor: Optional[LogitsProcessorList]=None, stopping_criteria: Optional[StoppingCriteriaList]=None, prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]]=None, synced_gpus: Optional[bool]=None, assistant_model: Optional['PreTrainedModel']=None, streamer: Optional['BaseStreamer']=None, negative_prompt_ids: Optional[torch.Tensor]=None, negative_prompt_attention_mask: Optional[torch.Tensor]=None, mode: Optional[str]=None, main_task: Optional[str]='generation', dummy: Optional[list]=None, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
        self.gen_model._validate_model_class()
        tokenizer = kwargs.pop('tokenizer', None)
        (generation_config, model_kwargs) = self.gen_model._prepare_generation_config(generation_config, **kwargs)
        if inputs_pos is not None:
            model_kwargs_pos = model_kwargs.copy()
            model_kwargs_pos.pop('attention_mask', None)
            if attention_mask_pos is not None:
                model_kwargs_pos['attention_mask'] = attention_mask_pos
        if inputs_neg is not None:
            model_kwargs_neg = model_kwargs.copy()
            model_kwargs_neg.pop('attention_mask', None)
            if attention_mask_neg is not None:
                model_kwargs_neg['attention_mask'] = attention_mask_neg
        self.gen_model._validate_model_kwargs(model_kwargs.copy())
        self.gen_model._validate_assistant(assistant_model)
        if synced_gpus is None:
            synced_gpus = is_deepspeed_zero3_enabled() and dist.get_world_size() > 1
        logits_processor = logits_processor or LogitsProcessorList()
        stopping_criteria = stopping_criteria or StoppingCriteriaList()
        accepts_attention_mask = 'attention_mask' in set(inspect.signature(self.gen_model.forward).parameters.keys())
        requires_attention_mask = 'encoder_outputs' not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get('attention_mask', None) is not None
        (inputs_tensor, model_input_name, model_kwargs) = self.gen_model._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        if inputs_pos is not None:
            (inputs_tensor_pos, _, model_kwargs_pos) = self.gen_model._prepare_model_inputs(inputs_pos, generation_config.bos_token_id, model_kwargs_pos)
        if inputs_neg is not None:
            (inputs_tensor_neg, _, model_kwargs_neg) = self.gen_model._prepare_model_inputs(inputs_neg, generation_config.bos_token_id, model_kwargs_neg)
        batch_size = inputs_tensor.shape[0]
        device = inputs_tensor.device
        self.gen_model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
        if not self.gen_model.config.is_encoder_decoder and (not is_torchdynamo_compiling()):
            if generation_config._pad_token_tensor is not None and batch_size > 1 and (len(inputs_tensor.shape) == 2) and (torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0):
                logger.warning("检测到右填充：decoder-only 模型建议 padding_side='left'。")
        force_cache = not self.gen_model.config.is_encoder_decoder and model_input_name == 'inputs_embeds'
        use_cache_flag = True if force_cache else generation_config.use_cache
        model_kwargs['use_cache'] = use_cache_flag
        if inputs_pos is not None:
            model_kwargs_pos['use_cache'] = use_cache_flag
        if inputs_neg is not None:
            model_kwargs_neg['use_cache'] = use_cache_flag
        if requires_attention_mask and accepts_attention_mask:
            if not kwargs_has_attention_mask:
                model_kwargs['attention_mask'] = self.gen_model._prepare_attention_mask_for_generation(inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor)
            if inputs_pos is not None and attention_mask_pos is None:
                model_kwargs_pos['attention_mask'] = self.gen_model._prepare_attention_mask_for_generation(inputs_tensor_pos, generation_config._pad_token_tensor, generation_config._eos_token_tensor)
            if inputs_neg is not None and attention_mask_neg is None:
                model_kwargs_neg['attention_mask'] = self.gen_model._prepare_attention_mask_for_generation(inputs_tensor_neg, generation_config._pad_token_tensor, generation_config._eos_token_tensor)
        if self.gen_model.config.is_encoder_decoder and 'encoder_outputs' not in model_kwargs:
            model_kwargs = self.gen_model._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, model_kwargs, model_input_name, generation_config)
            if inputs_pos is not None:
                model_kwargs_pos = self.gen_model._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor_pos, model_kwargs_pos, model_input_name, generation_config)
            if inputs_neg is not None:
                model_kwargs_neg = self.gen_model._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor_neg, model_kwargs_neg, model_input_name, generation_config)
        if self.gen_model.config.is_encoder_decoder:
            (input_ids, model_kwargs) = self.gen_model._prepare_decoder_input_ids_for_generation(batch_size=batch_size, model_input_name=model_input_name, model_kwargs=model_kwargs, decoder_start_token_id=generation_config._decoder_start_token_tensor, device=inputs_tensor.device)
            if inputs_pos is not None:
                (input_ids_pos, model_kwargs_pos) = self.gen_model._prepare_decoder_input_ids_for_generation(batch_size=batch_size, model_input_name=model_input_name, model_kwargs=model_kwargs_pos, decoder_start_token_id=generation_config._decoder_start_token_tensor, device=inputs_tensor_pos.device)
            if inputs_neg is not None:
                (input_ids_neg, model_kwargs_neg) = self.gen_model._prepare_decoder_input_ids_for_generation(batch_size=batch_size, model_input_name=model_input_name, model_kwargs=model_kwargs_neg, decoder_start_token_id=generation_config._decoder_start_token_tensor, device=inputs_tensor_neg.device)
        else:
            input_ids = inputs_tensor if model_input_name == 'input_ids' else model_kwargs.pop('input_ids')
            if inputs_pos is not None:
                input_ids_pos = inputs_tensor_pos if model_input_name == 'input_ids' else model_kwargs_pos.pop('input_ids')
            if inputs_neg is not None:
                input_ids_neg = inputs_tensor_neg if model_input_name == 'input_ids' else model_kwargs_neg.pop('input_ids')
        if getattr(generation_config, 'token_healing', False) and hasattr(self.gen_model, 'heal_tokens'):
            input_ids = self.gen_model.heal_tokens(input_ids, tokenizer)
            if inputs_pos is not None:
                input_ids_pos = self.gen_model.heal_tokens(input_ids_pos, tokenizer)
            if inputs_neg is not None:
                input_ids_neg = self.gen_model.heal_tokens(input_ids_neg, tokenizer)
        if streamer is not None:
            streamer.put(input_ids.cpu())
        input_ids_length = input_ids.shape[-1]
        if inputs_pos is not None:
            input_ids_length_pos = input_ids_pos.shape[-1]
        if inputs_neg is not None:
            input_ids_length_neg = input_ids_neg.shape[-1]
        has_default_max_length = kwargs.get('max_length') is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get('min_length') is None and generation_config.min_length is not None
        generation_config = self.gen_model._prepare_generated_length(generation_config=generation_config, has_default_max_length=has_default_max_length, has_default_min_length=has_default_min_length, model_input_name=model_input_name, inputs_tensor=inputs_tensor, input_ids_length=input_ids_length)
        cache_name = 'cache_params' if 'mamba' in self.gen_model.__class__.__name__.lower() else 'past_key_values'
        if assistant_model is not None and generation_config.cache_implementation is not None and self.gen_model._supports_default_dynamic_cache():
            logger.warning_once('assistant_model 已提供，将使用 dynamic cache。')
            generation_config.cache_implementation = None
        if model_kwargs.get(cache_name) is not None and is_torchdynamo_compiling():
            raise ValueError('torch.compile 情况下不支持显式传 past_key_values。')
        if generation_config.cache_implementation is not None and model_kwargs.get(cache_name) is not None:
            raise ValueError('cache_implementation 与 past_key_values 只能二选一。')
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == 'static' and (not self.gen_model._supports_static_cache):
                    raise ValueError('该模型不支持 static cache。')
                model_kwargs[cache_name] = self.gen_model._get_cache(cache_implementation=generation_config.cache_implementation, max_batch_size=generation_config.num_beams * generation_config.num_return_sequences * batch_size, max_cache_len=generation_config.max_length, device=device, model_kwargs=model_kwargs)
            elif generation_config.cache_implementation == 'quantized':
                if not self.gen_model._supports_quantized_cache:
                    raise ValueError('该模型不支持 quantized cache。')
                cache_config = generation_config.cache_config or QuantizedCacheConfig()
                cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]
                if cache_config.backend == 'quanto' and (not is_quanto_available()):
                    raise ImportError('需要安装 quanto：pip install quanto')
                elif cache_config.backend == 'HQQ' and (not is_hqq_available()):
                    raise ImportError('需要安装 hqq：pip install hqq')
                model_kwargs[cache_name] = cache_class(cache_config)
            elif generation_config.cache_implementation == 'offloaded':
                model_kwargs[cache_name] = OffloadedCache()
        elif generation_config.cache_implementation is None and self.gen_model._supports_default_dynamic_cache():
            past = model_kwargs.get(cache_name, None)
            requires_cross_attention_cache = self.gen_model.config.is_encoder_decoder or model_kwargs.get('encoder_outputs') is not None
            if past is None:
                model_kwargs[cache_name] = DynamicCache() if not requires_cross_attention_cache else EncoderDecoderCache(DynamicCache(), DynamicCache())
                use_dynamic_cache_by_default = True
            elif isinstance(past, tuple):
                model_kwargs[cache_name] = DynamicCache.from_legacy_cache(past) if not requires_cross_attention_cache else EncoderDecoderCache.from_legacy_cache(past)
                use_dynamic_cache_by_default = True
            else:
                use_dynamic_cache_by_default = False
        else:
            use_dynamic_cache_by_default = False
        self.gen_model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        generation_mode = generation_config.get_generation_mode(assistant_model)
        if streamer is not None and generation_config.num_beams > 1:
            raise ValueError('暂不支持在 beam 下使用 streamer。')
        if not is_torchdynamo_compiling() and self.gen_model.device.type != input_ids.device.type:
            warnings.warn('input_ids 与模型设备不一致，可能导致性能问题。', UserWarning)
        prepared_logits_processor = self.gen_model._get_logits_processor(generation_config=generation_config, input_ids_seq_length=input_ids_length, encoder_input_ids=inputs_tensor, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, logits_processor=logits_processor, device=inputs_tensor.device, model_kwargs=model_kwargs, negative_prompt_ids=negative_prompt_ids, negative_prompt_attention_mask=negative_prompt_attention_mask)
        if inputs_pos is not None:
            prepared_logits_processor_pos = self.gen_model._get_logits_processor(generation_config=generation_config, input_ids_seq_length=input_ids_length_pos, encoder_input_ids=inputs_tensor_pos, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, logits_processor=logits_processor, device=inputs_tensor_pos.device, model_kwargs=model_kwargs_pos, negative_prompt_ids=negative_prompt_ids, negative_prompt_attention_mask=negative_prompt_attention_mask)
        if inputs_neg is not None:
            prepared_logits_processor_neg = self.gen_model._get_logits_processor(generation_config=generation_config, input_ids_seq_length=input_ids_length_neg, encoder_input_ids=inputs_tensor_neg, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, logits_processor=logits_processor, device=inputs_tensor_neg.device, model_kwargs=model_kwargs_neg, negative_prompt_ids=negative_prompt_ids, negative_prompt_attention_mask=negative_prompt_attention_mask)
        prepared_stopping_criteria = self.gen_model._get_stopping_criteria(generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs)
        if (inputs_pos is not None or inputs_neg is not None) and mode == 'acd' and (generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH, GenerationMode.GROUP_BEAM_SEARCH, GenerationMode.CONSTRAINED_BEAM_SEARCH)):
            raise ValueError('ACD 目前仅支持 greedy/sample，不支持 beam 系列。')
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            raise ValueError('本包装未对 assisted 模式做适配。')
        elif generation_mode == GenerationMode.DOLA_GENERATION:
            raise ValueError('本包装未对 DoLa 模式做适配。')
        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs['use_cache']:
                raise ValueError('contrastive search 需要 use_cache=True')
            result = self.gen_model._contrastive_search(input_ids, logits_processor=prepared_logits_processor, stopping_criteria=prepared_stopping_criteria, generation_config=generation_config, synced_gpus=synced_gpus, streamer=streamer, **model_kwargs)
        elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            prepared_logits_warper = self.gen_model._get_logits_warper(generation_config, device=input_ids.device) if generation_config.do_sample else None
            (input_ids, model_kwargs) = self.gen_model._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_return_sequences, is_encoder_decoder=self.gen_model.config.is_encoder_decoder, **model_kwargs)
            if inputs_pos is not None:
                (input_ids_pos, model_kwargs_pos) = self.gen_model._expand_inputs_for_generation(input_ids=input_ids_pos, expand_size=generation_config.num_return_sequences, is_encoder_decoder=self.gen_model.config.is_encoder_decoder, **model_kwargs_pos)
            if inputs_neg is not None:
                (input_ids_neg, model_kwargs_neg) = self.gen_model._expand_inputs_for_generation(input_ids=input_ids_neg, expand_size=generation_config.num_return_sequences, is_encoder_decoder=self.gen_model.config.is_encoder_decoder, **model_kwargs_neg)
            result = self._my_sample(input_ids, input_ids_pos=input_ids_pos if inputs_pos is not None else None, input_ids_neg=input_ids_neg if inputs_neg is not None else None, logits_processor=prepared_logits_processor, logits_processor_pos=prepared_logits_processor_pos if inputs_pos is not None else None, logits_processor_neg=prepared_logits_processor_neg if inputs_neg is not None else None, logits_warper=prepared_logits_warper, stopping_criteria=prepared_stopping_criteria, generation_config=generation_config, synced_gpus=synced_gpus, streamer=streamer, model_kwargs_pos=model_kwargs_pos if inputs_pos is not None else None, model_kwargs_neg=model_kwargs_neg if inputs_neg is not None else None, mode=mode, main_task=main_task, dummy=dummy, **model_kwargs)
        elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
            prepared_logits_warper = self.gen_model._get_logits_warper(generation_config, device=input_ids.device) if generation_config.do_sample else None
            beam_scorer = BeamSearchScorer(batch_size=batch_size, num_beams=generation_config.num_beams, device=inputs_tensor.device, length_penalty=generation_config.length_penalty, do_early_stopping=generation_config.early_stopping, num_beam_hyps_to_keep=generation_config.num_return_sequences, max_length=generation_config.max_length)
            (input_ids, model_kwargs) = self.gen_model._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_beams, is_encoder_decoder=self.gen_model.config.is_encoder_decoder, **model_kwargs)
            result = self.gen_model._beam_search(input_ids, beam_scorer, logits_processor=prepared_logits_processor, logits_warper=prepared_logits_warper, stopping_criteria=prepared_stopping_criteria, generation_config=generation_config, synced_gpus=synced_gpus, **model_kwargs)
        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            beam_scorer = BeamSearchScorer(batch_size=batch_size, num_beams=generation_config.num_beams, device=inputs_tensor.device, length_penalty=generation_config.length_penalty, do_early_stopping=generation_config.early_stopping, num_beam_hyps_to_keep=generation_config.num_return_sequences, num_beam_groups=generation_config.num_beam_groups, max_length=generation_config.max_length)
            (input_ids, model_kwargs) = self.gen_model._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_beams, is_encoder_decoder=self.gen_model.config.is_encoder_decoder, **model_kwargs)
            result = self.gen_model._group_beam_search(input_ids, beam_scorer, logits_processor=prepared_logits_processor, stopping_criteria=prepared_stopping_criteria, generation_config=generation_config, synced_gpus=synced_gpus, **model_kwargs)
        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints
            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError('`force_words_ids` 类型非法。')
                if not isinstance(generation_config.force_words_ids, list) or len(generation_config.force_words_ids) == 0:
                    typeerror()
                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_ids, list) for token_ids in word_ids)):
                            typeerror()
                        if any((any((not isinstance(token_id, int) or token_id < 0 for token_id in token_ids)) for token_ids in word_ids)):
                            typeerror()
                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0 for token_id in word_ids)):
                            typeerror()
                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)
            constrained_beam_scorer = ConstrainedBeamSearchScorer(constraints=final_constraints, batch_size=batch_size, num_beams=generation_config.num_beams, device=inputs_tensor.device, length_penalty=generation_config.length_penalty, do_early_stopping=generation_config.early_stopping, num_beam_hyps_to_keep=generation_config.num_return_sequences, max_length=generation_config.max_length)
            (input_ids, model_kwargs) = self.gen_model._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_beams, is_encoder_decoder=self.gen_model.config.is_encoder_decoder, **model_kwargs)
            result = self.gen_model._constrained_beam_search(input_ids, constrained_beam_scorer=constrained_beam_scorer, logits_processor=prepared_logits_processor, stopping_criteria=prepared_stopping_criteria, generation_config=generation_config, synced_gpus=synced_gpus, **model_kwargs)
        if use_dynamic_cache_by_default and generation_config.return_legacy_cache:
            if isinstance(result, ModelOutput) and hasattr(result, 'past_key_values'):
                if isinstance(result.past_key_values, (DynamicCache, EncoderDecoderCache)):
                    result.past_key_values = result.past_key_values.to_legacy_cache()
        return result

    def _my_sample(self, input_ids: torch.LongTensor, input_ids_pos: Optional[torch.LongTensor]=None, input_ids_neg: Optional[torch.LongTensor]=None, logits_processor: LogitsProcessorList=None, logits_processor_pos: Optional[LogitsProcessorList]=None, logits_processor_neg: Optional[LogitsProcessorList]=None, stopping_criteria: StoppingCriteriaList=None, generation_config: GenerationConfig=None, synced_gpus: bool=None, streamer: Optional['BaseStreamer']=None, logits_warper: Optional[LogitsProcessorList]=None, model_kwargs_pos: Optional[dict]=None, model_kwargs_neg: Optional[dict]=None, mode: Optional[str]=None, main_task: Optional[str]='generation', dummy: Optional[list]=None, **model_kwargs) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = getattr(generation_config, 'output_logits', False)
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any((hasattr(criteria, 'eos_token_id') for criteria in stopping_criteria))
        do_sample = generation_config.do_sample
        acd_debug = bool(getattr(generation_config, 'acd_debug', False))
        acd_debug_topk = int(getattr(generation_config, 'acd_debug_topk', 0))
        acd_debug_steps = int(getattr(generation_config, 'acd_debug_steps', 0))
        acd_dbg = {'chosen_tokens': [], 'scores_before_acd_topk': [], 'scores_after_acd_topk': [], 'scores_delta_topk': [], 'topk_token_ids': [], 'ent': [], 'ent_pos': [], 'ent_neg': [], 'd_alpha': [], 'd_alpha_neg': []}
        if do_sample and (not isinstance(logits_warper, LogitsProcessorList)):
            raise ValueError('do_sample=True 时 logits_warper 必须是 LogitsProcessorList 实例。')
        scores = () if return_dict_in_generate and output_scores else None
        raw_logits = () if return_dict_in_generate and output_logits else None
        decoder_attentions = () if return_dict_in_generate and output_attentions else None
        cross_attentions = () if return_dict_in_generate and output_attentions else None
        decoder_hidden_states = () if return_dict_in_generate and output_hidden_states else None
        if return_dict_in_generate and self.gen_model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs['encoder_outputs'].get('attentions') if output_attentions else None
            encoder_hidden_states = model_kwargs['encoder_outputs'].get('hidden_states') if output_hidden_states else None
        (batch_size, cur_len) = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self.gen_model._get_initial_cache_position(input_ids, model_kwargs)
        if input_ids_pos is not None:
            model_kwargs_pos = self.gen_model._get_initial_cache_position(input_ids_pos, model_kwargs_pos)
        if input_ids_neg is not None:
            model_kwargs_neg = self.gen_model._get_initial_cache_position(input_ids_neg, model_kwargs_neg)
        is_acd = mode == 'acd' and (input_ids_pos is not None or input_ids_neg is not None)
        acd_alpha_pos = float(getattr(generation_config, 'acd_alpha_pos', 1.0))
        acd_alpha_neg = float(getattr(generation_config, 'acd_alpha_neg', 1.0))
        while self.gen_model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length):
            model_inputs = self.gen_model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if input_ids_pos is not None:
                model_inputs_pos = self.gen_model.prepare_inputs_for_generation(input_ids_pos, **model_kwargs_pos)
            if input_ids_neg is not None:
                model_inputs_neg = self.gen_model.prepare_inputs_for_generation(input_ids_neg, **model_kwargs_neg)
            if output_attentions:
                model_inputs.update({'output_attentions': True})
                if input_ids_pos is not None:
                    model_inputs_pos.update({'output_attentions': True})
                if input_ids_neg is not None:
                    model_inputs_neg.update({'output_attentions': True})
            if output_hidden_states:
                model_inputs.update({'output_hidden_states': True})
                if input_ids_pos is not None:
                    model_inputs_pos.update({'output_hidden_states': True})
                if input_ids_neg is not None:
                    model_inputs_neg.update({'output_hidden_states': True})
            outputs = self.gen_model(**model_inputs, return_dict=True)
            if input_ids_pos is not None:
                outputs_pos = self.gen_model(**model_inputs_pos, return_dict=True)
            if input_ids_neg is not None:
                outputs_neg = self.gen_model(**model_inputs_neg, return_dict=True)
            if synced_gpus and this_peer_finished:
                continue
            next_token_logits = outputs.logits[:, -1, :].clone()
            if input_ids_pos is not None:
                next_token_logits_pos = outputs_pos.logits[:, -1, :].clone()
            if input_ids_neg is not None:
                next_token_logits_neg = outputs_neg.logits[:, -1, :].clone()
            if return_dict_in_generate and output_logits:
                raw_logits += (next_token_logits,)
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if input_ids_pos is not None:
                next_token_scores_pos = logits_processor_pos(input_ids_pos, next_token_logits_pos)
            if input_ids_neg is not None:
                next_token_scores_neg = logits_processor_neg(input_ids_neg, next_token_logits_neg)
            scores_before_acd = next_token_scores.clone()
            ent = ent_pos = ent_neg = None
            d_alpha = d_alpha_neg = None
            if is_acd:
                eps = 1e-10
                log2_const = torch.log(torch.tensor(2.0, device=next_token_scores.device))
                base = next_token_scores.float()
                probs = F.softmax(base, dim=-1)
                ent = -(probs * (probs.add(eps).log() / log2_const)).sum(dim=-1)
                if input_ids_pos is not None:
                    base_pos = next_token_scores_pos.float()
                    probs_pos = F.softmax(base_pos, dim=-1)
                    ent_pos = -(probs_pos * (probs_pos.add(eps).log() / log2_const)).sum(dim=-1)
                    d_alpha = (ent / (ent_pos + ent + eps)).unsqueeze(-1).clamp_(0.0, 1.0)
                    next_token_scores = next_token_scores + acd_alpha_pos * d_alpha * (next_token_scores_pos - next_token_scores)
                if input_ids_neg is not None:
                    base_neg = next_token_scores_neg.float()
                    probs_neg = F.softmax(base_neg, dim=-1)
                    ent_neg = -(probs_neg * (probs_neg.add(eps).log() / log2_const)).sum(dim=-1)
                    d_alpha_neg = (ent / (ent_neg + ent + eps)).unsqueeze(-1).clamp_(0.0, 1.0)
                    next_token_scores = next_token_scores - acd_alpha_neg * d_alpha_neg * (next_token_scores_neg - next_token_scores)
            scores_after_acd = next_token_scores
            if acd_debug:
                sb = scores_before_acd[0]
                sa = scores_after_acd[0]
                k = acd_debug_topk if acd_debug_topk > 0 else sb.numel()
                topk_before = torch.topk(sb, k)
                idx = topk_before.indices
                vals_before = topk_before.values
                vals_after = sa.index_select(0, idx)
                delta = vals_after - vals_before
                acd_dbg['topk_token_ids'].append(idx.detach().cpu())
                acd_dbg['scores_before_acd_topk'].append(vals_before.detach().cpu())
                acd_dbg['scores_after_acd_topk'].append(vals_after.detach().cpu())
                acd_dbg['scores_delta_topk'].append(delta.detach().cpu())
                acd_dbg['ent'].append(None if ent is None else ent[0].detach().cpu())
                acd_dbg['ent_pos'].append(None if ent_pos is None else ent_pos[0].detach().cpu())
                acd_dbg['ent_neg'].append(None if ent_neg is None else ent_neg[0].detach().cpu())
                acd_dbg['d_alpha'].append(None if d_alpha is None else d_alpha[0, 0].detach().cpu())
                acd_dbg['d_alpha_neg'].append(None if d_alpha_neg is None else d_alpha_neg[0, 0].detach().cpu())
                if acd_debug_steps > 0 and len(acd_dbg['ent']) >= acd_debug_steps:
                    acd_debug = False
            if return_dict_in_generate and output_scores:
                scores += (scores_after_acd,)
            if do_sample:
                next_token_scores = logits_warper(input_ids, scores_after_acd)
                probs_final = F.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs_final, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(scores_after_acd, dim=-1)
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            if acd_dbg is not None:
                if len(acd_dbg['chosen_tokens']) < len(acd_dbg['ent']):
                    acd_dbg['chosen_tokens'].append(int(next_tokens[0].item()))
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if input_ids_pos is not None:
                input_ids_pos = torch.cat([input_ids_pos, next_tokens[:, None]], dim=-1)
            if input_ids_neg is not None:
                input_ids_neg = torch.cat([input_ids_neg, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self.gen_model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.gen_model.config.is_encoder_decoder)
            if input_ids_pos is not None:
                model_kwargs_pos = self.gen_model._update_model_kwargs_for_generation(outputs_pos, model_kwargs_pos, is_encoder_decoder=self.gen_model.config.is_encoder_decoder)
            if input_ids_neg is not None:
                model_kwargs_neg = self.gen_model._update_model_kwargs_for_generation(outputs_neg, model_kwargs_neg, is_encoder_decoder=self.gen_model.config.is_encoder_decoder)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            del outputs
            if input_ids_pos is not None:
                del outputs_pos
            if input_ids_neg is not None:
                del outputs_neg
        if streamer is not None:
            streamer.end()
        if return_dict_in_generate:
            if self.gen_model.config.is_encoder_decoder:
                out = GenerateEncoderDecoderOutput(sequences=input_ids, scores=scores, logits=raw_logits, encoder_attentions=encoder_attentions if return_dict_in_generate else None, encoder_hidden_states=encoder_hidden_states if return_dict_in_generate else None, decoder_attentions=decoder_attentions, cross_attentions=cross_attentions, decoder_hidden_states=decoder_hidden_states, past_key_values=model_kwargs.get('past_key_values'))
            else:
                out = GenerateDecoderOnlyOutput(sequences=input_ids, scores=scores, logits=raw_logits, attentions=decoder_attentions, hidden_states=decoder_hidden_states, past_key_values=model_kwargs.get('past_key_values'))
            if acd_dbg['ent']:
                out.acd_debug = acd_dbg
            return out
        else:
            return input_ids
