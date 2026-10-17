import math
import random
import numpy as np
import torch
import torch.nn.functional as F
BRIEF_PROMPT = 'Answer the following question as briefly as possible.\n'

class EntailmentDeberta:

    def __init__(self, model_path='microsoft/deberta-v2-xlarge-mnli', device='cuda'):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    def check_implication(self, text1, text2):
        inputs = self.tokenizer(text1, text2, padding=True, truncation=True, return_tensors='pt').to(self.device)
        logits = self.model(**inputs).logits
        return (torch.argmax(F.softmax(logits, dim=1), dim=1).detach(), logits.detach())

def check_entail(text1, text2, model, entailment_option='bi'):
    (implication_1, logits_1) = model.check_implication(text1, text2)
    (implication_2, logits_2) = model.check_implication(text2, text1)
    if entailment_option == 'bi':
        equivalent = (implication_1 == 2) & (implication_2 == 2)
    elif entailment_option == 'a_entails_b':
        equivalent = implication_1 == 2
    elif entailment_option == 'b_entails_a':
        equivalent = implication_2 == 2
    else:
        equivalent = (implication_1 != 0) & (implication_2 != 0) & ~((implication_1 == 1) & (implication_2 == 1))
    return (equivalent, {'a_entails_b': logits_1, 'b_entails_a': logits_2})

def calculate_uncertainty_soft_batch(example, entailment_model, computation_chunk_size=128):
    bsz = len(example['question'])
    n_responses = len(example['response_text'][0])
    (questions, responses, answers, likelihood) = (example['question'], example['response_text'], example['answers'], example['likelihood'])
    (all_responses, all_answers, bid_counter) = ([], [], [0])
    for bi in range(bsz):
        n_gt = len(answers[bi])
        assert n_gt > 0, f'no ground truth for question: {questions[bi]}'
        all_responses += [[f'{questions[bi]} {r}' for r in responses[bi]]] * n_gt
        all_answers += [[f'{questions[bi]} {a}'] * n_responses for a in answers[bi]]
        bid_counter.append(bid_counter[-1] + n_gt * n_responses)
    all_responses = np.array(all_responses).flatten()
    all_answers = np.array(all_answers).flatten()
    (resp_entails_ans, ans_entails_resp) = ([], [])
    for i in range(0, len(all_responses), computation_chunk_size):
        (_, logits) = check_entail(all_responses[i:i + computation_chunk_size].tolist(), all_answers[i:i + computation_chunk_size].tolist(), entailment_model, 'loose')
        resp_entails_ans.extend(F.softmax(logits['a_entails_b'], dim=-1)[..., 2].cpu().numpy().tolist())
        ans_entails_resp.extend(F.softmax(logits['b_entails_a'], dim=-1)[..., 2].cpu().numpy().tolist())
    sepers = []
    for bi in range(bsz):
        r_e_a = np.array(resp_entails_ans[bid_counter[bi]:bid_counter[bi + 1]]).reshape(n_responses, -1)
        a_e_r = np.array(ans_entails_resp[bid_counter[bi]:bid_counter[bi + 1]]).reshape(n_responses, -1)
        lik = likelihood[bi][..., np.newaxis]
        entail_mass = np.stack([lik * r_e_a, lik * a_e_r], axis=-1)
        sepers.append(np.max(np.sum(np.max(entail_mass, axis=-1), axis=0)))
    return sepers

def create_collate_fn(keys):

    def collate(batch):
        return {k: [b[k] for b in batch] for k in keys}
    return collate

def make_prompt(question, context=None, max_words=512, instruction=BRIEF_PROMPT):
    prompt = instruction
    if context:
        words = context.split()
        prompt += f"Context: {(context if len(words) <= max_words else ' '.join(words[:max_words]))}\n"
    return prompt + f'Question: {question}\nAnswer:'

def gen_answers_batch(example, model, temperature, num_generations, sub_batch_size=1, max_new_tokens=None, device='cuda', max_context_words=512):
    prompt = make_prompt(example['question'], context=example['context'], max_words=max_context_words)
    (answers, log_liks) = ([], [])
    model.model.eval()
    with torch.no_grad():
        for idx in range(math.ceil(num_generations / sub_batch_size)):
            n = min(sub_batch_size, num_generations - idx * sub_batch_size)
            (a, ll, _) = model.batch_predict(prompt, num_generations=n, temperature=temperature, device=device, max_new_tokens=max_new_tokens)
            answers.extend(a)
            log_liks.extend(ll)
    example['responses'] = list(zip(answers, log_liks))
    return example

def process_item_for_seper(item, subsample=-1, n_repeat=0):
    if subsample > 0:
        random.seed(n_repeat)
        item['responses'] = random.sample(item['responses'], subsample)
    log_liks = [r[1] for r in item['responses']]
    log_liks_agg = [np.mean(ll) for ll in log_liks]
    log_z = np.log(np.sum(np.exp(log_liks_agg)))
    question = item['question']
    return {'question': question if question.endswith('?') else question + '?', 'answers': item['answers'], 'context': item['context'], 'context_label': item.get('context_label'), 'response_text': [r[0] for r in item['responses']], 'response_log_likelihood': log_liks, 'log_liks_agg': log_liks_agg, 'likelihood': np.exp(log_liks_agg - log_z)}
