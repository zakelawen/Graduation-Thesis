import torch
import torch.nn.functional as F

def collate_fn(prompts, labels, pad_id: int=0):
    max_seq_len = max((prompt.size(1) for prompt in prompts))
    batch_size = len(prompts)
    dtype = prompts[0].dtype
    prompts_padded = torch.full((batch_size, max_seq_len), pad_id, dtype=dtype)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    for (i, prompt) in enumerate(prompts):
        seq_len = prompt.size(1)
        prompts_padded[i, :seq_len] = prompt.squeeze(0)
        attention_mask[i, :seq_len] = 1
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return (prompts_padded, labels_tensor, attention_mask)

def get_last_non_padded_token_rep(hidden_states, attention_mask):
    mask = attention_mask
    if mask.dim() == 3:
        mask = mask.squeeze(1)
    lengths = mask.sum(dim=1).long()
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, lengths - 1]

def compute_ot_loss_cos(last_token_rep, centroids, pseudo_label, cos_temp):
    last_token_rep = F.normalize(last_token_rep.float(), p=2, dim=-1)
    centroids_norm = F.normalize(centroids.float(), p=2, dim=-1)
    similarities = torch.matmul(last_token_rep, centroids_norm.T)
    similarities = similarities / cos_temp
    pt = F.softmax(similarities, dim=-1)
    loss = -torch.sum(pseudo_label.float() * torch.log(pt + 1e-08)) / pseudo_label.shape[0]
    return (loss, similarities)

def update_centroids_ema_hard(centroids, last_token_rep, pseudo_label, ema_decay):
    last_token_rep_norm = F.normalize(last_token_rep.float(), p=2, dim=1)
    centroids_f = F.normalize(centroids.float(), p=2, dim=1)
    max_indices = torch.argmax(pseudo_label, dim=1)
    discrete_labels = torch.zeros_like(pseudo_label, dtype=torch.float32)
    discrete_labels[torch.arange(pseudo_label.size(0)), max_indices] = 1
    weighted_sum = torch.matmul(discrete_labels.T, last_token_rep_norm)
    pseudo_label_sum = discrete_labels.sum(dim=0).unsqueeze(1) + 1e-08
    new_centroids_batch = weighted_sum / pseudo_label_sum
    updated_centroids = F.normalize(ema_decay * centroids_f + (1 - ema_decay) * new_centroids_batch, p=2, dim=-1)
    return updated_centroids
