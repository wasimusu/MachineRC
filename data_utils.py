import numpy as np
import torch
import config

config = config.Config()


def get_mask_from_seq_len(seq_mask):
    seq_lens = np.sum(seq_mask, 1)
    max_len = np.max(seq_lens)
    indices = np.arange(0, max_len)
    mask = (indices < np.expand_dims(seq_lens, 1)).astype(int)
    return mask


def get_data(batch, is_train=True):
    qn_mask = get_mask_from_seq_len(batch.qn_mask)
    qn_mask_var = torch.from_numpy(qn_mask).long()

    context_mask = get_mask_from_seq_len(batch.context_mask)
    context_mask_var = torch.from_numpy(context_mask).long()

    qn_seq_var = torch.from_numpy(batch.qn_ids).long()
    context_seq_var = torch.from_numpy(batch.context_ids).long()

    if config.mode == "train":
        span_var = torch.from_numpy(batch.ans_span).long()

    if is_train:
        qn_mask_var = qn_mask_var.to(config.device)
        context_mask_var = context_mask_var.to(config.device)
        qn_seq_var = qn_seq_var.to(config.device)
        context_seq_var = context_seq_var.to(config.device)
        if is_train:
            span_var = span_var.to(config.device)

    if config.mode == "train":
        return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var, span_var
    else:
        return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var
