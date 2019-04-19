"""
Train the Coattention Network for Query Answering
"""
import os
import json

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import networks as N
from data_util.vocab import get_glove
from data_util.data_batcher import get_batch_generator
from data_util.evaluate import exact_match_score, f1_score
from preprocessing.download_wordvecs import maybe_download
from config import Config
from data_utils import get_data
from data_utils import timeit

config = Config()

# Embeddings and word2id and id2word
glove_path = os.path.join(config.vectors_cache, "glove.6B.{}d.txt".format(config.embedding_dim))
if not os.path.exists(glove_path):
    print("\nDownloading wordvecs to {}".format(config.vectors_cache))
    if not os.path.exists(config.vectors_cache):
        os.makedirs(config.vectors_cache)
    maybe_download(config.glove_base_url, config.glove_filename, config.vectors_cache, 862182613)

emb_matrix, word2index, index2word = get_glove(glove_path, config.embedding_dim)

train_context_path = os.path.join(config.data_dir, "train.context")
train_qn_path = os.path.join(config.data_dir, "train.question")
train_ans_path = os.path.join(config.data_dir, "train.span")
dev_context_path = os.path.join(config.data_dir, "dev.context")
dev_qn_path = os.path.join(config.data_dir, "dev.question")
dev_ans_path = os.path.join(config.data_dir, "dev.span")


# @timeit
def step(model, optimizer, batch, params):
    """
    One batch of training
    :return: loss
    """
    # Here goes one batch of training
    q_seq, q_mask, d_seq, d_mask, target_span = get_data(batch, config.mode.lower() == 'train')
    model.zero_grad()
    # The loss is individual loss for each pair of question, context and answer
    loss, start_pos, end_pos = model(q_seq, q_mask, d_seq, d_mask, target_span)
    loss = torch.sum(loss)
    loss.backward(retain_graph=True)  # TODO : Is this causing memory consumption to increase
    clip_grad_norm_(params, config.max_grad_norm)
    optimizer.step()

    del start_pos, end_pos
    del q_mask, q_seq, d_mask, d_seq, target_span
    return loss


# @timeit
def evaluate(model, batch):
    """
    Evaluate the training and test set accuracy
    :return:
    """
    # Here goes one batch of training
    q_seq, q_mask, d_seq, d_mask, target_span = get_data(batch, config.mode.lower() == 'train')
    with torch.no_grad():
        # The loss is individual loss for each pair of question, context and answer
        loss, start_pos_pred, end_pos_pred = model(q_seq, q_mask, d_seq, d_mask, target_span)
        start_pos_pred = start_pos_pred.tolist()
        end_pos_pred = end_pos_pred.tolist()
        f1 = 0
        for i, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(
                zip(start_pos_pred, end_pos_pred, batch.ans_tokens)):
            pred_ans_tokens = batch.context_tokens[i][pred_ans_start: pred_ans_end + 1]
            prediction = " ".join(pred_ans_tokens)
            ground_truth = " ".join(true_ans_tokens)
            f1 += f1_score(prediction, ground_truth)
        f1 = f1 / (i + 1)
        return f1


def train(context_path, qn_path, ans_path):
    """ Train the network """
    model = N.CoattentionNetwork(device=config.device,
                                 hidden_size=config.hidden_size,
                                 emb_matrix=emb_matrix,
                                 num_encoder_layers=config.num_encoder_layers,
                                 num_fusion_bilstm_layers=config.num_fusion_bilstm_layers,
                                 num_decoder_layers=config.num_decoder_layers,
                                 batch_size=config.batch_size,
                                 max_dec_steps=config.max_dec_steps,
                                 fusion_dropout_rate=config.fusion_dropout_rate,
                                 encoder_bidirectional=config.encoder_bidirectional,
                                 decoder_bidirectional=config.decoder_bidirectional)

    # Select the parameters which require grad / backpropagation
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.SGD(params, lr=config.learning_rate, weight_decay=config.l2_norm)

    # Set up directories for this experiment
    if not os.path.exists(config.experiments_root_dir):
        os.makedirs(config.experiments_root_dir)

    serial_number = len(os.listdir(config.experiments_root_dir))
    if config.restore:
        serial_number -= 1  # Check into the latest model
    experiment_dir = os.path.join(config.experiments_root_dir, 'experiment_{}'.format(serial_number))

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    model_dir = os.path.join(experiment_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save config as config.json
    with open(os.path.join(experiment_dir, "config.json"), 'w') as fout:
        json.dump(vars(config), fout)

    iteration = 0
    if config.restore:
        saved_models = os.listdir(model_dir)
        if len(saved_models):
            print(saved_models)
            saved_models = [int(name.split('-')[-1]) for name in saved_models]
            latest_iter = max(saved_models)
            checkpoint_name = "checkpoint-embed{}-iter-{}".format(config.embedding_dim, latest_iter)
            checkpoint_name = os.path.join(model_dir, checkpoint_name)

            state = torch.load(checkpoint_name)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            iteration = state['iter']
            print("Model restored from ", checkpoint_name)
        else:
            print("Training with fresh parameters")

    for epoch in range(config.num_epochs):
        for batch in get_batch_generator(word2index, context_path, qn_path, ans_path,
                                         config.batch_size, config.context_len,
                                         config.question_len, discard_long=True):

            # When the batch is partially filled, ignore it.
            if batch.batch_size < config.batch_size:
                del batch
                continue

            # Take step in training
            loss = step(model, optimizer, batch, params)

            # Displaying results
            if iteration % config.evaluate_every == 0 and iteration % config.print_every != 0:
                print("Iter {}\t\tloss : {}\tf1 : {}".format(iteration, "%.2f" % loss, "%.2f" % -1))

            if iteration % config.evaluate_every == 0:
                f1 = evaluate(model, batch)
                print("Iter {}\t\tloss : {}\tf1 : {}".format(iteration, "%.2f" % loss, "%.2f" % f1))

                # Maybe you want to do random evaluations as well for sanity check

            # Saving the model
            if iteration % config.save_every == 0:
                state = {
                    'iter': iteration,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss
                }
                checkpoint_name = "checkpoint-embed{}-iter-{}".format(config.embedding_dim, iteration)

                fname = os.path.join(model_dir, checkpoint_name)
                torch.save(state, fname)

            del loss
            iteration += 1


if __name__ == '__main__':
    if config.mode == 'train':
        context_path = train_context_path
        qn_path = train_qn_path
        ans_path = train_ans_path
    else:
        context_path = dev_context_path
        qn_path = dev_qn_path
        ans_path = dev_ans_path
    train(context_path, qn_path, ans_path)
