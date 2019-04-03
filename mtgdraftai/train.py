import os
import sys
import json
import numpy as np
from random import random
import tensorflow as tf
from mtgdraftai.model import Model

def train():
    homedir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(homedir, "train_config.json"), "rt") as f:
        train_config = json.load(f)
    train_data, test_data, card_vocab_encode, card_vocab_decode = load_data(train_config)
    with open(os.path.join(homedir, "model_config.json"), "rt") as f:
        model_config = json.load(f)
    model_config['cards_count'] = len(card_vocab_decode)
    if not os.path.exists(train_config['output_dir']):
        os.makedirs(train_config['output_dir'])
    with open(os.path.join(train_config['output_dir'], "model_config.json"), "wt") as f:
        json.dump(model_config, f)
    with open(os.path.join(train_config['output_dir'], "model_vocab.json"), "wt") as f:
        json.dump(card_vocab_decode, f)
    sys.stdout.write("\nEncoding train data\n")
    train_data = encode_data(train_data, card_vocab_encode)
    sys.stdout.write("\nEncoding test data\n")
    test_data = encode_data(test_data, card_vocab_encode)
    with tf.Session() as sess:
        model = Model(sess, model_config)
        model.build()
        sess.run(tf.global_variables_initializer())
        run_train(model, sess, train_data, test_data, train_config)


def load_data(train_config):
    train_data = []
    test_data = []
    card_vocab = set()
    draft_filenames = os.listdir(train_config['source_dir'])
    sys.stdout.write("Loading data\n")
    for draft_id, draft_filename in enumerate(draft_filenames):
        if draft_id%100 == 0:
            sys.stdout.write("\r{} / {} : {}       ".format(draft_id, len(draft_filenames), draft_filename))
        with open(os.path.join(train_config['source_dir'], draft_filename), "rt") as f:
            draft_json = json.load(f)
            for round in draft_json['packs']:
                for booster in round:
                    card_vocab.update(booster)
            picked = np.zeros((len(draft_json['player_names']), 3*train_config['pack_size']), dtype=np.int32)
            for pack_num, booster in enumerate(draft_json['boosters']):
                for pick_num in range(train_config['pack_size']):
                    for player_pos, bot in enumerate(draft_json['bots']):
                        if pack_num % 2 == 0:
                            original_pack_pos = (player_pos-pick_num)%(len(draft_json['player_names']))
                        else:
                            original_pack_pos = (player_pos+pick_num)%(len(draft_json['player_names']))
                        pick_pos = draft_json['picks'][pack_num][player_pos][pick_num] - 1
                        if not bot and draft_json['packs'][pack_num][original_pack_pos][pick_pos]!=0:
                            sample = dict(pack=draft_json['packs'][pack_num][original_pack_pos][:],
                                          picks=picked[player_pos].tolist(),
                                          choice=pick_pos)
                            if random() < train_config['train_test_split']:
                                train_data.append(sample)
                            else:
                                test_data.append(sample)
                        picked[player_pos][pack_num*train_config['pack_size']+pick_num] = draft_json['packs'][pack_num][original_pack_pos][pick_pos]
                        draft_json['packs'][pack_num][original_pack_pos][pick_pos] = 0

    card_vocab_decode = [0]+sorted(card_vocab)
    card_vocab_encode = dict()
    for pos, id in enumerate(card_vocab_decode):
        card_vocab_encode[id] = pos
    return train_data, test_data, card_vocab_encode, card_vocab_decode


def encode_data(data, card_vocab_encode):
    for sample_id, sample in enumerate(data):
        if sample_id%100 == 0:
            sys.stdout.write("\r{} / {}     ".format(sample_id, len(data)))
        sample['picks'] = [card_vocab_encode[card] for card in sample['picks']]
        sample['pack'] = [card_vocab_encode[card] for card in sample['pack']]
    return data


def run_train(model, sess, train_data, test_data, train_config):
    saver = tf.train.Saver(sess.graph._collections['variables'])
    best_acc = 0
    for epoch_num in range(999):
        sys.stdout.write("\nEpoch {}\nTraining\n".format(epoch_num))
        losses = []
        for batch_num in range(len(train_data) // train_config['batch_size']):
            picks = [sample['picks'] for sample in train_data[train_config['batch_size']*batch_num:train_config['batch_size']*(batch_num+1)]]
            packs = [sample['pack'] for sample in train_data[train_config['batch_size']*batch_num:train_config['batch_size']*(batch_num+1)]]
            choice = [sample['choice'] for sample in train_data[train_config['batch_size']*batch_num:train_config['batch_size']*(batch_num+1)]]
            loss = model.train(packs, picks, choice, train_config['dropout'], train_config['lr'])
            losses.append(loss)
            if batch_num%10 == 0:
                sys.stdout.write("\r{}/{} loss={:.4f}     ".format(batch_num*train_config['batch_size'], len(train_data), np.mean(losses)))
        sys.stdout.write("\nTesting\n")
        losses = []
        accs = []
        for batch_num in range(len(test_data) // train_config['batch_size']):
            picks = [sample['picks'] for sample in test_data[train_config['batch_size']*batch_num:train_config['batch_size']*(batch_num+1)]]
            packs = [sample['pack'] for sample in test_data[train_config['batch_size']*batch_num:train_config['batch_size']*(batch_num+1)]]
            choice = [sample['choice'] for sample in test_data[train_config['batch_size']*batch_num:train_config['batch_size']*(batch_num+1)]]
            loss, acc = model.evaluate(packs, picks, choice)
            losses.append(loss)
            accs.append(acc)
            if batch_num%10 == 0:
                sys.stdout.write("\r{}/{} loss={:.4f} acc={:.4f}     ".format(batch_num*train_config['batch_size'], len(test_data), np.mean(losses), np.mean(accs)))
        if np.mean(accs) > best_acc:
            best_acc = np.mean(accs)
            sys.stdout.write("\nSaving best model acc = {:.4f}\n".format(best_acc))
            saver.save(sess, os.path.join(train_config['output_dir'],"transformer_model"))

if __name__ == "__main__":
    train()