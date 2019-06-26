# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.python.ops import standard_ops
import numpy as np
from math import pi, sqrt
import json

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

with open(os.path.join(__location__, "model_settings.json"), "rt") as f:
    MODEL_SETTINGS = json.load(f)
PACK_SIZE = 15

BASICS = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest'}

class Model(object):
    def __init__(self, sess):
        self.sess = sess

    def build_trainer(self):
        packs = tf.placeholder(tf.int32, shape=[None, PACK_SIZE], name="packs")
        picks = tf.placeholder(tf.int32, shape=[None, 3*PACK_SIZE], name="picks")

        this_pick = tf.placeholder(tf.int32, shape=[None,], name="this_pick")

        dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        scores, total_scores, _, _, _ = self.calc_scores(packs, picks, dropout)
        total_scores = tf.where(tf.equal(packs, 0), tf.ones_like(total_scores)*(-100000), total_scores)

        chosen_cards = tf.cast(tf.argmax(total_scores, axis=-1), tf.int32, name="chosen_cards")
        scores_pred = tf.identity(scores, name="scores_pred")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=total_scores,labels=this_pick)
        loss = tf.reduce_mean(loss, name="loss")


        optimizer = tf.train.AdamOptimizer(lr)
        self.gradients = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(self.gradients, name="train_op")
        #self.check = tf.add_check_numerics_ops()

    def build_draft_predictor(self):
        with tf.variable_scope("draft"):
            #draft_id, pack_num, player_id, card_id
            packs = tf.placeholder(tf.int32, shape=[None, 3, 8, PACK_SIZE], name="packs")
            picks = tf.placeholder(tf.int32, shape=[None, 3, 8, PACK_SIZE], name="picks")

            def pick_step(initializer, elems):
                picks = elems[0]
                packs = initializer[0]
                picked = initializer[1]
                pack_num = initializer[2]

                scores, total_scores, packs_embedded_bias, importance_scores, base_scores = self.calc_scores(packs, picked, 1.0)
                total_scores = tf.where(tf.equal(packs, 0), tf.ones_like(total_scores)*(-100000), total_scores)
                any_cards_in_pack_mask = tf.reduce_max(tf.cast(tf.not_equal(packs, 0), tf.float32), axis=-1)

                chosen_cards = tf.cast(tf.argmax(total_scores, axis=-1), tf.int32, name="chosen_cards")
                chosen_cards = tf.where(tf.equal(any_cards_in_pack_mask, 0), tf.ones_like(chosen_cards)*(-1), chosen_cards)
                chosen_cards = tf.where(tf.equal(picks, 0), chosen_cards, picks-1)

                chosen_cards_in_pack = tf.one_hot(chosen_cards, PACK_SIZE, dtype=tf.int32)
                picked_cards_id = tf.reduce_sum(packs*chosen_cards_in_pack, axis=-1)
                picked_cards_id_tiled = tf.tile(tf.expand_dims(picked_cards_id, axis=2), [1,1,3*PACK_SIZE])

                picked_mask = tf.cast(tf.not_equal(picked, 0), tf.float32, name="picked_mask")
                next_pick_pos_mask = tf.concat([tf.ones_like(picked_mask[:,:,:1]), picked_mask[:,:,:-1]], axis=-1) - picked_mask
                new_picked = tf.where(tf.equal(next_pick_pos_mask, 1), picked_cards_id_tiled, picked)

                new_packs = packs*(1-chosen_cards_in_pack)

                new_packs = tf.where(tf.equal(tf.mod(pack_num, 2), 0),
                                         tf.concat([new_packs[:,-1:], new_packs[:,:-1]], axis=1),
                                         tf.concat([new_packs[:,1:], new_packs[:,:1]], axis=1))

                return [new_packs, new_picked, pack_num, chosen_cards, importance_scores]


            def pack_step(initializer, elems):
                picked = initializer[0]
                chosen_cards = initializer[1]
                importance_scores = initializer[2]
                picks = elems[0]
                packs = elems[1]
                pack_num = elems[2]

                picks = tf.transpose(picks, [2,0,1])
                result = tf.scan(fn=pick_step,
                                 elems=[picks],
                                 initializer=[packs,
                                              picked,
                                              pack_num,
                                              chosen_cards[:,:,0],
                                              importance_scores[:,:,0]])

                new_picked = result[1][-1]
                chosen_cards = tf.transpose(result[3], [1,2,0])
                importance_scores = tf.transpose(result[4], [1,2,0,3])
                return [new_picked, chosen_cards, importance_scores]


            packs = tf.transpose(packs, [1,0,2,3])
            picks = tf.transpose(picks, [1,0,2,3])
            picked = tf.zeros_like(packs[0])
            picked = tf.tile(picked, [1,1,3])

            chosen_cards_init = tf.zeros_like(packs[0])
            importance_scores_init = tf.zeros_like(packs[0], dtype=tf.float32)
            importance_scores_init = tf.tile(tf.expand_dims(importance_scores_init, -1), [1,1,1,3*PACK_SIZE])


            result = tf.scan(fn=pack_step,
                             elems=[picks, packs, tf.range(0,3)],
                             initializer=[picked,
                                          chosen_cards_init,
                                          importance_scores_init])
            chosen_cards = tf.transpose(result[1], [1,0,2,3], name="predicted_picks")
            importance_scores = tf.transpose(result[2], [1,0,2,3,4], name="importance_scores")


    def calc_scores(self, packs, picked, dropout):
        importance_scores = self.calc_importance_scores2(picked, dropout)
        synergy_scores = self.calc_synergy_scores(packs, picked, dropout)
        card_power_scores = tf.nn.relu(tf.nn.embedding_lookup(self.card_embeddings_bias, packs, name="packs_embeddings_bias"))
        picks_mask = tf.cast(tf.not_equal(picked, 0), dtype=tf.float32, name="picks_mask")
        picks_count = tf.count_nonzero(picked, axis=-1, dtype=tf.float32)

        base_scores = synergy_scores * tf.tile(tf.expand_dims(picks_mask, axis=-2), [1]*(len(picks_mask.shape)-1)+[PACK_SIZE,1])
        scores = base_scores * tf.tile(tf.expand_dims(importance_scores, axis=-2), [1]*(len(importance_scores.shape)-1)+[PACK_SIZE,1])
        scores = scores * tf.expand_dims(card_power_scores, axis=-1)
        qq = tf.tile(tf.expand_dims(tf.equal(picks_count, 0), axis=-1), [1]*(len(picks_count.shape))+[PACK_SIZE])
        total_scores = tf.where(qq, card_power_scores, tf.reduce_sum(scores, axis=-1))
        return scores, total_scores, card_power_scores, importance_scores, base_scores

    def calc_importance_scores2(self, picked, dropout):
        picked_embedded = tf.nn.embedding_lookup(self.card_embeddings, picked, name="packed_embeddings")
        picked_embedded = tf.nn.dropout(picked_embedded, keep_prob=dropout)

        mask = tf.cast(tf.not_equal(picked, 0), dtype=tf.float32)
        expanded_mask = tf.expand_dims(mask, -1) * tf.expand_dims(mask, -2)
        x = picked_embedded
        for i in range(MODEL_SETTINGS['transformer_layers_num']):
            with tf.variable_scope("layer_{}".format(i)):
                with tf.variable_scope("self_attention"):
                    #delta = x
                    delta = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
                    delta = self.self_attention(delta, expanded_mask, MODEL_SETTINGS, dropout)
                    delta = tf.nn.dropout(delta, dropout)
                    x = x + delta
                with tf.variable_scope("feed_forward"):
                    #delta = x
                    delta = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
                    delta = self.feed_forward(delta, MODEL_SETTINGS, dropout)
                    delta = tf.nn.dropout(delta, dropout)
                    x = x + delta
        with tf.variable_scope("output_norm"):
            x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
        x = tf.layers.Dense(1, name="attention_score_transform", dtype=tf.float32)(x)
        x = tf.reduce_sum(x, axis=-1)
        x = mask*x+(1-mask)*(-1e10)
        x = tf.nn.softmax(x)
        return x

    def self_attention(self, inputs, mask, params, dropout):
        q = tf.layers.Dense(inputs.shape[-1], name="q", dtype=tf.float32)(inputs)
        k = tf.layers.Dense(inputs.shape[-1], name="k", dtype=tf.float32)(inputs)
        v = tf.layers.Dense(inputs.shape[-1], name="v", dtype=tf.float32)(inputs)

        logits = tf.matmul(q, k, transpose_b=True)

        logits = logits * tf.pow(tf.cast(inputs.shape[-1], dtype=tf.float32), -0.5)
        logits = mask*logits + (1-mask)*tf.ones_like(logits)*(-1e10)
        e_logits = tf.exp(logits - tf.reduce_max(logits, axis=-1, keepdims=True))
        att_weights = e_logits / (tf.reduce_sum(e_logits, axis=-1, keepdims=True)+1e-12)
        att_weights = tf.nn.dropout(att_weights, dropout)
        attention_output = tf.matmul(att_weights, v)
        attention_output = tf.layers.Dense(inputs.shape[-1], name="output_transform", dtype=tf.float32)(attention_output)
        return attention_output

    def feed_forward(self, x, params, dropout):
        inputs = x + tf.get_variable(name="fixup_block_bias1",shape=[1,1,1])
        x = tf.layers.Dense(params['transformer_filter_size'], name="filter_layer", dtype=tf.float32)(x)
        x = x + tf.get_variable(name="ff_block_bias2",shape=[1,1,1], initializer=tf.initializers.zeros)
        x = 0.5*x*(1+tf.tanh(sqrt(2/pi)*(x+0.044715*tf.pow(x, 3))))
        x = tf.nn.dropout(x, dropout)
        x = tf.layers.Dense(inputs.shape[-1], name="output_layer", dtype=tf.float32)(x)
        return x


    def calc_synergy_scores(self, packs, picked, dropout):
        packs_embedded = tf.nn.embedding_lookup(self.card_embeddings, packs, name="packs_embeddings")
        picked_embedded = tf.nn.embedding_lookup(self.card_embeddings, picked, name="packed_embeddings")
        packs_embedded = tf.nn.dropout(packs_embedded, keep_prob=dropout)
        picked_embedded = tf.nn.dropout(picked_embedded, keep_prob=dropout)
        query = tf.nn.bias_add(standard_ops.tensordot(packs_embedded, self.attention_vars[0], axes=[len(packs_embedded.shape)-1,0]), self.attention_vars[1])
        keys = tf.nn.bias_add(standard_ops.tensordot(picked_embedded, self.attention_vars[2], axes=[len(picked_embedded.shape)-1,0]), self.attention_vars[3])
        synergy_scores = tf.sigmoid(tf.matmul(query, keys, transpose_b=True), name="scores")
        return synergy_scores


    def init_no_weights(self, card_count):
        self.card_embeddings = tf.get_variable(name="card_embeddings_var",
                                               dtype=tf.float32,
                                               initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
                                               shape=[card_count, MODEL_SETTINGS['emb_dim']])
        self.card_embeddings_bias = tf.get_variable(name="card_embeddings_var_bias",
                                                    dtype=tf.float32,
                                                    initializer=tf.initializers.random_uniform(minval=0.1, maxval=10),
                                                    shape=[card_count])

        self.attention_vars = []
        for i in range(4):
            w = tf.get_variable(dtype=tf.float32,
                                shape=[MODEL_SETTINGS['emb_dim'], MODEL_SETTINGS['emb_dim']],
                                initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
                                name="attention{}_w".format(i))
            b = tf.get_variable(dtype=tf.float32,
                                shape=[MODEL_SETTINGS['emb_dim']],
                                initializer=tf.initializers.zeros,
                                name="attention{}_b".format(i))
            self.attention_vars.append(w)
            self.attention_vars.append(b)


    def init_weights(self, weights):
        self.card_embeddings = tf.get_variable(name="card_embeddings_var", initializer=weights[0])
        self.card_embeddings_bias = tf.get_variable(name="card_embeddings_var_bias", initializer=weights[1])
        self.attention_vars = []
        for i in range(4):
            w = tf.get_variable(name="attention{}_w".format(i), initializer=weights[2*i+2])
            b = tf.get_variable(name="attention{}_b".format(i), initializer=weights[2*i+3])
            self.attention_vars.append(w)
            self.attention_vars.append(b)


    def build(self):
        self.build_trainer()
        self.build_draft_predictor()

    def get_weights(self):
        return self.sess.run(self.sess.graph._collections['variables'])

    def get_feed_dict_predict(self, packs, picks):
        feed = {
            self.sess.graph.get_tensor_by_name("packs:0"): packs,
            self.sess.graph.get_tensor_by_name("picks:0"): picks,
            self.sess.graph.get_tensor_by_name("dropout:0"): 1.0
        }
        return feed

    def get_feed_dict_train(self, packs, picks, Y):
        feed = self.get_feed_dict_predict(packs, picks)
        feed[self.sess.graph.get_tensor_by_name("dropout:0")] = MODEL_SETTINGS['dropout']
        feed[self.sess.graph.get_tensor_by_name("lr:0")] = MODEL_SETTINGS['lr']
        feed[self.sess.graph.get_tensor_by_name("this_pick:0")] = Y
        return feed

    def train(self, packs, picks, Y):
        fd = self.get_feed_dict_train(packs, picks, Y)
        _, train_loss, scores, Y_pred = self.sess.run([self.sess.graph.get_operation_by_name("train_op"),
                                                                         self.sess.graph.get_tensor_by_name("loss:0"),
                                                                      self.sess.graph.get_tensor_by_name("scores_pred:0"),
                                                                      self.sess.graph.get_tensor_by_name("chosen_cards:0")],
                                                      feed_dict=fd)
        accuracy = np.average(Y==Y_pred)
        return train_loss, accuracy

    def evaluate(self, packs, picks, Y):
        fd = self.get_feed_dict_train(packs, picks, Y)
        fd[self.sess.graph.get_tensor_by_name("dropout:0")]=1.0
        test_loss, Y_pred = self.sess.run([self.sess.graph.get_tensor_by_name("loss:0"),
                                      self.sess.graph.get_tensor_by_name("chosen_cards:0")], feed_dict=fd)
        accuracy = np.average(Y==Y_pred)

        return test_loss, accuracy

    def predict(self, packs, picks):
        fd = self.get_feed_dict_predict(packs, picks)
        Y_pred, scores = self.sess.run([self.sess.graph.get_tensor_by_name("scores_pred:0"),
                                   self.sess.graph.get_tensor_by_name("scores:0")], feed_dict=fd)
        return Y_pred, scores


    def predict_draft(self, packs, picks):
        feed_dict = {
            self.sess.graph.get_tensor_by_name("draft/packs:0"): packs,
            self.sess.graph.get_tensor_by_name("draft/picks:0"): picks
        }
        Y_pred, importance_scores = self.sess.run([self.sess.graph.get_tensor_by_name("draft/predicted_picks:0"),
                                                   self.sess.graph.get_tensor_by_name("draft/importance_scores:0")], feed_dict=feed_dict)
        return Y_pred, importance_scores
