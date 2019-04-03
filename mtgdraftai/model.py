import tensorflow as tf
import numpy as np
from math import pi, sqrt
from tensorflow.contrib.opt.python.training.weight_decay_optimizers import AdamWOptimizer

EPS = 1e-12

class Model(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config

    def build_trainer(self):
        pack = tf.placeholder(tf.int32, shape=[None, None], name="packs")
        picks = tf.placeholder(tf.int32, shape=[None, None], name="picks")

        dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        wd = tf.placeholder(dtype=tf.float32, shape=[], name="wd")

        final_embeddings = self.transformer(pack, picks, dropout)
        final_embeddings = final_embeddings[:,:tf.shape(pack)[1]]
        total_scores = tf.layers.Dense(1, name="q")(final_embeddings)
        total_scores = tf.reduce_sum(total_scores, axis=-1)
        total_scores = tf.where(tf.equal(pack, 0), tf.ones_like(total_scores)*(-1e10), total_scores)

        chosen_cards = tf.cast(tf.argmax(total_scores, axis=-1), tf.int32, name="chosen_cards")
        scores_pred = tf.identity(total_scores, name="scores_pred")

        this_pick = tf.placeholder(tf.int32, shape=[None], name="choice")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=total_scores,labels=this_pick)
        loss = tf.reduce_mean(loss, name="loss")


        optimizer = AdamWOptimizer(weight_decay=wd, learning_rate=lr)
        self.gradients = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(self.gradients, name="train_op")
        #self.check = tf.add_check_numerics_ops()

    def transformer(self, pack, picks, dropout):
        embeddings_var = tf.get_variable(name="embeddings_var",
                                               dtype=tf.float32,
                                               initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
                                               shape=[self.config['cards_count'], self.config['emb_dim']])
        pack_var = tf.get_variable(name="pack_var",
                                   dtype=tf.float32,
                                   initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
                                   shape=[1, 1, self.config['emb_dim']])
        picks_var = tf.get_variable(name="picks_var",
                                   dtype=tf.float32,
                                   initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
                                   shape=[1, 1, self.config['emb_dim']])

        mask = tf.cast(tf.equal(tf.concat([pack, picks], axis=-1), 0), dtype=tf.float32)
        pack_embedded = tf.nn.embedding_lookup(embeddings_var, pack, name="embeddings") + pack_var
        picks_embedded = tf.nn.embedding_lookup(embeddings_var, picks, name="embeddings") + picks_var
        x = tf.concat([pack_embedded, picks_embedded], axis=-2)
        for i in range(self.config['transformer_layers_num']):
            with tf.variable_scope("layer_{}".format(i)):
                with tf.variable_scope("self_attention"):
                    delta = tf.contrib.layers.layer_norm(x)
                    delta = self.attention(delta, delta, mask, dropout)
                    delta = tf.nn.dropout(delta, dropout)
                    x = x + delta
                with tf.variable_scope("feed_forward"):
                    delta = tf.contrib.layers.layer_norm(x)
                    delta = self.feed_forward(delta, dropout)
                    delta = tf.nn.dropout(delta, dropout)
                    x = x + delta
        with tf.variable_scope("output_norm"):
            x = tf.contrib.layers.layer_norm(x)
        return x


    def attention(self, query, memory, mask, dropout):
        depth = self.config['emb_dim'] // self.config['transformer_num_heads']

        q = tf.layers.Dense(self.config['emb_dim'], name="q")(query)
        k = tf.layers.Dense(self.config['emb_dim'], name="k")(memory)
        v = tf.layers.Dense(self.config['emb_dim'], name="v")(memory)

        q = tf.reshape(q, [tf.shape(q)[0], tf.shape(q)[1], self.config['transformer_num_heads'], depth])
        k = tf.reshape(k, [tf.shape(k)[0], tf.shape(k)[1], self.config['transformer_num_heads'], depth])
        v = tf.reshape(v, [tf.shape(v)[0], tf.shape(v)[1], self.config['transformer_num_heads'], depth])

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        logits = tf.matmul(q, k, transpose_b=True)

        logits = logits * (depth ** -0.5)

        e_logits = tf.exp(logits - tf.reduce_max(logits, axis=-1, keepdims=True))
        e_logits = e_logits * tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)
        att_weights = e_logits / (tf.reduce_sum(e_logits, axis=-1, keepdims=True)+EPS)
        att_weights = tf.nn.dropout(att_weights, dropout)

        attention_output = tf.matmul(att_weights, v)
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, [tf.shape(attention_output)[0], tf.shape(attention_output)[1], self.config['emb_dim']])

        attention_output = tf.layers.Dense(self.config['emb_dim'], name="output_transform")(attention_output)
        return attention_output

    def feed_forward(self, x, dropout):
        x = tf.layers.Dense(self.config['transformer_filter_size'], name="filter_layer")(x)
        x = 0.5*x*(1+tf.tanh(sqrt(2/pi)*(x+0.044715*tf.pow(x, 3))))
        x = tf.nn.dropout(x, dropout)
        x = tf.layers.Dense(self.config['emb_dim'], name="output_layer")(x)
        return x


    def build(self):
        self.build_trainer()

    def get_weights(self):
        return self.sess.run(self.sess.graph._collections['variables'])

    def set_weights(self, weights):
        ops = []
        feed = {}
        for idx, variable in enumerate(self.sess.graph._collections['variables']):
            ops.append(variable._initializer_op)
            feed[variable._initializer_op.inputs[1]] = weights[idx]
        self.sess.run(ops, feed_dict=feed)

    def train(self, packs, picks, choice, dropout, lr, wd):
        feed_dict = {self.sess.graph.get_tensor_by_name("packs:0"): packs,
                     self.sess.graph.get_tensor_by_name("picks:0"): picks,
                     self.sess.graph.get_tensor_by_name("choice:0"): choice,
                     self.sess.graph.get_tensor_by_name("dropout:0"): dropout,
                     self.sess.graph.get_tensor_by_name("lr:0"): lr,
                     self.sess.graph.get_tensor_by_name("wd:0"): wd}

        _, train_loss, scores, choice_pred = self.sess.run([self.sess.graph.get_operation_by_name("train_op"),
                                                            self.sess.graph.get_tensor_by_name("loss:0"),
                                                            self.sess.graph.get_tensor_by_name("scores_pred:0"),
                                                            self.sess.graph.get_tensor_by_name("chosen_cards:0")],
                                                           feed_dict=feed_dict)
        accuracy = np.average(choice==choice_pred)
        return train_loss, accuracy

    def evaluate(self, packs, picks, choice):
        feed_dict = {self.sess.graph.get_tensor_by_name("packs:0"): packs,
                     self.sess.graph.get_tensor_by_name("picks:0"): picks,
                     self.sess.graph.get_tensor_by_name("choice:0"): choice,
                     self.sess.graph.get_tensor_by_name("dropout:0"): 1.0}
        test_loss, choice_pred = self.sess.run([self.sess.graph.get_tensor_by_name("loss:0"),
                                                self.sess.graph.get_tensor_by_name("chosen_cards:0")],
                                               feed_dict=feed_dict)
        accuracy = np.average(choice==choice_pred)

        return test_loss, accuracy