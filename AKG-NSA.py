# @Author: Juan Du
# @Time: 2024/10/02 11:22

import os
import tensorflow as tf
import numpy as np
import random
from AKG_Config import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 2025
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
param = ParamConfig()

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def get_inputs():
    user = tf.placeholder(dtype=tf.int32, shape=[None, ], name='user')
    items = tf.placeholder(dtype=tf.int32, shape=[None, None], name='items')
    length = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='length')
    target = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')
    return user, items, length, target, learning_rate, dropout_rate

def loss_calculation(target, pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=pred)
    loss_mean = tf.reduce_mean(loss, name='loss_mean')
    return loss_mean

def optimizer(loss, learning_rate):
    basic_op = tf.train.AdamOptimizer(learning_rate)
    gradients = basic_op.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    model_op = basic_op.apply_gradients(capped_gradients)
    return model_op

class AKG_NSA:
    def __init__(self, num_items, num_users, adj_matrix_A, adj_matrix_B):
        os.environ['CUDA_VISIBLE_DEVICES'] = param.gpu_index
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.num_items = num_items
        self.num_users = num_users
        
        # Graph parameters
        self.adj_A = adj_matrix_A
        self.adj_B = adj_matrix_B
        self.num_layers = param.num_layers
        self.embed_size = param.embedding_size
        self.block_size = param.block_size
        
        # NSA parameters
        self.num_heads = param.num_heads
        self.comp_ratio = param.comp_ratio
        self.temperature = param.temperature
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.user, self.items, self.length, self.target, self.lr, self.dropout_rate = get_inputs()

            with tf.name_scope('PGCN'):
                self.weights = self._init_weights()
                self.E_A, self.E_U, self.E_I = self._parallel_gcn()

            with tf.name_scope('MNSA'):
                self.E_A_sel = self._archive_nsa_channel(self.E_A)
                self.E_U_sel = self._user_nsa_channel(self.E_U)
                self.E_I_sel = self._item_nsa_channel(self.E_I)

            with tf.name_scope('prediction'):
                self.pred = self._final_prediction()

            with tf.name_scope('loss'):
                self.loss = loss_calculation(self.target, self.pred)
                self.reg_loss = self.loss + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            with tf.name_scope('optimizer'):
                self.train_op = optimizer(self.reg_loss, self.lr)

    def _init_weights(self):
        initializer = tf.contrib.layers.xavier_initializer()
        weights = {
            # PGCN weights
            'W_A': tf.get_variable('W_A', [self.embed_size, self.embed_size], initializer=initializer),
            'W_B': tf.get_variable('W_B', [self.embed_size, self.embed_size], initializer=initializer),
            
            # NSA weights
            'W_comp': tf.get_variable('W_comp', [self.block_size, self.embed_size], initializer=initializer),
            'W_sel': tf.get_variable('W_sel', [self.embed_size, self.embed_size], initializer=initializer),
            
            # Final projection
            'W_f': tf.get_variable('W_f', [3*self.embed_size, self.num_items], initializer=initializer),
            'b_f': tf.get_variable('b_f', [self.num_items], initializer=tf.zeros_initializer())
        }
        return weights

    def _parallel_gcn(self):
        # Graph convolution for archive knowledge graph
        E_A = tf.get_variable("E_A", initializer=tf.random_normal([self.adj_A.shape[0], self.embed_size]))
        norm_adj_A = self._normalize_adj(self.adj_A)
        
        # Graph convolution for user-item graph
        E_U = tf.get_variable("E_U", initializer=tf.random_normal([self.num_users, self.embed_size]))
        E_I = tf.get_variable("E_I", initializer=tf.random_normal([self.num_items, self.embed_size]))
        norm_adj_B = self._normalize_adj(self.adj_B)

        all_embeddings_A = [E_A]
        all_embeddings_B = [tf.concat([E_U, E_I], 0)]
        
        for k in range(self.num_layers):
            # Archive graph convolution
            E_A = tf.sparse_tensor_dense_matmul(norm_adj_A, E_A)
            E_A = tf.matmul(E_A, self.weights['W_A'])
            E_A = tf.nn.leaky_relu(E_A)
            all_embeddings_A.append(E_A)
            
            # User-item graph convolution
            embeddings_B = tf.sparse_tensor_dense_matmul(norm_adj_B, all_embeddings_B[-1])
            embeddings_B = tf.matmul(embeddings_B, self.weights['W_B'])
            embeddings_B = tf.nn.leaky_relu(embeddings_B)
            all_embeddings_B.append(embeddings_B)

        # Layer aggregation
        E_A = tf.reduce_mean(tf.stack(all_embeddings_A), axis=0)
        E_B = tf.reduce_mean(tf.stack(all_embeddings_B), axis=0)
        E_U, E_I = tf.split(E_B, [self.num_users, self.num_items], 0)
        
        return E_A, E_U, E_I

    def _normalize_adj(self, adj):
        adj = adj + tf.eye(adj.shape[0])
        rowsum = tf.reduce_sum(adj, 1)
        d_inv_sqrt = tf.pow(rowsum, -0.5)
        d_mat_inv_sqrt = tf.diag(d_inv_sqrt)
        return tf.sparse.matmul(tf.sparse.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def _split_into_blocks(self, embeddings):
        batch_size = tf.shape(embeddings)[0]
        num_blocks = tf.cast(tf.ceil(tf.shape(embeddings)[#citation-1](citation-1) / self.block_size), tf.int32)
        paddings = [[0,0], [0, num_blocks*self.block_size - tf.shape(embeddings)[#citation-1](citation-1)], [0,0]]
        padded = tf.pad(embeddings, paddings)
        blocks = tf.reshape(padded, [batch_size, num_blocks, self.block_size, self.embed_size])
        return blocks

    def _compressed_attention(self, queries, keys, values):
        # Compute similarity scores
        scale = tf.sqrt(tf.cast(tf.shape(keys)[-1], tf.float32))
        scores = tf.matmul(queries, keys, transpose_b=True) / scale
        
        # Compute attention weights
        attn_weights = tf.nn.softmax(scores / self.temperature)
        
        # Compress attention
        compressed = tf.matmul(attn_weights, values)
        return compressed

    def _selected_attention(self, queries, keys, values, comp_ratio):
        # Compute importance scores
        scores = tf.reduce_sum(queries * keys, axis=-1)
        topk = tf.cast(comp_ratio * tf.cast(tf.shape(keys)[#citation-1](citation-1), tf.float32), tf.int32)
        
        # Select top-k blocks
        _, indices = tf.nn.top_k(scores, k=topk)
        selected_keys = tf.gather(keys, indices, batch_dims=1)
        selected_values = tf.gather(values, indices, batch_dims=1)
        
        # Compute attention
        scale = tf.sqrt(tf.cast(tf.shape(selected_keys)[-1], tf.float32))
        scores = tf.matmul(queries, selected_keys, transpose_b=True) / scale
        attn_weights = tf.nn.softmax(scores)
        return tf.matmul(attn_weights, selected_values)

    def _archive_nsa_channel(self, E_A):
        with tf.variable_scope("archive_nsa"):
            # Split into blocks
            blocks = self._split_into_blocks(E_A)
            
            # Compressed attention
            compressed = self._compressed_attention(
                tf.expand_dims(self.E_U, 1),  # User embeddings as query
                tf.tile(tf.expand_dims(blocks, 0), [tf.shape(self.E_U)[0],1,1,1]),
                tf.tile(tf.expand_dims(blocks, 0), [tf.shape(self.E_U)[0],1,1,1])
            )
            
            # Selected attention
            selected = self._selected_attention(
                self.E_U,
                compressed,
                compressed,
                self.comp_ratio
            )
            return selected

    def _user_nsa_channel(self, E_U):
        with tf.variable_scope("user_nsa"):
            blocks = self._split_into_blocks(E_U)
            compressed = self._compressed_attention(
                tf.expand_dims(E_U, 1),
                blocks,
                blocks
            )
            selected = self._selected_attention(
                E_U,
                compressed,
                compressed,
                self.comp_ratio
            )
            return tf.nn.dropout(selected, keep_prob=1-self.dropout_rate)

    def _item_nsa_channel(self, E_I):
        with tf.variable_scope("item_nsa"):
            blocks = self._split_into_blocks(E_I)
            compressed = self._compressed_attention(
                tf.expand_dims(E_I, 1),
                blocks,
                blocks
            )
            selected = self._selected_attention(
                E_I,
                compressed,
                compressed,
                self.comp_ratio
            )
            return tf.nn.dropout(selected, keep_prob=1-self.dropout_rate)

    def _final_prediction(self):
        # Concatenate multi-channel representations
        combined = tf.concat([
            tf.nn.embedding_lookup(self.E_A_sel, self.user),
            tf.nn.embedding_lookup(self.E_U_sel, self.user),
            tf.reduce_mean(tf.nn.embedding_lookup(self.E_I_sel, self.items), axis=1)
        ], axis=1)

        # Final projection
        logits = tf.matmul(combined, self.weights['W_f']) + self.weights['b_f']
        return logits

    def train(self, sess, user, items, length, target, lr, dropout):
        feed_dict = {
            self.user: user,
            self.items: items,
            self.length: length,
            self.target: target,
            self.lr: lr,
            self.dropout_rate: dropout
        }
        return sess.run([self.loss, self.train_op], feed_dict)

    def predict(self, sess, user, items, length):
        feed_dict = {
            self.user: user,
            self.items: items,
            self.length: length,
            self.dropout_rate: 0.0
        }
        return sess.run(self.pred, feed_dict)
