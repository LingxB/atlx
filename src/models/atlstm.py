import tensorflow as tf
from tensorflow.contrib import rnn
from src.models.base_model import BaseModel
from src.utils import Logger, __fn__, seq_length, matmul_2_3
from copy import deepcopy


logger = Logger(__fn__())


class ATLSTM(BaseModel):

    NAME = 'ATLSTM'

    def __init__(self, datamanager=None, parameters=None):
        super().__init__(datamanager=datamanager, parameters=parameters)


    def build_graph(self):

        graph = tf.Graph()

        with graph.as_default():
            # Input place holders
            # -------------------
            X = tf.placeholder(tf.int32, shape=(None, None), name='X')
            asp = tf.placeholder(tf.int32, shape=(None, 1), name='asp')
            y = tf.placeholder(tf.int32, shape=(None, 3), name='y')
            lx = tf.placeholder(tf.float32, shape=(None, None, None), name='lx')
            dropout_keep = tf.placeholder_with_default(1.0, shape=(), name='dropout_keep')

            # Initializer
            # -----------
            initializer = self._get_initializer()

            # Embedding
            # ---------
            with tf.name_scope('Embedding'):
                glove = tf.get_variable(name='glove',
                                        shape=self.dm.emb.shape,
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(self.dm.emb.values),
                                        )
                pad = tf.get_variable(name='pad',
                                      shape=(1, glove.shape[1]),
                                      dtype=tf.float32,
                                      initializer=tf.zeros_initializer(),
                                      trainable=False)
                unk = tf.get_variable(name='unk',
                                      shape=(1, glove.shape[1]),
                                      dtype=tf.float32,
                                      initializer=initializer,
                                      )
                embedding = tf.concat([pad, unk, glove], axis=0, name='embedding')

                X_ = tf.nn.embedding_lookup(embedding, X)  # (batch, N, d)
                seq_len = seq_length(X_)
                asp_ = tf.nn.embedding_lookup(embedding, asp)  # (batch, 1, da)

            # Encoder
            # -------
            with tf.name_scope('Encoder'):
                # cell = rnn.BasicLSTMCell(hyparams['cell_num'])
                cell = rnn.LSTMCell(self.p['cell_num'], initializer=initializer)
                cells = [deepcopy(cell) for i in range(self.p['layer_num'])]
                cell = rnn.MultiRNNCell(cells)

                H, (s,) = tf.nn.dynamic_rnn(cell=cell, inputs=X_, sequence_length=seq_len,
                                            dtype=tf.float32)  # (batch, N, d)
                hN = s.h  # (batch, d)
                assert H.shape.as_list() == tf.TensorShape([X_.shape[0], X_.shape[1], self.p['cell_num']]).as_list()

            # Attention
            # ---------
            with tf.name_scope('Attention'):
                H_T = tf.transpose(H, [0, 2, 1])  # (batch, d, N)
                Wh = tf.get_variable('Wh', shape=(self.p['cell_num'], self.p['cell_num']), dtype=tf.float32, initializer=initializer)  # (d, d)
                WhH = matmul_2_3(Wh, H_T)  # (batch, d, N)
                assert WhH.shape.as_list() == H_T.shape.as_list()

                VaeN = tf.tile(asp_, [1, tf.shape(X_)[1], 1])  # (batch, N, da), da==d in this setting
                assert VaeN.shape.as_list() == X_.shape.as_list()
                VaeN_T = tf.transpose(VaeN, [0, 2, 1])  # (batch, da, N)
                Wv = tf.get_variable('Wv', shape=(asp_.shape[2], asp_.shape[2]), dtype=tf.float32, initializer=initializer)  # (da, da)
                WvVaeN = matmul_2_3(Wv, VaeN_T)  # (batch, da, N)
                assert WvVaeN.shape.as_list() == VaeN_T.shape.as_list()

                M = tf.tanh(tf.concat([WhH, WvVaeN], axis=1))  # (batch, d+da, N)

                w = tf.get_variable('w', shape=(X_.shape[2] + asp_.shape[2], 1), dtype=tf.float32, initializer=initializer)  # (d+da, 1)
                w_T = tf.transpose(w)  # (1, d+da)
                alpha = tf.nn.softmax(matmul_2_3(w_T, M), name='ALPHA')  # (batch, 1, N)
                assert alpha.shape.as_list() == tf.TensorShape([X_.shape[0], 1, M.shape[2]]).as_list()
                # alpha = tf.reshape(alpha, (tf.shape(alpha)[0], tf.shape(alpha)[2])) # (batch, N)

                _r = tf.matmul(alpha, H)  # (batch, 1, d)
                r = tf.squeeze(_r, 1)  # (batch, d)
                assert r.shape.as_list() == tf.TensorShape([X_.shape[0], H.shape[2]]).as_list()

                Wp = tf.get_variable('Wp', shape=(r.shape[1], r.shape[1]), dtype=tf.float32, initializer=initializer)  # (d, d)
                Wx = tf.get_variable('Wx', shape=(hN.shape[1], hN.shape[1]), dtype=tf.float32, initializer=initializer)  # (d, d)
                h_star = tf.tanh(tf.matmul(r, Wp) + tf.matmul(hN, Wx))
                h_star = tf.nn.dropout(h_star, dropout_keep, seed=self.p['seed']+40)  # 0.5 dropout on h_star was found in author's code
                assert h_star.shape.as_list() == tf.TensorShape([H.shape[0], H.shape[2]]).as_list()

            # Output Layer
            # ------------
            with tf.name_scope('Output'):
                logits = tf.layers.dense(h_star, 3, kernel_initializer=initializer, name='s')
                pred = tf.nn.softmax(logits, name='PRED')

            # Loss
            # ----
            with tf.name_scope('Loss'):
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
                reg_params = [p for p in tf.trainable_variables() if p.name not in {'glove:0', 'unk:0'}]
                #regularizer = tf.multiply(self.p['lambda'], tf.add_n([tf.nn.l2_loss(p) for p in reg_params]), name='REGL')
                regularizer = tf.divide(self.p['lambda']*tf.add_n([tf.nn.l2_loss(p) for p in reg_params]),
                                        tf.to_float(tf.shape(X_)[0]), name='REGL')
                loss = tf.add(cross_entropy, regularizer, name='LOSS')

            # Train Op
            # --------
            with tf.name_scope('TrainOp'):
                optimizer = self._get_optimizer()
                train_op = optimizer.minimize(loss, name='TRAIN_OP')

            # Evaluation on the fly
            # ---------------------
            with tf.name_scope('Evaluation'):
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='ACC3')

        return graph