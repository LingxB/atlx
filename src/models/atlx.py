import tensorflow as tf
from tensorflow.contrib import rnn
from src.models.base_model import BaseModel
from src.utils import Logger, __fn__, seq_length, matmul_2_3, var_by_len, ent_by_len
from copy import deepcopy


logger = Logger(__fn__())


class ATLX(BaseModel):

    NAME = 'ATLX'

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
            lx = tf.placeholder(tf.float32, shape=(None, None, 3), name='lx') # (batch, N, dl)
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
                if self.p.get('concat_emb_lx', False):
                    X_ = tf.concat([X_, lx], axis=-1)
                    assert X_.shape.as_list()[-1] == lx.shape.as_list()[-1] + self.dm.emb.shape[1]
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

            # Prepare Lexicon input
            # ---------------------
            with tf.name_scope('Lexicon'):
                assert X_.shape.as_list()[:2] == lx.shape.as_list()[:2] # lx.shape = (batch, N, dl)

                Wlx = tf.get_variable('Wlx', shape=(self.p['lx_dim'], lx.shape[2]), dtype=tf.float32, initializer=initializer)  # (dx, dl)
                lx_T = tf.transpose(lx, [0, 2, 1])  # (batch, dl, N)
                L = tf.transpose(matmul_2_3(Wlx, lx_T), [0, 2, 1])  # (batch, N, dx)
                if self.p.get('lx_activation'):
                    L = tf.tanh(L)

            # Attention
            # ---------
            with tf.name_scope('Attention'):
                H_T = tf.transpose(H, [0, 2, 1])  # (batch, d, N)
                Wh = tf.get_variable('Wh', shape=(self.p['cell_num'], self.p['cell_num']), dtype=tf.float32, initializer=initializer)  # (d, d)
                WhH = matmul_2_3(Wh, H_T)  # (batch, d, N)
                assert WhH.shape.as_list() == H_T.shape.as_list()

                VaeN = tf.tile(asp_, [1, tf.shape(X_)[1], 1])  # (batch, N, da), da==d in this setting
                assert VaeN.shape.as_list()[:-1] == X_.shape.as_list()[:-1]
                VaeN_T = tf.transpose(VaeN, [0, 2, 1])  # (batch, da, N)
                Wv = tf.get_variable('Wv', shape=(asp_.shape[2], asp_.shape[2]), dtype=tf.float32, initializer=initializer)  # (da, da)
                WvVaeN = matmul_2_3(Wv, VaeN_T)  # (batch, da, N)
                assert WvVaeN.shape.as_list() == VaeN_T.shape.as_list()

                attention_with_lx = self.p.get('attention_with_lx')
                if attention_with_lx:
                    L_T = tf.transpose(L, [0, 2, 1]) # (batch, dx, N)
                    WL = tf.get_variable('WL', shape=(L.shape[2], L.shape[2]), dtype=tf.float32, initializer=initializer) # (dx, dx))
                    WLL = matmul_2_3(WL, L_T) # (batch, dx, N)

                    M = tf.tanh(tf.concat([WhH, WvVaeN, WLL], axis=1)) # (batch, d+da+dx, N)
                    w = tf.get_variable('w', shape=(H.shape[2] + asp_.shape[2] + L.shape[2], 1), dtype=tf.float32, initializer=initializer)  # (d+da+dx, 1)
                else:
                    M = tf.tanh(tf.concat([WhH, WvVaeN], axis=1))  # (batch, d+da, N)
                    w = tf.get_variable('w', shape=(H.shape[2] + asp_.shape[2], 1), dtype=tf.float32, initializer=initializer)  # (d+da, 1)

                w_T = tf.transpose(w)  # (1, d+da/d+da+dx)
                alpha = tf.nn.softmax(matmul_2_3(w_T, M), name='ALPHA')  # (batch, 1, N)
                assert alpha.shape.as_list() == tf.TensorShape([X_.shape[0], 1, M.shape[2]]).as_list()
                # alpha = tf.reshape(alpha, (tf.shape(alpha)[0], tf.shape(alpha)[2])) # (batch, N)

                _r = tf.matmul(alpha, H)  # (batch, 1, d)
                r = tf.squeeze(_r, 1)  # (batch, d)
                assert r.shape.as_list() == tf.TensorShape([X_.shape[0], H.shape[2]]).as_list()

            # Lexicon
            # -------
            with tf.name_scope('Lexicon'):
                lx_mode = self.p.get('lx_mode')
                # if lx_mode == 'linear':
                #     Wli = tf.get_variable('Wli', shape=(X_.shape[1], 1), dtype=tf.float32, initializer=initializer) # (N, 1)
                #     Wli_T = tf.transpose(Wli) # (1, N)
                #     l = tf.squeeze(matmul_2_3(Wli_T, L), 1) # (batch, d)
                if lx_mode == 'att':
                    l = tf.squeeze(tf.matmul(alpha, L), 1) # (batch, dx)
                elif lx_mode == None:
                    pass
                # elif lx_mode == 'conv':
                #     pass
                else:
                    raise NotImplementedError

            # Merge all as h*
            # ---------------
            with tf.name_scope('Hstar'):
                Wp = tf.get_variable('Wp', shape=(r.shape[1], r.shape[1]), dtype=tf.float32, initializer=initializer)  # (d, d)
                Wx = tf.get_variable('Wx', shape=(hN.shape[1], hN.shape[1]), dtype=tf.float32, initializer=initializer)  # (d, d)
                if lx_mode is not None:
                    Wl = tf.get_variable('Wl', shape=(l.shape[1], l.shape[1]), dtype=tf.float32, initializer=initializer) # (d, d)

                merge_mode = self.p.get('merge_mode')

                if merge_mode == 'add':
                    h_star = tf.tanh(tf.matmul(r, Wp) + tf.matmul(hN, Wx) + tf.matmul(l, Wl)) # (batch, d)
                elif merge_mode == 'concat':
                    h_star = tf.tanh(tf.concat([tf.matmul(r, Wp), tf.matmul(hN, Wx), tf.matmul(l, Wl)], axis=1)) # (batch, 3d)
                elif merge_mode == 'att':
                    # H_star = tf.stack([tf.matmul(r, Wp) + tf.matmul(hN, Wx), tf.matmul(l, Wl)], axis=2) # (batch, d, 2)
                    H_star = tf.tanh(tf.stack([tf.matmul(r, Wp) + tf.matmul(hN, Wx), tf.matmul(l, Wl)], axis=2)) # (batch, d, 2) exp_5.1
                    wb = tf.get_variable('wb', shape=(H_star.shape[1], 1), dtype=tf.float32, initializer=initializer) # (d, 1)
                    wb_T = tf.transpose(wb) # (1, d)
                    beta = tf.nn.softmax(matmul_2_3(wb_T, H_star), name='BETA') # (batch, 1, 2)
                    H_star_T = tf.transpose(H_star, [0, 2, 1]) # (batch, 2, d)
                    h_star = tf.squeeze(tf.matmul(beta, H_star_T), 1) # (batch, d)
                elif merge_mode == None:
                    h_star = tf.tanh(tf.matmul(r, Wp) + tf.matmul(hN, Wx))
                else:
                    raise NotImplementedError

                h_star = tf.nn.dropout(h_star, dropout_keep, seed=self.p['seed']+40)
                #assert h_star.shape.as_list() == tf.TensorShape([H.shape[0], H.shape[2]]).as_list()

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
                if self.p.get('att_reg'):
                    epsilon = self.p.get('epsilon')
                    # compute excluding padding positions
                    condition = tf.sequence_mask(seq_len, tf.reduce_max(seq_len))
                    a_true = tf.boolean_mask(tf.squeeze(alpha, 1), condition)
                    mean, var = tf.nn.moments(a_true, [0])

                    # mean, var = tf.nn.moments(tf.squeeze(alpha, 1), [0,1]) # compute on all alpha including padding positions

                    std = tf.sqrt(var)
                    att_regularizer = tf.divide(epsilon * std, tf.to_float(tf.shape(X_)[0]), name='REGLATT')
                    loss = tf.add_n([cross_entropy, regularizer, att_regularizer], name='LOSS')
                elif self.p.get('ent_reg'):
                    epsilon = self.p.get('epsilon')
                    condition = tf.sequence_mask(seq_len, tf.reduce_max(seq_len))
                    a_true = tf.boolean_mask(tf.squeeze(alpha, 1), condition)
                    ent = - tf.reduce_sum(a_true * tf.log(a_true))
                    ent_reg = -1 * epsilon * ent / tf.to_float(tf.shape(X_)[0])
                    loss = tf.add_n([cross_entropy, regularizer, ent_reg], name='LOSS')
                elif self.p.get('att_std_mask'):
                    epsilon = self.p.get('epsilon')
                    a = tf.squeeze(alpha, 1)
                    batch_var = var_by_len(a, tf.reshape(seq_len, [-1,1]))
                    att_std_mask = epsilon * tf.reduce_mean(tf.sqrt(batch_var))
                    loss = tf.add_n([cross_entropy, regularizer, att_std_mask], name='LOSS')
                elif self.p.get('att_ent_mask'):
                    epsilon = self.p.get('epsilon')
                    a = tf.squeeze(alpha, 1)
                    batch_ent = ent_by_len(a, tf.reshape(seq_len, [-1,1]))
                    att_ent_mask = -1 * epsilon * tf.reduce_mean(batch_ent)
                    loss = tf.add_n([cross_entropy, regularizer, att_ent_mask], name='LOSS')
                elif self.p.get('att_enh'):
                    # TODO: Review
                    gamma = self.p.get('gamma')
                    condition = tf.not_equal(tf.reduce_sum(lx, axis=-1), 0) # lx (batch, N, dl)
                    a_true = tf.boolean_mask(tf.squeeze(alpha, 1), condition)
                    att_enh = -1 * gamma * tf.reduce_mean(a_true)
                    loss = tf.add_n([cross_entropy, regularizer, att_enh], name='LOSS')
                elif self.p.get('att_std'):
                    epsilon = self.p.get('epsilon')
                    _, var = tf.nn.moments(tf.squeeze(alpha, 1), axes=1)
                    att_std = epsilon * tf.reduce_mean(tf.sqrt(var))
                    loss = tf.add_n([cross_entropy, regularizer, att_std], name='LOSS')
                elif self.p.get('att_lmax'):
                    # + max
                    epsilon = self.p.get('epsilon')
                    _alpha = tf.squeeze(alpha, 1)
                    lmax = epsilon * tf.reduce_mean(tf.reduce_max(_alpha, axis=1))
                    loss = tf.add_n([cross_entropy, regularizer, lmax], name='LOSS')
                elif self.p.get('att_ent'):
                    # - entropy want it to be bigger, more suprise, less sharp
                    epsilon = self.p.get('epsilon')
                    _alpha = tf.squeeze(alpha, 1)
                    ent = -1 * epsilon * tf.reduce_mean(- tf.reduce_sum(_alpha * tf.log(_alpha), axis=1))
                    loss = tf.add_n([cross_entropy, regularizer, ent], name='LOSS')
                else:
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