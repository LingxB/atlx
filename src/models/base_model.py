import abc
import tensorflow as tf
from src.utils import Logger, __fn__, mkdir, filter_params, pickle_dump, pickle_load
import numpy as np
import sys

logger = Logger(__fn__())


class BaseModel(object, metaclass=abc.ABCMeta):

    NAME = 'BaseModel'

    TENSORS = dict(
        loss='Loss/LOSS',
        regularizer='Loss/REGL',
        acc3='Evaluation/ACC3',
        pred='Output/PRED',
        alpha='Attention/ALPHA',
        X='X',
        asp='asp',
        lx='lx',
        y='y',
        dropout_keep='dropout_keep'
    )

    OPS = dict(
        train_op='TrainOp/TRAIN_OP'
    )

    OPTIMIZERS = dict(
        adagrad=tf.train.AdagradOptimizer,
        adam=tf.train.AdamOptimizer,
        sgd=tf.train.GradientDescentOptimizer,
        momentum=tf.train.MomentumOptimizer,
        rmsprop=tf.train.RMSPropOptimizer
    )

    INITIALIZERS = dict(
        random_uniform=tf.random_uniform_initializer,
        random_normal=tf.random_normal_initializer,
        xavier=tf.contrib.layers.xavier_initializer
    )

    def __init__(self, datamanager=None, parameters=None):
        self.graph = None
        self.sess = None
        self.dm = datamanager
        self.p = parameters

    @abc.abstractmethod
    def build_graph(self):
        # graph = tf.Graph()
        # with graph.as_default():
        #     pass
        # return graph
        raise NotImplementedError

    def __retrieve_tensors(self):
        return {k: self.graph.get_tensor_by_name(v + ':0') for k, v in self.TENSORS.items()}

    def __retrieve_ops(self):
        return {k: self.graph.get_operation_by_name(v) for k, v in self.OPS.items()}

    def _get_optimizer(self):
        optimizer = self.OPTIMIZERS[self.p['optimizer']]
        params = filter_params(optimizer, self.p)
        logger.info('Applying params to {} optimizer: {}'.format(self.p['optimizer'], params))
        return optimizer(**params)

    def _get_initializer(self):
        initializer = self.INITIALIZERS[self.p['initializer']]
        params = filter_params(initializer, self.p)
        logger.info('Applying params to {} initializer: {}'.format(self.p['initializer'], params))
        return initializer(**params)

    def train(self, train_df, val_df=None, test_df=None, early_stop=False):

        if self.graph is None:
            self.graph = self.build_graph()

        if val_df is not None:
            _, val_batch = next(self.dm.batch_generator(val_df, batch_size=-1))
            # X_val, asp_val, lx_val, y_val = val_batch
        if test_df is not None:
            _, test_batch = next(self.dm.batch_generator(test_df, batch_size=-1))

        T = self.__retrieve_tensors()
        O = self.__retrieve_ops()

        run_args = (T['loss'], T['regularizer'], O['train_op'], T['acc3'])
        placeholders = (T['X'], T['asp'], T['lx'], T['y'], T['dropout_keep'])
        best_loss = sys.maxsize
        best_acc = 0
        recoder = []

        with self.graph.as_default():
            sess = tf.Session(graph=self.graph)
            init = tf.global_variables_initializer()
            sess.run(init)
            self.saver = tf.train.Saver()
            for epoch in range(self.p['epochs']):
                batch_generator = self.dm.batch_generator(train_df, batch_size=self.p['batch_size'], shuffle=self.p['shuffle'])
                epoch_memory = None

                for i,(_, batch) in enumerate(batch_generator):
                    if epoch_memory is None:
                        epoch_memory = np.zeros([self.dm.n_batches, 2])
                    batch += (self.p['dropout_keep_prob'],)
                    loss_, regl_, _, acc3_ = sess.run(run_args, feed_dict=dict(zip(placeholders, batch)))
                    epoch_memory[i,:] = [loss_, acc3_]
                    logger.debug('epoch {epoch:03d}/{epochs:03d} '
                                 'batch {i:03d}/{n_batches:03d} '
                                 'loss={loss:.4f} '
                                 'l2={l2:.4f} '
                                 'train_acc3={acc:.2%}'
                                 .format(epoch=epoch+1, epochs=self.p['epochs'], i=i+1, n_batches=self.dm.n_batches,
                                         loss=loss_, acc=acc3_, l2=regl_))

                epoch_loss, epoch_acc = epoch_memory.mean(axis=0)

                epoch_str = 'epoch {epoch:03d}/{epochs:03d} ' \
                            'train_loss={loss:.4f} ' \
                            'train_acc3={acc:.2%}'\
                    .format(epoch=epoch+1, epochs=self.p['epochs'], loss=epoch_loss, acc=epoch_acc)

                if val_df is not None:
                    val_acc3_, val_loss_ = sess.run([T['acc3'], T['loss']], feed_dict=dict(zip(placeholders, val_batch)))
                    val_str = 'val_loss={loss:.4f} ' \
                              'val_acc3={acc:.2%}'\
                        .format(loss=val_loss_, acc=val_acc3_)
                    epoch_str += ' ' + val_str

                if test_df is not None:
                    test_acc3_, test_loss_ = sess.run([T['acc3'], T['loss']], feed_dict=dict(zip(placeholders, test_batch)))
                    test_str = 'test_loss={loss:.4f} ' \
                               'test_acc3={acc:.2%}'\
                        .format(loss=test_loss_, acc=test_acc3_)
                    epoch_str += ' ' + test_str

                if bool(early_stop):
                    recoder.append(epoch_str)
                    stop_on = early_stop if isinstance(early_stop, str) else 'loss'
                    if stop_on == 'loss':
                        if val_loss_ > best_loss:
                            logger.info(epoch_str)
                            logger.info('**EARLY STOP** loss={:.4f} > previous={:.4f}'.format(val_loss_, best_loss))
                            break
                        else:
                            best_loss = val_loss_
                    elif stop_on == 'acc3':
                        if val_acc3_ < best_acc and epoch != 1:
                            logger.info(epoch_str)
                            logger.info('**EARLY STOP** acc3={:.2%} < previous={:.2%}'.format(val_acc3_, best_acc))
                            break
                        elif val_acc3_ == best_acc and val_loss_ > best_loss and epoch != 1:
                            logger.info(epoch_str)
                            logger.info('**EARLY STOP** acc3={:.2%} = previous={:.2%}, stop on smaller loss'.format(val_acc3_, best_acc))
                            break
                        else:
                            best_acc = val_acc3_
                            best_loss = val_loss_
                    else:
                        raise AttributeError('Invalid early_stop argument {}, bool, loss or acc3'.format(stop_on))

                logger.info(epoch_str)

            self.sess = sess

            if early_stop:
                return recoder

    def save(self, path):
        mkdir(path)
        self.saver.save(self.sess, path + self.NAME)
        pickle_dump(self.dm, path + 'dm.pkl')
        logger.info("Model saved to '{}'".format(path))

    def load(self, ckpt_dir):
        dm_path = ckpt_dir + '/dm.pkl'
        self.dm = pickle_load(dm_path)
        logger.info('Datamanager restored form {}'.format(dm_path))
        meta_file = ckpt_dir + '/' + self.NAME + '.meta'
        if self.graph is None and self.sess is None:
            self.sess = tf.Session()
            saver = tf.train.import_meta_graph(meta_file)
            self.graph = tf.get_default_graph()
            logger.info('Graph restored from {}'.format(meta_file))
            saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        else:
            raise AttributeError('Graph/Session alreay exist!')

    def predict(self, pred_df):
        T = self.__retrieve_tensors()
        run_args = (T['pred'], T['alpha'], T['loss'])
        _, batch = next(self.dm.batch_generator(pred_df, batch_size=-1))
        _X, _asp, _lx, _y = batch
        pred_, alpha_, loss_ = self.sess.run(run_args, feed_dict={T['X']: _X, T['asp']: _asp, T['lx']: _lx})
        return pred_, alpha_, loss_

    def score(self, pred_df):
        T = self.__retrieve_tensors()
        run_args = (T['pred'], T['alpha'], T['loss'], T['acc3'])
        placeholders = (T['X'], T['asp'], T['lx'], T['y'])
        _, batch = next(self.dm.batch_generator(pred_df, batch_size=-1))
        pred_, alpha_, loss_, acc3_ = self.sess.run(run_args, feed_dict=dict(zip(placeholders, batch)))
        return pred_, alpha_, loss_, acc3_

    def close_session(self):
        self.sess.close()