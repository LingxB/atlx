import tensorflow as tf


def matmul_2_3(tensor_2d, tensor_3d):
    """
    Matmul broadcasting 2d tensor to 3d tensor

    Parameters
    ----------
    tensor_2d : tensor
        i.e. a (5,5) tensor
    tensor_3d : tensor
        i.e. a (3, 5, 4)

    Returns
    -------
    tensor
        i.e. a (3, 5, 4) tensor

    """

    # reshape 2d to 3d

    left = tf.reshape(tensor_2d, (1, tensor_2d.shape[0], tensor_2d.shape[1]))
    left = tf.tile(left, [tf.shape(tensor_3d)[0], 1, 1])

    return tf.matmul(left, tensor_3d)


def seq_length(sequence):
    """
    Compute real sequence length on 3d tensor where 0 is used for padding. i.e. Tensor with shape (batch, N, d),
    sequence is padded with 0 vectors to form N as max_length of batch. This function computes the real length of each
    example in batch.

    Parameters
    ----------
    sequence : tensor
        3D tensor with shape (batch, N, d) where empty time steps are padded with 0 vector

    Returns
    -------
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def var_by_len(a, l):
    """Compute variance of `a` ignoring padding according to length `l`"""
    n_items = tf.shape(a)[0]
    init_ary = tf.TensorArray(dtype=tf.float32,
                              size=n_items,
                              infer_shape=False)
    def _variances(i, ta, begin=tf.convert_to_tensor([0], tf.int32)):
        mean, varian = tf.nn.moments(
            tf.slice(input_=a[i], begin=begin, size=l[i]),
            axes=[0]) # <-- compute variance
        ta = ta.write(i, varian) # <-- write variance of each row to `TensorArray`
        return i+1, ta

    _, variances = tf.while_loop(lambda i, ta: i < n_items,
                                 _variances,
                                 [ 0, init_ary])
    variances = variances.stack() # <-- read from `TensorArray` to `Tensor`

    return variances


def ent_by_len(a, l):
    """Compute entropy of `a` ignoring padding according to length `l`"""
    n_items = tf.shape(a)[0]
    init_ary = tf.TensorArray(dtype=tf.float32,
                              size=n_items,
                              infer_shape=False)

    def _entropy(i, ta, begin=tf.convert_to_tensor([0], tf.int32)):
        s = tf.slice(input_=a[i], begin=begin, size=l[i])
        ent = -1 * tf.reduce_sum(s * tf.log(s))
        ta = ta.write(i, ent)
        return i+1, ta

    _, variances = tf.while_loop(lambda i, ta: i < n_items,
                                 _entropy,
                                 [ 0, init_ary])
    variances = variances.stack() # <-- read from `TensorArray` to `Tensor`

    return variances
