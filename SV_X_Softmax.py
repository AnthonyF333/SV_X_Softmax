def SV_X_Softmax(embedding, label, out_num, w_init=None, s=32, m=0.45, mask=1.12, is_am=True):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param out_num: output class num
    :param s: scalar value, default is 32
    :param m: the margin value, default is 0.45
    :param mask: the parameter for calculating hard example or support vector
    :param is_am: choose AM_Softmax or ArcFace
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)

    with tf.variable_scope('SV_X_Softmax'):
        # normalize inputs and weights
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')

        cos_t = tf.matmul(embedding, weights, name='cos_t')

        # onehot label
        onehot_labels = tf.one_hot(label, depth=out_num, name='one_hot_mask')
        gt_cos = cos_t * onehot_labels
        gt_cos = tf.reduce_sum(gt_cos, reduction_indices=1)

        if is_am:
            cos_t_cmp = tf.expand_dims(gt_cos - m,1)
            cos_t_cmp = tf.tile(cos_t_cmp,[1,out_num])
            hard_example_index = cos_t > cos_t_cmp
            cos_t = tf.where(onehot_labels == 1 and cos_t > m, cos_t - m, cos_t)
        else:
            gt_sin = tf.sqrt(1.0 - tf.pow(gt_cos, 2))
            cos_t_m = gt_cos * cos_m - gt_sin * sin_m
            cos_tm_cmp = tf.expand_dims(cos_t_m, 1)
            cos_tm_cmp = tf.tile(cos_tm_cmp, [1, out_num])
            hard_example_index = cos_t > cos_tm_cmp
            cos_t = tf.where(onehot_labels == 1 and cos_t > 0.0, cos_t_m, cos_t)

        # process hard example
        cos_t = tf.where(hard_example_index, mask * cos_t + mask - 1.0, cos_t)
        cos_t *= s
    return cos_t
