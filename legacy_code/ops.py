import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
from tensorflow.python.ops import rnn

from perceptual_class import Vgg16_perceptual_loss, Inception_v3_feature, Vgg16_class, Inception_v3_class

from utils import pytorch_kaiming_weight_factor

##################################################################################
# Initialization
##################################################################################

# factor, mode, uniform = pytorch_kaiming_weight_factor(a=0.0, uniform=False)
# weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

weight_regularizer = None
weight_regularizer_fully = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer_fully)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer_fully,
                                use_bias=use_bias)

        return x


def flatten(x):
    return tf.layers.flatten(x)

def various_rnn(x, n_layer=1, n_hidden=128, dropout_rate=0.5, bidirectional=True, rnn_type='lstm', scope='rnn') :

    if rnn_type.lower() == 'lstm' :
        cell_type = tf.nn.rnn_cell.LSTMCell
    elif rnn_type.lower() == 'gru' :
        cell_type = tf.nn.rnn_cell.GRUCell
    else :
        raise NotImplementedError

    with tf.variable_scope(scope):
        if bidirectional:
            if n_layer > 1 :
                fw_cells = [cell_type(n_hidden) for _ in range(n_layer)]
                bw_cells = [cell_type(n_hidden) for _ in range(n_layer)]

                if dropout_rate > 0.0:
                    fw_cell = [tf.nn.rnn_cell.DropoutWrapper(cell=fw_cell, output_keep_prob=1 - dropout_rate) for fw_cell in fw_cells]
                    bw_cell = [tf.nn.rnn_cell.DropoutWrapper(cell=bw_cell, output_keep_prob=1 - dropout_rate) for bw_cell in bw_cells]

                fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell)
                bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell)

            else :
                fw_cell = cell_type(n_hidden)
                bw_cell = cell_type(n_hidden)

                if dropout_rate > 0.0 :
                    fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=fw_cell, output_keep_prob=1 - dropout_rate)
                    bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=bw_cell, output_keep_prob=1 - dropout_rate)

            outputs, states = rnn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=x, dtype=tf.float32)
            # outputs = 모든 state
            # states = 마지막 state = output[-1]
            output_fw, output_bw = outputs[0], outputs[1] # [bs, seq_len, n_hidden]
            state_fw, state_bw = states[0], states[1]

            words_emb = tf.concat([output_fw, output_bw], axis=-1) # [bs, seq_len, n_hidden * 2]

            # state_fw[0] = cell state
            # state_fw[1] = hidden state

            if rnn_type.lower() == 'lstm':
                sent_emb = tf.concat([state_fw[1], state_bw[1]], axis=-1) # [bs, n_hidden * 2]
            elif rnn_type.lower() == 'gru':
                sent_emb = tf.concat([state_fw, state_bw], axis=-1)  # [bs, n_hidden * 2]
            else :
                raise NotImplementedError

        else :
            if n_layer > 1 :
                cells = [cell_type(n_hidden) for _ in range(n_layer)]

                if dropout_rate > 0.0 :
                    cell = [tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=1 - dropout_rate) for cell in cells]

                cell = tf.nn.rnn_cell.MultiRNNCell(cell)
            else :
                cell = cell_type(n_hidden)

                if dropout_rate > 0.0 :
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=1 - dropout_rate)

            outputs, states = rnn.dynamic_rnn(cell, inputs=x, dtype=tf.float32)

            words_emb = outputs # [bs, seq_len, n_hidden]

            # states[0] = cell state
            # states[1] = hidden state
            if rnn_type.lower() == 'lstm' :
                sent_emb = states[1] # [bs, n_hidden]
            elif rnn_type.lower() == 'gru' :
                sent_emb = states # [bs, n_hidden]
            else :
                raise NotImplementedError

        return words_emb, sent_emb


##################################################################################
# Residual-block
##################################################################################


def resblock(x_init, channels, is_training=True, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels * 2, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)
            x = glu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)

        return x + x_init

def up_block(x_init, channels, is_training=True, use_bias=True, sn=False, scope='up_block'):
    with tf.variable_scope(scope):
        x = up_sample(x_init, scale_factor=2)
        x = conv(x, channels * 2, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
        x = batch_norm(x, is_training)
        x = glu(x)

        return x

def down_block(x_init, channels, is_training=True, use_bias=True, sn=False, scope='down_block'):
    with tf.variable_scope(scope):
        x = conv(x_init, channels, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
        x = batch_norm(x, is_training)
        x = lrelu(x, 0.2)

        return x


def spatial_attn_net(x, word_emb, mask, channels, use_bias=True, sn=False, scope='spatial_attn_net'):
    with tf.variable_scope(scope):
        bs, h, w = x.shape[0], x.shape[1], x.shape[2]
        hw = h * w # length of query
        seq_len = word_emb.shape[1] # length of source
        # channels = idf

        x = tf.reshape(x, shape=[bs, hw, -1])
        word_emb = tf.expand_dims(word_emb, axis=1)
        word_emb = conv(word_emb, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv')
        word_emb = tf.squeeze(word_emb, axis=1)


        attn = tf.matmul(x, word_emb, transpose_b=True) # [bs, hw, seq_len]
        attn = tf.reshape(attn, shape=[bs * hw, seq_len])


        mask = tf.tile(mask, multiples=[hw, 1])
        attn = tf.where(tf.equal(mask, True), x=tf.constant(-float('inf'), dtype=tf.float32, shape=mask.shape), y=attn)
        attn = tf.nn.softmax(attn)

        attn = tf.reshape(attn, shape=[bs, hw, seq_len])

        weighted_context = tf.matmul(word_emb, attn, transpose_a=True, transpose_b=True) # [bs, hw, channels]
        weighted_context = tf.reshape(weighted_context, shape=[bs, h, w, -1])
        attn = tf.reshape(attn, shape=[bs, h, w, -1])

        return weighted_context, attn

def channel_attn_net(weighted_context, word_emb, use_bias=True, sn=False, scope='channel_attn_net'):
    with tf.variable_scope(scope):
        bs, h, w, idf = weighted_context.shape[0], weighted_context.shape[1], weighted_context.shape[2], weighted_context.shape[3]
        seq_len = word_emb.shape[1]
        word_emb = tf.expand_dims(word_emb, axis=1)
        word_emb = conv(word_emb, channels=h*w, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv')
        word_emb = tf.squeeze(word_emb, axis=1)

        # weighted_context = [bs, hw, idf]
        # word_emb = [bs, seq_len, hw]
        weighted_context = tf.reshape(weighted_context, [bs, h*w, -1])
        attn_c = tf.matmul(weighted_context, word_emb, transpose_a=True, transpose_b=True) # [bs, idf, seq_len]
        attn_c = tf.reshape(attn_c, [bs * idf, seq_len])
        attn_c = tf.nn.softmax(attn_c)

        attn_c = tf.reshape(attn_c, [bs, idf, seq_len])
        weightedContext_c = tf.matmul(word_emb, attn_c, transpose_a=True, transpose_b=True)
        weightedContext_c = tf.reshape(weightedContext_c, [bs, h, w, idf])

        return weightedContext_c, attn_c


##################################################################################
# Sampling
##################################################################################

def dropout(x, drop_rate=0.5, is_training=True):
    return tf.layers.dropout(x, drop_rate, training=is_training)

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def resize(x, target_size):
    return tf.image.resize_bilinear(x, size=target_size)

def down_sample_avg(x, scale_factor=2):
    return tf.layers.average_pooling2d(x, pool_size=3, strides=scale_factor, padding='SAME')

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def reparametrize(mean, logvar):
    eps = tf.random_normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

def simoid(x) :
    return tf.sigmoid(x)

def glu(x) :
    ch = x.shape[-1]
    ch = ch // 2

    n_dim = len(np.shape(x))

    if n_dim == 2:
        return x[:, :ch] * simoid(x[:, ch:])

    else : # n_dim = 4
        return x[:, :, :, :ch] * simoid(x[:, :, :, ch:])


##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=False, scope='batch_norm'):
    """
    if x_norm = tf.layers.batch_normalization
    # ...
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss)
    """

    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

    # return tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-05, center=True, scale=True, training=is_training, name=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


##################################################################################
# Loss function
##################################################################################

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y)) # [64, h, w, c]

    return loss

def L2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))

    return loss

def discriminator_loss(gan_type, real_logit, fake_logit):
    real_loss = 0
    fake_loss = 0

    if real_logit is None :
        if gan_type == 'lsgan':
            fake_loss = tf.reduce_mean(tf.square(fake_logit))
        if gan_type == 'gan':
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))

        if gan_type == 'hinge':
            fake_loss = tf.reduce_mean(relu(1 + fake_logit))
    else :
        if gan_type == 'lsgan':
            real_loss = tf.reduce_mean(tf.squared_difference(real_logit, 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake_logit))

        if gan_type == 'gan':
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))

        if gan_type == 'hinge':

            real_loss = tf.reduce_mean(relu(1 - real_logit))
            fake_loss = tf.reduce_mean(relu(1 + fake_logit))

    return real_loss, fake_loss


def generator_loss(gan_type, fake_logit):
    fake_loss = 0

    if gan_type == 'lsgan':
        fake_loss = tf.reduce_mean(tf.squared_difference(fake_logit, 1.0))

    if gan_type == 'gan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit))

    if gan_type == 'hinge':
        fake_loss = -tf.reduce_mean(fake_logit)

    return fake_loss

def get_inception_feature(x, channel, class_inception_v3, sn=False) :

    x = resize(x, [299, 299])
    mixed_7_feature = Inception_v3_feature(class_inception_v3)(x)
    emb_feature = conv(mixed_7_feature, channel, kernel=1, stride=1, use_bias=False, sn=sn, scope='emb_conv')

    return emb_feature


def vgg16_perceptual_loss(x, y, class_vgg_16):

    loss = Vgg16_perceptual_loss(class_vgg_16)(x, y)

    return loss

def regularization_loss(scope_name):
    """
    If you want to use "Regularization"
    g_loss += regularization_loss('generator')
    d_loss += regularization_loss('discriminator')
    """
    collection_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = []
    for item in collection_regularization:
        if scope_name in item.name:
            loss.append(item)

    return tf.reduce_sum(loss)


def kl_loss(mean, logvar):
    # shape : [batch_size, channel]
    # loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - 1 - logvar, axis=-1)
    # loss = tf.reduce_mean(loss)
    loss = 0.5 * tf.reduce_mean(tf.square(mean) + tf.exp(logvar) - 1 - logvar)

    return loss

def cosine_similarity(x, y):

    xy = tf.reduce_sum(x * y, axis=-1)
    x = tf.norm(x, axis=-1)
    y = tf.norm(y, axis=-1)

    similarity = (xy / ((x * y) + 1e-8))

    return similarity

def word_level_correlation_loss(img_feature, word_emb, gamma1=4.0, gamma2=5.0):

    # img_feature = [bs, 17, 17, 256] = context
    # word_emb = [bs, seq_len, 256 = hidden * 2] = query

    # func_attention
    batch_size = img_feature.shape[0]
    seq_len = word_emb.shape[1]
    similar_list = []

    for i in range(batch_size) :
        context = tf.expand_dims(img_feature[i], axis=0)
        word = tf.expand_dims(word_emb[i], axis=0)

        weighted_context, attn = func_attention(context, word, gamma1)
        # weighted_context = [bs, 256, seq_len]
        # attn = [bs, h, w, seq_len]

        aver_word = tf.reduce_mean(word, axis=1, keepdims=True) # [bs, 1, 256]

        res_word = tf.matmul(aver_word, word, transpose_b=True) # [bs, 1, seq_len]
        res_word_softmax = tf.nn.softmax(res_word, axis=1)
        res_word_softmax = tf.tile(res_word_softmax, multiples=[1, weighted_context.shape[1], 1]) # [bs, 256, seq_len]

        self_weighted_context = tf.transpose(weighted_context * res_word_softmax, perm=[0, 2, 1]) # [bs, seq_len, 256]

        word = tf.reshape(word, [seq_len, -1]) # [seq_len, 256]
        self_weighted_context = tf.reshape(self_weighted_context, [seq_len, -1]) # [seq_len, 256]

        row_sim = cosine_similarity(word, self_weighted_context) #[seq_len]

        row_sim = tf.exp(row_sim * gamma2)
        row_sim = tf.reduce_sum(row_sim) # []
        row_sim = tf.log(row_sim)

        similar_list.append(row_sim)

    word_match_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=similar_list, labels=tf.ones_like(similar_list)))
    word_mismatch_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=similar_list, labels=tf.zeros_like(similar_list)))

    loss = (word_match_loss + word_mismatch_loss) / 2.0

    return loss

def func_attention(img_feature, word_emb, gamma1=4.0):
    # word_emb = query
    # img_feature = context
    # 256 = self.emb_dim

    bs, seq_len = word_emb.shape[0], word_emb.shape[1] # seq_len = length of query
    h, w = img_feature.shape[1], img_feature.shape[2]
    hw = h * w # length of source

    # context = [bs, 17, 17, 256]
    # query = [bs, seq_len, 256]
    # 256 = ndf
    context = tf.reshape(img_feature, [bs, hw, -1]) # [bs, hw, 256]
    attn = tf.matmul(context, word_emb, transpose_b=True) # [bs, hw, seq_len]
    attn = tf.reshape(attn, [bs*hw, seq_len])
    attn = tf.nn.softmax(attn)

    attn = tf.reshape(attn, [bs, hw, seq_len])
    attn = tf.transpose(attn, perm=[0, 2, 1])
    attn = tf.reshape(attn, [bs*seq_len, hw])

    attn = attn * gamma1
    attn = tf.nn.softmax(attn)
    attn = tf.reshape(attn, [bs, seq_len, hw])

    weighted_context = tf.matmul(context, attn, transpose_a=True, transpose_b=True) # [bs, 256, seq_len]

    return weighted_context, tf.reshape(tf.transpose(attn, [0, 2, 1]), [bs, h, w, seq_len])

##################################################################################
# Natural Language Processing
##################################################################################

def embed_sequence(x, n_words, embed_dim, init_range=0.1, trainable=True, scope='embed_layer') :
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) :
        embeddings = tf_contrib.layers.embed_sequence(x, n_words, embed_dim,
                                                      initializer=tf.random_uniform_initializer(minval=-init_range, maxval=init_range),
                                                      trainable=trainable)

        return embeddings
