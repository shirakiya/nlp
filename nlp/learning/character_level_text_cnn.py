import tensorflow as tf


class CharacterLevelTextCNN(object):

    def __init__(self, sequence_length, num_classes, embedding_dim,
                 filter_sizes, num_filters, l2_reg_lambda, batch_normalized=False):
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_dim], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        l2_loss = tf.constant(0.0)

        input_x_expanded = tf.expand_dims(self.input_x, -1)

        pooled_output = []
        for filter_size in filter_sizes:
            with tf.name_scope(f'conv-maxpool-{filter_size}'):
                filter_shape = [filter_size, embedding_dim, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                conv = tf.nn.conv2d(input_x_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')
                if batch_normalized:
                    print('use batch normalization')
                    mean, variance = tf.nn.moments(conv, [0])
                    offset = tf.Variable(tf.truncated_normal([num_filters], stddev=0.1), name='offset')
                    scale = tf.Variable(tf.truncated_normal([num_filters], stddev=0.1), name='scale')
                    f = tf.nn.batch_normalization(conv,
                                                  mean,
                                                  variance,
                                                  offset,
                                                  scale,
                                                  variance_epsilon=1e-5)
                else:
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                    f = tf.nn.bias_add(conv, b)
                h = tf.nn.relu(f, name='relu')
                pool = tf.nn.max_pool(h,
                                      ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                      strides=[1, 1, 1, 1],
                                      padding='VALID',
                                      name='pool')
                pooled_output.append(pool)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_output, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
