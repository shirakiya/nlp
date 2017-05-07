import os
import time
import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from nlp.datasets.livedoor import Livedoor
from nlp.preprocessing.cleaning import clean_text_ja
from nlp.word_vectors.word2vec import Word2Vec
from nlp.learning.character_level_text_cnn import CharacterLevelTextCNN
from nlp.dataset import generate_batch


base_path = os.path.dirname(os.path.abspath(__file__))

# Data Parameters
tf.flags.DEFINE_string('data_dir', '/Users/shirakiya/datasets/nlp/livedoor-news-data/origin',
                       'Data source directory (assume "Livedoor Text Corpus")')
tf.flags.DEFINE_string('embeddings_dir', os.path.join(base_path, 'data', 'livedoor_char.w2v.bin'),
                       'Directory path of pre-trained char embeddings model')
tf.flags.DEFINE_float('test_ratio', 0.2,
                      'Percentage of the training data to use for validation')
# Model Hyperparameters
tf.flags.DEFINE_integer('max_document_length', 1024,
                        'Length per sequences (= Character count of one x-data)')
tf.flags.DEFINE_integer('min_document_length', 0,
                        'Minimum length per sequences')
tf.flags.DEFINE_integer('embedding_dim', 200,
                        'Dimensionality of character embedding (default: 200)')
tf.flags.DEFINE_string('filter_sizes', '3,4,5',
                       'Comma-separated filter sizes (default: "3,4,5")')
tf.flags.DEFINE_integer('num_filters', 64,
                        'Number of filters per filter size (default: 64)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5,
                      'Dropout keep probability (default: 0.5)')
tf.flags.DEFINE_float('l2_reg_lambda', 0.0,
                      'L2 regularization lambda (default: 0.0)')
# Training Parameters
tf.flags.DEFINE_integer('batch_size', 64,
                        'Batch Size (default: 64)')
tf.flags.DEFINE_integer('num_epochs', 200,
                        'Number of training epochs (default: 200)')
tf.flags.DEFINE_integer('evaluate_every', 100,
                        'Evaluate model on dev set after this many steps (default: 100)')
tf.flags.DEFINE_integer('checkpoint_every', 100,
                        'Save model after this many steps (default: 100)')
tf.flags.DEFINE_integer('num_checkpoints', 5,
                        'Number of checkpoints to store (default: 5)')
# Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True,
                        'Allow device soft device placement')
tf.flags.DEFINE_boolean('log_device_placement', False,
                        'Log placement of ops on devices')
tf.flags.DEFINE_string('output_dir', os.path.join(base_path, 'runs'),
                       'Summary log placement of train')

FLAGS = tf.flags.FLAGS


def pretrain_embeddings():
    params = {
        'sg': 0,  # 0: CBOW, 1: skip-gram
        'alpha': 0.025,
        'min_alpha': 0.0001,
        'size': FLAGS.embedding_dim,
        'window': 5,
        'min_count': 3,
        'workers': os.cpu_count(),
        'iter': 10,
    }

    livedoor = Livedoor(FLAGS.data_dir)
    texts, _ = livedoor.get_data()

    chars = []
    for text in texts:
        for line in clean_text_ja(text.strip()).split('\n'):
            chars.extend(list(line))

    joined_chars = ' '.join(chars)
    word2vec = Word2Vec()
    word2vec.train([joined_chars], FLAGS.embeddings_dir, **params)


def load_data_and_labels():
    word2vec = Word2Vec()
    word2vec.load(FLAGS.embeddings_dir)

    livedoor = Livedoor(FLAGS.data_dir)
    texts, labels = livedoor.get_data()
    num_classes = len(list(set(labels)))

    # Vectorize
    x = []
    y = []
    for text, label in zip(texts, labels):
        # x
        chars = [char for line in clean_text_ja(text.strip()).split('\n') for char in line]
        chars = chars[:FLAGS.max_document_length]  # cut off more than max_document_length
        chars_len = len(chars)
        if chars_len < FLAGS.min_document_length:  # don't use too short data
            continue
        x_ = []
        for char in chars:
            try:
                vec = word2vec.get_word_vector(char)
            except KeyError:
                vec = np.zeros(FLAGS.embedding_dim)
            x_.append(vec)
        if chars_len < FLAGS.max_document_length:  # align max_document_length
            x_ = np.pad(x_, ((0, FLAGS.max_document_length - chars_len), (0, 0)), mode='constant')
        x.append(x_)
        # y
        y_ = np.zeros(num_classes)
        y_[label] = 1
        y.append(y_)
    return x, y


def train():
    x, y = load_data_and_labels()
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=FLAGS.test_ratio,
                                                        random_state=10)
    print('Train/Test split: {}/{}'.format(len(y_train), len(y_test)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        with tf.Session(config=session_conf) as sess:
            cnn = CharacterLevelTextCNN(
                sequence_length=FLAGS.max_document_length,
                num_classes=len(y_train[0]),
                embedding_dim=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                batch_normalized=True)

            # Define Training procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name), g)
                    sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.join(os.path.abspath(FLAGS.output_dir), timestamp)
            print(f'Writing to {out_dir}')

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar('loss', cnn.loss)
            acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Test summaries
            test_summary_op = tf.summary.merge([loss_summary, acc_summary])
            test_summary_dir = os.path.join(out_dir, 'summaries', 'test')
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.join(out_dir, 'checkpoints')
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                }
                _, step, summaries, loss, accuracy = sess.run([
                    train_op,
                    global_step,
                    train_summary_op,
                    cnn.loss,
                    cnn.accuracy
                ], feed_dict=feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print(f'{time_str}: step {step}, loss {loss:g}, acc {accuracy:g}')
                train_summary_writer.add_summary(summaries, step)

            def test_step(x_batch, y_batch, writer=None):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 0.0,
                }
                step, summaries, loss, accuracy = sess.run([
                    global_step,
                    test_summary_op,
                    cnn.loss,
                    cnn.accuracy
                ], feed_dict=feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print(f'{time_str}: step {step}, loss {loss:g}, acc {accuracy:g}')
                if writer:
                    writer.add_summary(summaries, step)

            batches = generate_batch(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print('Evaluation:')
                    test_step(x_test, y_test, writer=test_summary_writer)
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print(f'Saved model checkpoint to {path}')


if __name__ == '__main__':
    pretrain_embeddings()
    train()
