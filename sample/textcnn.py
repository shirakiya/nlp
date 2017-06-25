import os
import time
import datetime
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import base_import  # noqa
from nlp.preprocessing.cleaning import clean_text_en
from nlp.learning.text_cnn import TextCNN
from nlp.dataset import generate_batch


base_path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

# Data Parameters
tf.flags.DEFINE_float('test_sample_percentage', 0.1,
                      'Percentage of the training data to use for validation')
tf.flags.DEFINE_string('positive_data_file',
                       '/Users/shirakiya/datasets/nlp/sentence-polarity-data/rt-polaritydata/rt-polarity.pos',
                       'Data source for the positive data.')
tf.flags.DEFINE_string('negative_data_file',
                       '/Users/shirakiya/datasets/nlp/sentence-polarity-data/rt-polaritydata/rt-polarity.neg',
                       'Data source for the negative data.')
# Model Hyperparameters
tf.flags.DEFINE_integer('embedding_dim', 128,
                        'Dimensionality of character embedding (default: 128)')
tf.flags.DEFINE_string('filter_sizes', '3,4,5',
                       'Comma-separated filter sizes (default: "3,4,5")')
tf.flags.DEFINE_integer('num_filters', 128,
                        'Number of filters per filter size (default: 128)')
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
np.random.seed(10)


def load_data_and_labels(positive_data_file, negative_data_file):
    positive_examples = [s.strip() for s in open(positive_data_file, 'r')]
    negative_examples = [s.strip() for s in open(negative_data_file, 'r')]
    x_data = [clean_text_en(sent) for sent in positive_examples + negative_examples]
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    labels = np.concatenate([positive_labels, negative_labels], 0)
    return x_data, labels


# don't use (train_test_split is more useful)
def shuffle(x, y):
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    return x[shuffle_indices], y[shuffle_indices]


def train():
    # Load data
    x_text, y = load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    vocab_size = len(vocab_processor.vocabulary_)

    # Randomly shuffle and split data
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=FLAGS.test_sample_percentage,
                                                        random_state=10)
    print('Vocabulary Size: {}'.format(vocab_size))
    print('Train/Test split: {}/{}'.format(len(y_train), len(y_test)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        with tf.Session(config=session_conf) as sess:
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=vocab_size,
                embedding_size=FLAGS.embedding_dim,
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

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, 'vocab'))

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
                    cnn.dropout_keep_prob: 0,
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
    train()
