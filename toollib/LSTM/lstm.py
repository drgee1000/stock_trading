import tensorflow as tf
import time

class LSTM:


    def __init__(self, batch_size=64, num_steps=50,
                 lstm_size=512, num_layers=2, learning_rate=0.001,
                 grad_clip=5):
        self.batch_size = batch_size
        self.num_steps = num_steps  # Number of sequence steps per batch
        self.lstm_size = lstm_size  # Size of hidden layers in LSTMs
        self.num_layers = num_layers  # Number of LSTM layers
        self.learning_rate = learning_rate  # Learning rate
        self.keep_prob = 0.5  # Dropout keep probability
        self.model = CharRNN(num_classes = 3, batch_size=self.batch_size, num_steps=self.num_steps,
                        lstm_size=self.lstm_size, num_layers=self.num_layers,
                        learning_rate=self.learning_rate)

    def train(self,x,y):
        epochs = 20
        # Print losses every N interations
        print_every_n = 50

        # Save every N iterations
        save_every_n = 200

        saver = tf.train.Saver(max_to_keep=100)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Use the line below to load a checkpoint and resume training
            # saver.restore(sess, 'checkpoints/______.ckpt')
            counter = 0
            for e in range(epochs):
                # Train network
                new_state = sess.run(self.model.initial_state)
                loss = 0
                counter += 1
                start = time.time()
                feed = {self.model.inputs: x,
                        self.model.targets: y,
                        self.model.keep_prob: self.keep_prob,
                        self.model.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.model.loss,
                                                     self.model.final_state,
                                                     self.model.optimizer],
                                                    feed_dict=feed)
                if (counter % print_every_n == 0):
                    end = time.time()
                    print('Epoch: {}/{}... '.format(e + 1, epochs),
                          'Training Step: {}... '.format(counter),
                          'Training loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if (counter % save_every_n == 0):
                    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.lstm_size))

            saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.lstm_size))

    def predict(self,x_test):
        with tf.Session() as sess:
            new_state = sess.run(self.model.final_state)

            feed = {self.model.inputs: x_test,
                    self.model.keep_prob: 1.,
                    self.model.initial_state: new_state}
            preds, new_state = sess.run([self.model.prediction, self.model.final_state],
                                        feed_dict=feed)
        return preds

def build_inputs(batch_size, num_steps):
    ''' Define placeholders for inputs, targets, and dropout

        Arguments
        ---------
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch

    '''
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.float32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, ], name='targets')

    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, keep_prob


def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' Build LSTM cell.

        Arguments
        ---------
        keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
        lstm_size: Size of the hidden layers in the LSTM cells
        num_layers: Number of LSTM layers
        batch_size: Batch size

    '''

    ### Build the LSTM Cell

    def build_cell(lstm_size, keep_prob):
        # Use a basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state


def build_output(lstm_output, in_size, out_size):
    ''' Build a softmax layer, return the softmax output and logits.

        Arguments
        ---------

        x: Input tensor
        in_size: Size of the input tensor, for example, size of the LSTM cells
        out_size: Size of this softmax layer

    '''

    # Reshape output so it's a bunch of rows, one row for each step for each sequence.
    # That is, the shape should be batch_size*num_steps rows by lstm_size columns
    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])

    # Connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and sequence
    logits = tf.matmul(x, softmax_w) + softmax_b

    # Use softmax to get the probabilities for predicted characters
    out = tf.nn.softmax(logits, name='predictions')

    return out, logits


def build_loss(logits, targets, lstm_size, num_classes):
    ''' Calculate the loss from the logits and the targets.

        Arguments
        ---------
        logits: Logits from final fully connected layer
        targets: Targets for supervised learning
        lstm_size: Number of LSTM hidden units
        num_classes: Number of classes in targets

    '''

    # One-hot encode targets and reshape to match logits, one row per batch_size per step
    #y_reshaped = tf.reshape(targets, logits.get_shape())

    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=logits)
    loss = tf.reduce_mean(loss)
    return loss


def build_optimizer(loss, learning_rate, grad_clip):
    ''' Build optmizer for training, using gradient clipping.

        Arguments:
        loss: Network loss
        learning_rate: Learning rate for optimizer

    '''

    # Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer


class CharRNN:

    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5):


        tf.reset_default_graph()

        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        # Build the LSTM cell
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        ### Run the data through the RNN layers
        X = tf.expand_dims(self.inputs, axis=2)
        # Run each sequence step through the RNN and collect the outputs
        outputs, state = tf.nn.dynamic_rnn(cell, X, initial_state=self.initial_state)
        self.final_state = state

        # Get softmax predictions and logits
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

        # Loss and optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)