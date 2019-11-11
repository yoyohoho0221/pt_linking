# ## RNN with fully connected layers for predicting the position of a particle in the future frames
#
# based on the previous movements of a particle
#
#

import functools
import tensorflow as tf
from tensorflow.python.framework import ops
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
    sequence_length = inputs.shape[
        1].value  # the length of sequences processed in the antecedent RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal((hidden_size, attention_size), stddev=0.1))
    b_omega = tf.Variable(tf.random_normal((attention_size,), stddev=0.1))
    u_omega = tf.Variable(tf.random_normal((attention_size,), stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.add(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), w_omega),
                           tf.reshape(b_omega, [1, -1])))
        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
    variable_summaries('alphas', alphas)

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


def variable_summaries(name, var):
    """Attach mean/max/min/sd & histogram for TensorBoard visualization."""
    with tf.name_scope(name):
        # Find the mean of the variable say W.
        mean = tf.reduce_mean(var)
        # Log the mean as scalar
        tf.summary.scalar('ave', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # Log var as a histogram
        # tf.summary.histogram('histogram', var)


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def wxb_layer(inputs, in_size, out_size, n_layer, truncate=None, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('wxb_layer'):
        Weights = weight_variable((in_size, out_size), truncate=truncate)
        Biases = bias_variable((out_size,), 0.1)
        variable_summaries('layer_' + n_layer + '/weights', Weights)
        variable_summaries('layer_' + n_layer + '/bias', Biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), Biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        variable_summaries('layer_' + n_layer + '/outputs', outputs)
        return outputs


def weight_variable(shape, name='weights', truncate=None):
    if truncate is None:
        sd = 4.0 * tf.sqrt(6.0 / ((shape[0] + shape[1]) * 1.))
        initial = tf.random_uniform(shape, minval=-sd, maxval=sd)
    elif truncate:
        sd = 0.1  # tf.sqrt(6.0 / ((shape[0] + shape[1]) * 1.))
        initial = tf.truncated_normal(shape, stddev=sd)
    else:
        sd = 0.5
        initial = tf.random_normal(shape, mean=0., stddev=sd)
    return tf.Variable(initial_value=initial, name=name)


def bias_variable(shape, c, name='biases', trainable=True):
    initial = tf.constant(c, shape=shape)
    return tf.Variable(initial_value=initial, name=name, trainable=trainable)


def xlength(x):
    usedx = tf.sign(tf.reduce_max(tf.abs(x), reduction_indices=2))
    lengthx = tf.reduce_sum(usedx, reduction_indices=1)
    lengthx = tf.cast(lengthx, tf.int32)
    return lengthx


# position only or handcraft NN
class MotionRNN(object):
    def __init__(self, ninput, noutput, nclasses, lstms, lstmo, R, H, L, k,
                 at=None, rnn='lstm', delta=1., alpha=1., isTest=False, isAtt=False):
        self._ninput = ninput
        self._noutput = noutput
        self._lstms = lstms
        self._lstmo = lstmo
        self._nclasses = nclasses
        self._R = R
        self._H = H
        self._L = L
        self._k = k
        self._delta = delta
        self._alpha = alpha
        self._isTest = isTest
        self._isAtt = isAtt

        if at == 'tanh':
            act = tf.nn.tanh
        elif at == 'relu':
            act = tf.nn.relu
        elif at == 'relu6':
            act = tf.nn.relu6
        elif at == 'crelu':
            act = tf.nn.crelu
        elif at == 'elu':
            act = tf.nn.elu
        elif at == 'softsign':
            act = tf.nn.softsign
        elif at == 'softplus':
            act = tf.nn.softplus
        elif at == 'sigmoid':
            act = tf.sigmoid
        elif at == 'selu':
            act = selu
        else:
            act = None

        with tf.variable_scope('inputs'):
            # tf Graph input layer
            bs = None
            self.x = tf.placeholder(tf.float32, [bs, self._lstms, self._ninput], name='x')  # input
            self.xp = tf.placeholder(tf.float32, [], name='xp')  # input
            self.y = tf.placeholder(tf.float32, [bs, self._lstmo, self._ninput], name='y')  # input
            self.xlastloc = tf.placeholder(tf.float32, [bs, self._noutput], name='last')  # input
            self.pos = tf.placeholder(tf.float32, [bs, self._noutput], name='pos')
            self.c = tf.placeholder(tf.float32, [bs, self._nclasses], name='cls')
            self.d = tf.placeholder(tf.float32, [bs, 1], name='cls')
            self._dropout = tf.placeholder(tf.float32, [], name='keep_prop')
            self._learningrate = tf.placeholder(tf.float32, [], name='lr')

        with tf.variable_scope('rnnx'):
            # define cells for rnn and dropout between layers
            x_fc = tf.contrib.layers.fully_connected(self.x, self._L, activation_fn=None)
            if rnn == 'lstm':
                cellx = tf.nn.rnn_cell.LSTMCell(self._L, activation=act)
            elif rnn == 'gru':
                cellx = tf.nn.rnn_cell.GRUCell(self._L, activation=act)
            else:
                cellx = tf.nn.rnn_cell.BasicRNNCell(self._L, activation=act)
            cellx = tf.nn.rnn_cell.ResidualWrapper(cellx)
            cellx = tf.nn.rnn_cell.DropoutWrapper(cellx, output_keep_prob=self._dropout)
            cellx = tf.nn.rnn_cell.MultiRNNCell([cellx] * self._R)

            # create rnn work with variance lengths
            x_all_output, self.x_last_state = tf.nn.dynamic_rnn(cellx, x_fc, dtype=tf.float32)
            if self._isAtt:
                x_output, self.alphas = attention(x_all_output, self._L, return_alphas=True)
            else:
                x_output = tf.contrib.layers.flatten(x_all_output)
            if self._R == 1:
                if rnn == 'lstm':
                    x_last_output = self.x_last_state[0][-1]
                else:
                    x_last_output = self.x_last_state[0]
            elif self._R == 2:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1]],
                                              1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1]], 1)
            else:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1],
                                               self.x_last_state[2][-1]], 1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1],
                                               self.x_last_state[2]], 1)

        with tf.variable_scope('fcy'):
            # define fully connected layer for t+1 displacement vector
            y_fc = tf.contrib.layers.fully_connected(self.y, self._L, activation_fn=None)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_fc = tf.contrib.layers.fully_connected(y_dp, self._L, activation_fn=act)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_output = tf.contrib.layers.flatten(y_dp)

        with tf.variable_scope('conca'):
            # concatenate two vectors x_output and y_output
            con = tf.concat([x_output, y_output], 1)
            conca = tf.contrib.layers.fully_connected(con, self._k, activation_fn=act)
            concadp = tf.contrib.layers.dropout(conca, self._dropout)

        with tf.variable_scope('predc'):
            self.pred = wxb_layer(concadp, self._k, self._nclasses, n_layer='c',
                                  activation_function=act)
            self.predc = tf.nn.softmax(self.pred)

        with tf.variable_scope('predd'):
            self.predd = wxb_layer(concadp, self._k, 1, n_layer='d', truncate=False,
                                   activation_function=None)

        with tf.variable_scope('predy'):
            fc = tf.contrib.layers.fully_connected(x_last_output, self._k, activation_fn=None)
            dp = tf.contrib.layers.dropout(fc, self._dropout)
            self.predy = wxb_layer(dp, self._k, self._noutput, n_layer='y', truncate=False,
                                   activation_function=None)
            self.nextloc = tf.add(self.xlastloc, self.predy)

        with tf.name_scope('loss'):
            self.loss()

        with tf.name_scope('train'):
            # self.optimizer = tf.train.GradientDescentOptimizer(self._learningrate).minimize(self.total_error)
            self.optimizer = tf.train.AdamOptimizer(self._learningrate).minimize(self.total_error)
            tf.summary.scalar('learning_rate', self._learningrate)

        with tf.name_scope('prediction_power'):
            self.pred_power()

    def huberloss(self):
        x = tf.abs(tf.subtract(self.nextloc, self.pos))
        x = tf.where(x <= self._delta, 0.5 * tf.square(x), self._delta * (x - 0.5 * self._delta))
        return tf.reduce_sum(x)

    def loss(self):
        with tf.name_scope('RMSE'):
            # MSE(L2) to give us the loss of the regression
            # rmse = tf.reduce_sum(tf.squared_difference(self.predy, self.pos))
            # smoothL1 loss
            rmse = self.huberloss()
        with tf.name_scope('CROSS_ENTROPY'):
            # cross entropy to give us the loss of the classification
            # cross = -tf.reduce_sum(self.c * tf.log(self.predc))  # tf.clip_by_value(self.predc, 1e-10, 1.0)))
            cross = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.c, 1),
                                                               logits=self.pred))
        with tf.name_scope('CMSE'):
            # cross entropy to give us the loss of the classification
            cmse = tf.reduce_sum(tf.squared_difference(self.predd, self.d))
            # cmse = tf.losses.mean_squared_error(self.d, self.predd)
        with tf.name_scope('TOTAL_LOSS'):
            # total error
            self.total_error = rmse + cross + self._alpha * cmse
        tf.summary.scalar('rmse_loss', rmse)
        tf.summary.scalar('cross_entropy_loss', cross)
        tf.summary.scalar('cmse_loss', cmse)
        tf.summary.scalar('total_loss', self.total_error)
        return rmse, cross, cmse, self.total_error

    def pred_power(self):
        predictions = tf.cast(tf.argmax(self.predc, 1), tf.float32)
        actuals = tf.cast(tf.argmax(self.c, 1), tf.float32)
        with tf.name_scope('err'):
            self.error = tf.reduce_mean(tf.cast(tf.not_equal(actuals, predictions), tf.float32))
        with tf.name_scope('rmseloss'):
            self.rmse = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.squared_difference(self.nextloc, self.pos), axis=1)))
        with tf.name_scope('cmseloss'):
            self.cmse = tf.reduce_mean(tf.abs(self.predd - self.d))
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)
        with tf.name_scope('TP'):
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('TN'):
            tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FP'):
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FN'):
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('acc'):
            accuracy = (tp + tn) / (tp + fp + fn + tn)
        with tf.name_scope('rec'):
            if tp + fn == tf.constant(0):
                recall = tf.constant(0)
            else:
                recall = tp / (tp + fn)
        with tf.name_scope('pre'):
            if tp + fp == tf.constant(0):
                precision = tf.constant(0)
            else:
                precision = tp / (tp + fp)

        tf.summary.scalar('rmse', self.rmse)
        tf.summary.scalar('cmse', self.cmse)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('recall', recall)
        tf.summary.scalar('precision', precision)
        return self.cmse, accuracy, recall, precision, self.rmse


# CNN learned
class MotionCRNN(object):
    def __init__(self, ninput, noutput, nclasses, lstms, lstmo, R, H, L, k,
                 at=None, rnn='lstm', delta=1., alpha=1., isTest=False, isAtt=False):
        self._ninput = ninput
        self._noutput = noutput
        self._lstms = lstms
        self._lstmo = lstmo
        self._nclasses = nclasses
        self._R = R
        self._H = H
        self._L = L
        self._k = k
        self._delta = delta
        self._alpha = alpha
        self._isTest = isTest
        self._isAtt = isAtt

        if at == 'tanh':
            act = tf.nn.tanh
        elif at == 'relu':
            act = tf.nn.relu
        elif at == 'relu6':
            act = tf.nn.relu6
        elif at == 'crelu':
            act = tf.nn.crelu
        elif at == 'elu':
            act = tf.nn.elu
        elif at == 'softsign':
            act = tf.nn.softsign
        elif at == 'softplus':
            act = tf.nn.softplus
        elif at == 'sigmoid':
            act = tf.sigmoid
        elif at == 'selu':
            act = selu
        else:
            act = None

        with tf.variable_scope('inputs'):
            # tf Graph input layer
            bs = None
            self.x = tf.placeholder(tf.float32, [bs, self._lstms + self._lstmo, self._ninput],
                                    name='x')
            self.xp = tf.placeholder(tf.float32, [], name='xp')  # input
            self.y = tf.placeholder(tf.float32, [], name='y')
            self.xlastloc = tf.placeholder(tf.float32, [bs, self._noutput], name='last')
            self.pos = tf.placeholder(tf.float32, [bs, self._noutput], name='pos')
            self.c = tf.placeholder(tf.float32, [bs, self._nclasses], name='cls')
            self.d = tf.placeholder(tf.float32, [bs, 1], name='cls')
            self._dropout = tf.placeholder(tf.float32, [], name='keep_prop')
            self._learningrate = tf.placeholder(tf.float32, [], name='lr')

        with tf.variable_scope('cnn'):
            # define cells for rnn and dropout between layers
            conv11 = tf.layers.conv1d(self.x, self._H, 1, padding='same', activation=None)
            conv12 = tf.layers.conv1d(self.x, self._H, 2, padding='same', activation=None)
            conv13 = tf.layers.conv1d(self.x, self._H, 3, padding='same', activation=None)
            conv14 = tf.layers.conv1d(self.x, self._H, 4, padding='same', activation=None)
            conv15 = tf.layers.conv1d(self.x, self._H, 5, padding='same', activation=None)
            fxy = tf.concat([conv11, conv12, conv13, conv14, conv15], axis=2)
            fx, fy = tf.split(fxy, [self._lstms, self._lstmo], axis=1)

        with tf.variable_scope('rnn'):
            fc = tf.contrib.layers.fully_connected(fx, self._L, activation_fn=None)
            if rnn == 'lstm':
                cell = tf.nn.rnn_cell.LSTMCell(self._L, activation=act)
            elif rnn == 'gru':
                cell = tf.nn.rnn_cell.GRUCell(self._L, activation=act)
            else:
                cell = tf.nn.rnn_cell.BasicRNNCell(self._L, activation=act)
            cell = tf.nn.rnn_cell.ResidualWrapper(cell)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._dropout)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self._R)

            x_all_output, self.x_last_state = tf.nn.dynamic_rnn(cell, fc, dtype=tf.float32)
            if self._isAtt:
                x_output, self.alphas = attention(x_all_output, self._L, return_alphas=True)
            else:
                x_output = tf.contrib.layers.flatten(x_all_output)
            if self._R == 1:
                if rnn == 'lstm':
                    x_last_output = self.x_last_state[0][-1]
                else:
                    x_last_output = self.x_last_state[0]
            elif self._R == 2:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1]],
                                              1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1]], 1)
            else:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1],
                                               self.x_last_state[2][-1]], 1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1],
                                               self.x_last_state[2]], 1)

        with tf.variable_scope('fcy'):
            # define fully connected layer for t+1 displacement vector
            y_fc = tf.contrib.layers.fully_connected(fy, self._L, activation_fn=None)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_fc = tf.contrib.layers.fully_connected(y_dp, self._L, activation_fn=act)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_output = tf.contrib.layers.flatten(y_dp)

        with tf.variable_scope('conca'):
            # concatenate two vectors x_output and y_output
            con = tf.concat([x_output, y_output], 1)
            conca = tf.contrib.layers.fully_connected(con, self._k, activation_fn=act)
            concadp = tf.contrib.layers.dropout(conca, self._dropout)

        with tf.variable_scope('predc'):
            self.pred = wxb_layer(concadp, self._k, self._nclasses, n_layer='c',
                                  activation_function=act)
            self.predc = tf.nn.softmax(self.pred)

        with tf.variable_scope('predd'):
            self.predd = wxb_layer(concadp, self._k, 1, n_layer='d', truncate=False,
                                   activation_function=None)

        with tf.variable_scope('predy'):
            fc = tf.contrib.layers.fully_connected(x_last_output, self._k, activation_fn=None)
            dp = tf.contrib.layers.dropout(fc, self._dropout)
            self.predy = wxb_layer(dp, self._k, self._noutput, n_layer='y', truncate=False,
                                   activation_function=None)
            self.nextloc = tf.add(self.xlastloc, self.predy)

        with tf.name_scope('loss'):
            self.loss()

        with tf.name_scope('train'):
            # self.optimizer = tf.train.GradientDescentOptimizer(self._learningrate).minimize(self.total_error)
            self.optimizer = tf.train.AdamOptimizer(self._learningrate).minimize(self.total_error)
            tf.summary.scalar('learning_rate', self._learningrate)

        with tf.name_scope('prediction_power'):
            self.pred_power()

    def huberloss(self):
        x = tf.abs(tf.subtract(self.nextloc, self.pos))
        x = tf.where(x <= self._delta, 0.5 * tf.square(x), self._delta * (x - 0.5 * self._delta))
        return tf.reduce_sum(x)

    def loss(self):
        with tf.name_scope('RMSE'):
            # MSE(L2) to give us the loss of the regression
            # rmse = tf.reduce_sum(tf.squared_difference(self.predy, self.pos))
            # smoothL1 loss
            rmse = self.huberloss()
        with tf.name_scope('CROSS_ENTROPY'):
            # cross entropy to give us the loss of the classification
            # cross = -tf.reduce_sum(self.c * tf.log(self.predc))  # tf.clip_by_value(self.predc, 1e-10, 1.0)))
            cross = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.c, 1),
                                                               logits=self.pred))
        with tf.name_scope('CMSE'):
            # cross entropy to give us the loss of the classification
            cmse = tf.reduce_sum(tf.squared_difference(self.predd, self.d))
            # cmse = tf.losses.mean_squared_error(self.d, self.predd)
        with tf.name_scope('TOTAL_LOSS'):
            # total error
            self.total_error = rmse + cross + self._alpha * cmse
        tf.summary.scalar('rmse_loss', rmse)
        tf.summary.scalar('cross_entropy_loss', cross)
        tf.summary.scalar('cmse_loss', cmse)
        tf.summary.scalar('total_loss', self.total_error)
        return rmse, cross, cmse, self.total_error

    def pred_power(self):
        predictions = tf.cast(tf.argmax(self.predc, 1), tf.float32)
        actuals = tf.cast(tf.argmax(self.c, 1), tf.float32)
        with tf.name_scope('err'):
            self.error = tf.reduce_mean(tf.cast(tf.not_equal(actuals, predictions), tf.float32))
        with tf.name_scope('rmseloss'):
            self.rmse = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.squared_difference(self.nextloc, self.pos), axis=1)))
        with tf.name_scope('cmseloss'):
            self.cmse = tf.reduce_mean(tf.abs(self.predd - self.d))
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)
        with tf.name_scope('TP'):
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('TN'):
            tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FP'):
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FN'):
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('acc'):
            accuracy = (tp + tn) / (tp + fp + fn + tn)
        with tf.name_scope('rec'):
            if tp + fn == tf.constant(0):
                recall = tf.constant(0)
            else:
                recall = tp / (tp + fn)
        with tf.name_scope('pre'):
            if tp + fp == tf.constant(0):
                precision = tf.constant(0)
            else:
                precision = tp / (tp + fp)

        tf.summary.scalar('rmse', self.rmse)
        tf.summary.scalar('cmse', self.cmse)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('recall', recall)
        tf.summary.scalar('precision', precision)
        return self.cmse, accuracy, recall, precision, self.rmse


# RNN learned
class MotionRRNN(object):
    def __init__(self, ninput, noutput, nclasses, lstms, lstmo, R, H, L, k,
                 at=None, rnn='lstm', delta=1., alpha=1., isTest=False, isAtt=False):
        self._ninput = ninput
        self._noutput = noutput
        self._lstms = lstms
        self._lstmo = lstmo
        self._nclasses = nclasses
        self._R = R
        self._H = H
        self._L = L
        self._k = k
        self._delta = delta
        self._alpha = alpha
        self._isTest = isTest
        self._isAtt = isAtt

        if at == 'tanh':
            act = tf.nn.tanh
        elif at == 'relu':
            act = tf.nn.relu
        elif at == 'relu6':
            act = tf.nn.relu6
        elif at == 'crelu':
            act = tf.nn.crelu
        elif at == 'elu':
            act = tf.nn.elu
        elif at == 'softsign':
            act = tf.nn.softsign
        elif at == 'softplus':
            act = tf.nn.softplus
        elif at == 'sigmoid':
            act = tf.sigmoid
        elif at == 'selu':
            act = selu
        else:
            act = None

        with tf.variable_scope('inputs'):
            # tf Graph input layer
            bs = None
            self.x = tf.placeholder(tf.float32, [bs, self._lstms + self._lstmo, self._ninput],
                                    name='x')
            self.xp = tf.placeholder(tf.float32, [], name='xp')  # input
            self.y = tf.placeholder(tf.float32, [], name='y')
            self.xlastloc = tf.placeholder(tf.float32, [bs, self._noutput], name='last')
            self.pos = tf.placeholder(tf.float32, [bs, self._noutput], name='pos')
            self.c = tf.placeholder(tf.float32, [bs, self._nclasses], name='cls')
            self.d = tf.placeholder(tf.float32, [bs, 1], name='cls')
            self._dropout = tf.placeholder(tf.float32, [], name='keep_prop')
            self._learningrate = tf.placeholder(tf.float32, [], name='lr')

        with tf.variable_scope('features'):
            if rnn == 'lstm':
                cellfx = tf.nn.rnn_cell.LSTMCell(self._H, activation=None)
            elif rnn == 'gru':
                cellfx = tf.nn.rnn_cell.GRUCell(self._H, activation=None)
            else:
                cellfx = tf.nn.rnn_cell.BasicRNNCell(self._H, activation=None)
            # create rnn work with variance lengths
            fxy, _ = tf.nn.dynamic_rnn(cellfx, self.x, dtype=tf.float32)
            fx, fy = tf.split(fxy, [self._lstms, self._lstmo], axis=1)

        with tf.variable_scope('rnnx'):
            # define cells for rnn and dropout between layers
            fxrnn = tf.contrib.layers.fully_connected(fx, self._L, activation_fn=None)
            if rnn == 'lstm':
                cell = tf.nn.rnn_cell.LSTMCell(self._L, activation=act)
            elif rnn == 'gru':
                cell = tf.nn.rnn_cell.GRUCell(self._L, activation=act)
            else:
                cell = tf.nn.rnn_cell.BasicRNNCell(self._L, activation=act)
            cell = tf.nn.rnn_cell.ResidualWrapper(cell)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._dropout)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self._R)

            # create rnn work with variance lengths
            x_all_output, self.x_last_state = tf.nn.dynamic_rnn(cell, fxrnn, dtype=tf.float32)
            if self._isAtt:
                x_output, self.alphas = attention(x_all_output, self._L, return_alphas=True)
            else:
                x_output = tf.contrib.layers.flatten(x_all_output)
            if self._R == 1:
                if rnn == 'lstm':
                    x_last_output = self.x_last_state[0][-1]
                else:
                    x_last_output = self.x_last_state[0]
            elif self._R == 2:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1]],
                                              1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1]], 1)
            else:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1],
                                               self.x_last_state[2][-1]], 1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1],
                                               self.x_last_state[2]], 1)

        with tf.variable_scope('fcy'):
            # define fully connected layer for t+1 displacement vector
            y_fc = tf.contrib.layers.fully_connected(fy, self._L, activation_fn=None)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_fc = tf.contrib.layers.fully_connected(y_dp, self._L, activation_fn=act)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_output = tf.contrib.layers.flatten(y_dp)

        with tf.variable_scope('conca'):
            # concatenate two vectors x_output and y_output
            con = tf.concat([x_output, y_output], 1)
            conca = tf.contrib.layers.fully_connected(con, self._k, activation_fn=act)
            concadp = tf.contrib.layers.dropout(conca, self._dropout)

        with tf.variable_scope('predc'):
            self.pred = wxb_layer(concadp, self._k, self._nclasses, n_layer='c',
                                  activation_function=act)
            self.predc = tf.nn.softmax(self.pred)

        with tf.variable_scope('predd'):
            self.predd = wxb_layer(concadp, self._k, 1, n_layer='d', truncate=False,
                                   activation_function=None)

        with tf.variable_scope('predy'):
            fc = tf.contrib.layers.fully_connected(x_last_output, self._k, activation_fn=None)
            dp = tf.contrib.layers.dropout(fc, self._dropout)
            self.predy = wxb_layer(dp, self._k, self._noutput, n_layer='y', truncate=False,
                                   activation_function=None)
            self.nextloc = tf.add(self.xlastloc, self.predy)

        with tf.name_scope('loss'):
            self.loss()

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self._learningrate).minimize(self.total_error)
            tf.summary.scalar('learning_rate', self._learningrate)

        with tf.name_scope('prediction_power'):
            self.pred_power()

    def huberloss(self):
        x = tf.abs(tf.subtract(self.nextloc, self.pos))
        x = tf.where(x <= self._delta, 0.5 * tf.square(x), self._delta * (x - 0.5 * self._delta))
        return tf.reduce_sum(x)

    def loss(self):
        with tf.name_scope('RMSE'):
            # MSE(L2) to give us the loss of the regression
            # rmse = tf.reduce_sum(tf.squared_difference(self.predy, self.pos))
            # smoothL1 loss
            rmse = self.huberloss()
        with tf.name_scope('CROSS_ENTROPY'):
            # cross entropy to give us the loss of the classification
            # cross = -tf.reduce_sum(self.c * tf.log(self.predc))  # tf.clip_by_value(self.predc, 1e-10, 1.0)))
            cross = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.c, 1),
                                                               logits=self.pred))
        with tf.name_scope('CMSE'):
            # cross entropy to give us the loss of the classification
            cmse = tf.reduce_sum(tf.squared_difference(self.predd, self.d))
            # cmse = tf.losses.mean_squared_error(self.d, self.predd)
        with tf.name_scope('TOTAL_LOSS'):
            # total error
            self.total_error = rmse + cross + self._alpha * cmse
        tf.summary.scalar('rmse_loss', rmse)
        tf.summary.scalar('cross_entropy_loss', cross)
        tf.summary.scalar('cmse_loss', cmse)
        tf.summary.scalar('total_loss', self.total_error)
        return rmse, cross, cmse, self.total_error

    def pred_power(self):
        predictions = tf.cast(tf.argmax(self.predc, 1), tf.float32)
        actuals = tf.cast(tf.argmax(self.c, 1), tf.float32)
        with tf.name_scope('err'):
            self.error = tf.reduce_mean(tf.cast(tf.not_equal(actuals, predictions), tf.float32))
        with tf.name_scope('rmseloss'):
            self.rmse = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.squared_difference(self.nextloc, self.pos), axis=1)))
        with tf.name_scope('cmseloss'):
            self.cmse = tf.reduce_mean(tf.abs(self.predd - self.d))
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)
        with tf.name_scope('TP'):
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('TN'):
            tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FP'):
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FN'):
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('acc'):
            accuracy = (tp + tn) / (tp + fp + fn + tn)
        with tf.name_scope('rec'):
            if tp + fn == tf.constant(0):
                recall = tf.constant(0)
            else:
                recall = tp / (tp + fn)
        with tf.name_scope('pre'):
            if tp + fp == tf.constant(0):
                precision = tf.constant(0)
            else:
                precision = tp / (tp + fp)

        tf.summary.scalar('rmse', self.rmse)
        tf.summary.scalar('cmse', self.cmse)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('recall', recall)
        tf.summary.scalar('precision', precision)
        return self.cmse, accuracy, recall, precision, self.rmse


# combined RNN learned with handcraft
class MotionRHRNN(object):
    def __init__(self, ninput, ninputp, noutput, nclasses, lstms, lstmo, R, H, L, k,
                 at=None, rnn='lstm', delta=1., alpha=1., isTest=False, isAtt=False):
        self._ninput = ninput
        self._ninputp = ninputp
        self._noutput = noutput
        self._lstms = lstms
        self._lstmo = lstmo
        self._nclasses = nclasses
        self._R = R
        self._H = H
        self._L = L
        self._k = k
        self._delta = delta
        self._alpha = alpha
        self._isTest = isTest
        self._isAtt = isAtt

        if at == 'tanh':
            act = tf.nn.tanh
        elif at == 'relu':
            act = tf.nn.relu
        elif at == 'relu6':
            act = tf.nn.relu6
        elif at == 'crelu':
            act = tf.nn.crelu
        elif at == 'elu':
            act = tf.nn.elu
        elif at == 'softsign':
            act = tf.nn.softsign
        elif at == 'softplus':
            act = tf.nn.softplus
        elif at == 'sigmoid':
            act = tf.sigmoid
        elif at == 'selu':
            act = selu
        else:
            act = None

        with tf.variable_scope('inputs'):
            # tf Graph input layer
            bs = None
            self.x = tf.placeholder(tf.float32, [bs, self._lstms, self._ninput], name='x')  # input
            self.xp = tf.placeholder(tf.float32, [bs, self._lstms + self._lstmo, self._ninputp],
                                     name='xp')  # input
            self.y = tf.placeholder(tf.float32, [bs, self._lstmo, self._ninput], name='y')  # input
            self.xlastloc = tf.placeholder(tf.float32, [bs, self._noutput], name='last')  # input
            self.pos = tf.placeholder(tf.float32, [bs, self._noutput], name='pos')
            self.c = tf.placeholder(tf.float32, [bs, self._nclasses], name='cls')
            self.d = tf.placeholder(tf.float32, [bs, 1], name='cls')
            self._dropout = tf.placeholder(tf.float32, [], name='keep_prop')
            self._learningrate = tf.placeholder(tf.float32, [], name='lr')

        with tf.variable_scope('features'):
            if rnn == 'lstm':
                cellfx = tf.nn.rnn_cell.LSTMCell(self._H, activation=None)
            elif rnn == 'gru':
                cellfx = tf.nn.rnn_cell.GRUCell(self._H, activation=None)
            else:
                cellfx = tf.nn.rnn_cell.BasicRNNCell(self._H, activation=None)
            # create rnn work with variance lengths
            fxy, _ = tf.nn.dynamic_rnn(cellfx, self.xp, dtype=tf.float32)
            fx, fy = tf.split(fxy, [self._lstms, self._lstmo], axis=1)

        with tf.variable_scope('rnnx'):
            # define cells for rnn and dropout between layers
            x_con = tf.concat([self.x, fx], 2)
            x_fc = tf.contrib.layers.fully_connected(x_con, self._L, activation_fn=None)
            if rnn == 'lstm':
                cellx = tf.nn.rnn_cell.LSTMCell(self._L, activation=act)
            elif rnn == 'gru':
                cellx = tf.nn.rnn_cell.GRUCell(self._L, activation=act)
            else:
                cellx = tf.nn.rnn_cell.BasicRNNCell(self._L, activation=act)
            cellx = tf.nn.rnn_cell.ResidualWrapper(cellx)
            cellx = tf.nn.rnn_cell.DropoutWrapper(cellx, output_keep_prob=self._dropout)
            cellx = tf.nn.rnn_cell.MultiRNNCell([cellx] * self._R)

            # create rnn work with variance lengths
            x_all_output, self.x_last_state = tf.nn.dynamic_rnn(cellx, x_fc, dtype=tf.float32)
            if self._isAtt:
                x_output, self.alphas = attention(x_all_output, self._L, return_alphas=True)
            else:
                x_output = tf.contrib.layers.flatten(x_all_output)
            if self._R == 1:
                if rnn == 'lstm':
                    x_last_output = self.x_last_state[0][-1]
                else:
                    x_last_output = self.x_last_state[0]
            elif self._R == 2:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1]],
                                              1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1]], 1)
            else:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1],
                                               self.x_last_state[2][-1]], 1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1],
                                               self.x_last_state[2]], 1)

        with tf.variable_scope('fcy'):
            # define fully connected layer for t+1 displacement vector
            y_con = tf.concat([self.y, fy], 2)
            y_fc = tf.contrib.layers.fully_connected(y_con, self._L, activation_fn=None)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_fc = tf.contrib.layers.fully_connected(y_dp, self._L, activation_fn=act)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_output = tf.contrib.layers.flatten(y_dp)

        with tf.variable_scope('conca'):
            # concatenate two vectors x_output and y_output
            con = tf.concat([x_output, y_output], 1)
            conca = tf.contrib.layers.fully_connected(con, self._k, activation_fn=act)
            concadp = tf.contrib.layers.dropout(conca, self._dropout)

        with tf.variable_scope('predc'):
            self.pred = wxb_layer(concadp, self._k, self._nclasses, n_layer='c',
                                  activation_function=act)
            self.predc = tf.nn.softmax(self.pred)

        with tf.variable_scope('predd'):
            self.predd = wxb_layer(concadp, self._k, 1, n_layer='d', truncate=False,
                                   activation_function=None)

        with tf.variable_scope('predy'):
            fc = tf.contrib.layers.fully_connected(x_last_output, self._k, activation_fn=None)
            dp = tf.contrib.layers.dropout(fc, self._dropout)
            self.predy = wxb_layer(dp, self._k, self._noutput, n_layer='y', truncate=False,
                                   activation_function=None)
            self.nextloc = tf.add(self.xlastloc, self.predy)

        with tf.name_scope('loss'):
            self.loss()

        with tf.name_scope('train'):
            # self.optimizer = tf.train.GradientDescentOptimizer(self._learningrate).minimize(self.total_error)
            self.optimizer = tf.train.AdamOptimizer(self._learningrate).minimize(self.total_error)
            tf.summary.scalar('learning_rate', self._learningrate)

        with tf.name_scope('prediction_power'):
            self.pred_power()

    def huberloss(self):
        x = tf.abs(tf.subtract(self.nextloc, self.pos))
        x = tf.where(x <= self._delta, 0.5 * tf.square(x), self._delta * (x - 0.5 * self._delta))
        return tf.reduce_sum(x)

    def loss(self):
        with tf.name_scope('RMSE'):
            # MSE(L2) to give us the loss of the regression
            # rmse = tf.reduce_sum(tf.squared_difference(self.predy, self.pos))
            # smoothL1 loss
            rmse = self.huberloss()
        with tf.name_scope('CROSS_ENTROPY'):
            # cross entropy to give us the loss of the classification
            # cross = -tf.reduce_sum(self.c * tf.log(self.predc))  # tf.clip_by_value(self.predc, 1e-10, 1.0)))
            cross = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.c, 1),
                                                               logits=self.pred))
        with tf.name_scope('CMSE'):
            # cross entropy to give us the loss of the classification
            cmse = tf.reduce_sum(tf.squared_difference(self.predd, self.d))
            # cmse = tf.losses.mean_squared_error(self.d, self.predd)
        with tf.name_scope('TOTAL_LOSS'):
            # total error
            self.total_error = rmse + cross + self._alpha * cmse
        tf.summary.scalar('rmse_loss', rmse)
        tf.summary.scalar('cross_entropy_loss', cross)
        tf.summary.scalar('cmse_loss', cmse)
        tf.summary.scalar('total_loss', self.total_error)
        return rmse, cross, cmse, self.total_error

    def pred_power(self):
        predictions = tf.cast(tf.argmax(self.predc, 1), tf.float32)
        actuals = tf.cast(tf.argmax(self.c, 1), tf.float32)
        with tf.name_scope('err'):
            self.error = tf.reduce_mean(tf.cast(tf.not_equal(actuals, predictions), tf.float32))
        with tf.name_scope('rmseloss'):
            self.rmse = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.squared_difference(self.nextloc, self.pos), axis=1)))
        with tf.name_scope('cmseloss'):
            self.cmse = tf.reduce_mean(tf.abs(self.predd - self.d))
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)
        with tf.name_scope('TP'):
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('TN'):
            tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FP'):
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FN'):
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('acc'):
            accuracy = (tp + tn) / (tp + fp + fn + tn)
        with tf.name_scope('rec'):
            if tp + fn == tf.constant(0):
                recall = tf.constant(0)
            else:
                recall = tp / (tp + fn)
        with tf.name_scope('pre'):
            if tp + fp == tf.constant(0):
                precision = tf.constant(0)
            else:
                precision = tp / (tp + fp)

        tf.summary.scalar('rmse', self.rmse)
        tf.summary.scalar('cmse', self.cmse)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('recall', recall)
        tf.summary.scalar('precision', precision)
        return self.cmse, accuracy, recall, precision, self.rmse


# combined CNN learned with handcraft
class MotionRHCNN(object):
    def __init__(self, ninput, ninputp, noutput, nclasses, lstms, lstmo, R, H, L, k,
                 at=None, rnn='lstm', delta=1., alpha=1., isTest=False, isAtt=False):
        self._ninput = ninput
        self._ninputp = ninputp
        self._noutput = noutput
        self._lstms = lstms
        self._lstmo = lstmo
        self._nclasses = nclasses
        self._R = R
        self._H = H
        self._L = L
        self._k = k
        self._delta = delta
        self._alpha = alpha
        self._isTest = isTest
        self._isAtt = isAtt

        if at == 'tanh':
            act = tf.nn.tanh
        elif at == 'relu':
            act = tf.nn.relu
        elif at == 'relu6':
            act = tf.nn.relu6
        elif at == 'crelu':
            act = tf.nn.crelu
        elif at == 'elu':
            act = tf.nn.elu
        elif at == 'softsign':
            act = tf.nn.softsign
        elif at == 'softplus':
            act = tf.nn.softplus
        elif at == 'sigmoid':
            act = tf.sigmoid
        elif at == 'selu':
            act = selu
        else:
            act = None

        with tf.variable_scope('inputs'):
            # tf Graph input layer
            bs = None
            self.x = tf.placeholder(tf.float32, [bs, self._lstms, self._ninput], name='x')  # input
            self.xp = tf.placeholder(tf.float32, [bs, self._lstms + self._lstmo, self._ninputp],
                                     name='xp')  # input
            self.y = tf.placeholder(tf.float32, [bs, self._lstmo, self._ninput], name='y')  # input
            self.xlastloc = tf.placeholder(tf.float32, [bs, self._noutput], name='last')  # input
            self.pos = tf.placeholder(tf.float32, [bs, self._noutput], name='pos')
            self.c = tf.placeholder(tf.float32, [bs, self._nclasses], name='cls')
            self.d = tf.placeholder(tf.float32, [bs, 1], name='cls')
            self._dropout = tf.placeholder(tf.float32, [], name='keep_prop')
            self._learningrate = tf.placeholder(tf.float32, [], name='lr')

        with tf.variable_scope('features'):
            # define cells for rnn and dropout between layers
            conv11 = tf.layers.conv1d(self.xp, self._H, 1, padding='same', activation=None)
            conv12 = tf.layers.conv1d(self.xp, self._H, 2, padding='same', activation=None)
            conv13 = tf.layers.conv1d(self.xp, self._H, 3, padding='same', activation=None)
            conv14 = tf.layers.conv1d(self.xp, self._H, 4, padding='same', activation=None)
            conv15 = tf.layers.conv1d(self.xp, self._H, 5, padding='same', activation=None)
            fxy = tf.concat([conv11, conv12, conv13, conv14, conv15], axis=2)
            fx, fy = tf.split(fxy, [self._lstms, self._lstmo], axis=1)

        with tf.variable_scope('rnnx'):
            # define cells for rnn and dropout between layers
            x_con = tf.concat([self.x, fx], 2)
            x_fc = tf.contrib.layers.fully_connected(x_con, self._L, activation_fn=None)
            if rnn == 'lstm':
                cellx = tf.nn.rnn_cell.LSTMCell(self._L, activation=act)
            elif rnn == 'gru':
                cellx = tf.nn.rnn_cell.GRUCell(self._L, activation=act)
            else:
                cellx = tf.nn.rnn_cell.BasicRNNCell(self._L, activation=act)
            cellx = tf.nn.rnn_cell.ResidualWrapper(cellx)
            cellx = tf.nn.rnn_cell.DropoutWrapper(cellx, output_keep_prob=self._dropout)
            cellx = tf.nn.rnn_cell.MultiRNNCell([cellx] * self._R)

            # create rnn work with variance lengths
            x_all_output, self.x_last_state = tf.nn.dynamic_rnn(cellx, x_fc, dtype=tf.float32)
            if self._isAtt:
                x_output, self.alphas = attention(x_all_output, self._L, return_alphas=True)
            else:
                x_output = tf.contrib.layers.flatten(x_all_output)
            if self._R == 1:
                if rnn == 'lstm':
                    x_last_output = self.x_last_state[0][-1]
                else:
                    x_last_output = self.x_last_state[0]
            elif self._R == 2:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1]],
                                              1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1]], 1)
            else:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1],
                                               self.x_last_state[2][-1]], 1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1],
                                               self.x_last_state[2]], 1)

        with tf.variable_scope('fcy'):
            # define fully connected layer for t+1 displacement vector
            y_con = tf.concat([self.y, fy], 2)
            y_fc = tf.contrib.layers.fully_connected(y_con, self._L, activation_fn=None)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_fc = tf.contrib.layers.fully_connected(y_dp, self._L, activation_fn=act)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_output = tf.contrib.layers.flatten(y_dp)

        with tf.variable_scope('conca'):
            # concatenate two vectors x_output and y_output
            con = tf.concat([x_output, y_output], 1)
            conca = tf.contrib.layers.fully_connected(con, self._k, activation_fn=act)
            concadp = tf.contrib.layers.dropout(conca, self._dropout)

        with tf.variable_scope('predc'):
            self.pred = wxb_layer(concadp, self._k, self._nclasses, n_layer='c',
                                  activation_function=act)
            self.predc = tf.nn.softmax(self.pred)

        with tf.variable_scope('predd'):
            self.predd = wxb_layer(concadp, self._k, 1, n_layer='d', truncate=False,
                                   activation_function=None)

        with tf.variable_scope('predy'):
            fc = tf.contrib.layers.fully_connected(x_last_output, self._k, activation_fn=None)
            dp = tf.contrib.layers.dropout(fc, self._dropout)
            self.predy = wxb_layer(dp, self._k, self._noutput, n_layer='y', truncate=False,
                                   activation_function=None)
            self.nextloc = tf.add(self.xlastloc, self.predy)

        with tf.name_scope('loss'):
            self.loss()

        with tf.name_scope('train'):
            # self.optimizer = tf.train.GradientDescentOptimizer(self._learningrate).minimize(self.total_error)
            self.optimizer = tf.train.AdamOptimizer(self._learningrate).minimize(self.total_error)
            tf.summary.scalar('learning_rate', self._learningrate)

        with tf.name_scope('prediction_power'):
            self.pred_power()

    def huberloss(self):
        x = tf.abs(tf.subtract(self.nextloc, self.pos))
        x = tf.where(x <= self._delta, 0.5 * tf.square(x), self._delta * (x - 0.5 * self._delta))
        return tf.reduce_sum(x)

    def loss(self):
        with tf.name_scope('RMSE'):
            # MSE(L2) to give us the loss of the regression
            # rmse = tf.reduce_sum(tf.squared_difference(self.predy, self.pos))
            # smoothL1 loss
            rmse = self.huberloss()
        with tf.name_scope('CROSS_ENTROPY'):
            # cross entropy to give us the loss of the classification
            # cross = -tf.reduce_sum(self.c * tf.log(self.predc))  # tf.clip_by_value(self.predc, 1e-10, 1.0)))
            cross = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.c, 1),
                                                               logits=self.pred))
        with tf.name_scope('CMSE'):
            # cross entropy to give us the loss of the classification
            cmse = tf.reduce_sum(tf.squared_difference(self.predd, self.d))
            # cmse = tf.losses.mean_squared_error(self.d, self.predd)
        with tf.name_scope('TOTAL_LOSS'):
            # total error
            self.total_error = rmse + cross + self._alpha * cmse
        tf.summary.scalar('rmse_loss', rmse)
        tf.summary.scalar('cross_entropy_loss', cross)
        tf.summary.scalar('cmse_loss', cmse)
        tf.summary.scalar('total_loss', self.total_error)
        return rmse, cross, cmse, self.total_error

    def pred_power(self):
        predictions = tf.cast(tf.argmax(self.predc, 1), tf.float32)
        actuals = tf.cast(tf.argmax(self.c, 1), tf.float32)
        with tf.name_scope('err'):
            self.error = tf.reduce_mean(tf.cast(tf.not_equal(actuals, predictions), tf.float32))
        with tf.name_scope('rmseloss'):
            self.rmse = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.squared_difference(self.nextloc, self.pos), axis=1)))
        with tf.name_scope('cmseloss'):
            self.cmse = tf.reduce_mean(tf.abs(self.predd - self.d))
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)
        with tf.name_scope('TP'):
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('TN'):
            tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FP'):
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FN'):
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('acc'):
            accuracy = (tp + tn) / (tp + fp + fn + tn)
        with tf.name_scope('rec'):
            if tp + fn == tf.constant(0):
                recall = tf.constant(0)
            else:
                recall = tp / (tp + fn)
        with tf.name_scope('pre'):
            if tp + fp == tf.constant(0):
                precision = tf.constant(0)
            else:
                precision = tp / (tp + fp)

        tf.summary.scalar('rmse', self.rmse)
        tf.summary.scalar('cmse', self.cmse)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('recall', recall)
        tf.summary.scalar('precision', precision)
        return self.cmse, accuracy, recall, precision, self.rmse


# multi-task compare handcraft cmse+rmse
class MotionRNN2(object):
    def __init__(self, ninput, noutput, nclasses, lstms, lstmo, R, H, L, k,
                 at=None, rnn='lstm', delta=1., alpha=1., isTest=False, isAtt=False):
        self._ninput = ninput
        self._noutput = noutput
        self._lstms = lstms
        self._lstmo = lstmo
        self._nclasses = nclasses
        self._R = R
        self._H = H
        self._L = L
        self._k = k
        self._delta = delta
        self._alpha = alpha
        self._isTest = isTest
        self._isAtt = isAtt

        if at == 'tanh':
            act = tf.nn.tanh
        elif at == 'relu':
            act = tf.nn.relu
        elif at == 'relu6':
            act = tf.nn.relu6
        elif at == 'crelu':
            act = tf.nn.crelu
        elif at == 'elu':
            act = tf.nn.elu
        elif at == 'softsign':
            act = tf.nn.softsign
        elif at == 'softplus':
            act = tf.nn.softplus
        elif at == 'sigmoid':
            act = tf.sigmoid
        elif at == 'selu':
            act = selu
        else:
            act = None

        with tf.variable_scope('inputs'):
            # tf Graph input layer
            bs = None
            self.x = tf.placeholder(tf.float32, [bs, self._lstms, self._ninput], name='x')  # input
            self.xp = tf.placeholder(tf.float32, [], name='xp')  # input
            self.y = tf.placeholder(tf.float32, [bs, self._lstmo, self._ninput], name='y')  # input
            self.xlastloc = tf.placeholder(tf.float32, [bs, self._noutput], name='last')  # input
            self.pos = tf.placeholder(tf.float32, [bs, self._noutput], name='pos')
            self.c = tf.placeholder(tf.float32, [bs, self._nclasses], name='cls')
            self.d = tf.placeholder(tf.float32, [bs, 1], name='cls')
            self._dropout = tf.placeholder(tf.float32, [], name='keep_prop')
            self._learningrate = tf.placeholder(tf.float32, [], name='lr')

        with tf.variable_scope('rnnx'):
            # define cells for rnn and dropout between layers
            x_fc = tf.contrib.layers.fully_connected(self.x, self._L, activation_fn=None)
            if rnn == 'lstm':
                cellx = tf.nn.rnn_cell.LSTMCell(self._L, activation=act)
            elif rnn == 'gru':
                cellx = tf.nn.rnn_cell.GRUCell(self._L, activation=act)
            else:
                cellx = tf.nn.rnn_cell.BasicRNNCell(self._L, activation=act)
            cellx = tf.nn.rnn_cell.ResidualWrapper(cellx)
            cellx = tf.nn.rnn_cell.DropoutWrapper(cellx, output_keep_prob=self._dropout)
            cellx = tf.nn.rnn_cell.MultiRNNCell([cellx] * self._R)

            # create rnn work with variance lengths
            x_all_output, self.x_last_state = tf.nn.dynamic_rnn(cellx, x_fc, dtype=tf.float32)
            if self._isAtt:
                x_output, self.alphas = attention(x_all_output, self._L, return_alphas=True)
            else:
                x_output = tf.contrib.layers.flatten(x_all_output)
            if self._R == 1:
                if rnn == 'lstm':
                    x_last_output = self.x_last_state[0][-1]
                else:
                    x_last_output = self.x_last_state[0]
            elif self._R == 2:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1]],
                                              1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1]], 1)
            else:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1],
                                               self.x_last_state[2][-1]], 1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1],
                                               self.x_last_state[2]], 1)

        with tf.variable_scope('fcy'):
            # define fully connected layer for t+1 displacement vector
            y_fc = tf.contrib.layers.fully_connected(self.y, self._L, activation_fn=None)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_fc = tf.contrib.layers.fully_connected(y_dp, self._L, activation_fn=act)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_output = tf.contrib.layers.flatten(y_dp)

        with tf.variable_scope('conca'):
            # concatenate two vectors x_output and y_output
            con = tf.concat([x_output, y_output], 1)
            conca = tf.contrib.layers.fully_connected(con, self._k, activation_fn=act)
            concadp = tf.contrib.layers.dropout(conca, self._dropout)

        with tf.variable_scope('predd'):
            self.predd = wxb_layer(concadp, self._k, 1, n_layer='d', truncate=False,
                                   activation_function=None)

        with tf.variable_scope('predy'):
            fc = tf.contrib.layers.fully_connected(x_last_output, self._k, activation_fn=None)
            dp = tf.contrib.layers.dropout(fc, self._dropout)
            self.predy = wxb_layer(dp, self._k, self._noutput, n_layer='y', truncate=False,
                                   activation_function=None)
            self.nextloc = tf.add(self.xlastloc, self.predy)

        with tf.name_scope('loss'):
            self.loss()

        with tf.name_scope('train'):
            # self.optimizer = tf.train.GradientDescentOptimizer(self._learningrate).minimize(self.total_error)
            self.optimizer = tf.train.AdamOptimizer(self._learningrate).minimize(self.total_error)
            tf.summary.scalar('learning_rate', self._learningrate)

        with tf.name_scope('prediction_power'):
            self.pred_power()

    def huberloss(self):
        x = tf.abs(tf.subtract(self.nextloc, self.pos))
        x = tf.where(x <= self._delta, 0.5 * tf.square(x), self._delta * (x - 0.5 * self._delta))
        return tf.reduce_sum(x)

    def loss(self):
        with tf.name_scope('RMSE'):
            # MSE(L2) to give us the loss of the regression
            # rmse = tf.reduce_sum(tf.squared_difference(self.predy, self.pos))
            # smoothL1 loss
            rmse = self.huberloss()
        with tf.name_scope('CMSE'):
            # cross entropy to give us the loss of the classification
            cmse = tf.reduce_sum(tf.squared_difference(self.predd, self.d))
            # cmse = tf.losses.mean_squared_error(self.d, self.predd)
        with tf.name_scope('TOTAL_LOSS'):
            # total error
            self.total_error = rmse + self._alpha * cmse
        tf.summary.scalar('rmse_loss', rmse)
        tf.summary.scalar('cmse_loss', cmse)
        tf.summary.scalar('total_loss', self.total_error)
        return rmse, cmse, cmse, self.total_error

    def pred_power(self):
        with tf.name_scope('rmseloss'):
            self.rmse = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.squared_difference(self.nextloc, self.pos), axis=1)))
        with tf.name_scope('cmseloss'):
            self.cmse = tf.reduce_mean(tf.abs(self.predd - self.d))
        tf.summary.scalar('rmse', self.rmse)
        tf.summary.scalar('cmse', self.cmse)
        return self.cmse, self.rmse


# multi-task compare handcraft cmse
class MotionRNN1c(object):
    def __init__(self, ninput, noutput, nclasses, lstms, lstmo, R, H, L, k,
                 at=None, rnn='lstm', delta=1., alpha=1., isTest=False, isAtt=False):
        self._ninput = ninput
        self._noutput = noutput
        self._lstms = lstms
        self._lstmo = lstmo
        self._nclasses = nclasses
        self._R = R
        self._H = H
        self._L = L
        self._k = k
        self._delta = delta
        self._alpha = alpha
        self._isTest = isTest
        self._isAtt = isAtt

        if at == 'tanh':
            act = tf.nn.tanh
        elif at == 'relu':
            act = tf.nn.relu
        elif at == 'relu6':
            act = tf.nn.relu6
        elif at == 'crelu':
            act = tf.nn.crelu
        elif at == 'elu':
            act = tf.nn.elu
        elif at == 'softsign':
            act = tf.nn.softsign
        elif at == 'softplus':
            act = tf.nn.softplus
        elif at == 'sigmoid':
            act = tf.sigmoid
        elif at == 'selu':
            act = selu
        else:
            act = None

        with tf.variable_scope('inputs'):
            # tf Graph input layer
            bs = None
            self.x = tf.placeholder(tf.float32, [bs, self._lstms, self._ninput], name='x')  # input
            self.xp = tf.placeholder(tf.float32, [], name='xp')  # input
            self.y = tf.placeholder(tf.float32, [bs, self._lstmo, self._ninput], name='y')  # input
            self.xlastloc = tf.placeholder(tf.float32, [bs, self._noutput], name='last')  # input
            self.pos = tf.placeholder(tf.float32, [bs, self._noutput], name='pos')
            self.c = tf.placeholder(tf.float32, [bs, self._nclasses], name='cls')
            self.d = tf.placeholder(tf.float32, [bs, 1], name='cls')
            self._dropout = tf.placeholder(tf.float32, [], name='keep_prop')
            self._learningrate = tf.placeholder(tf.float32, [], name='lr')

        with tf.variable_scope('rnnx'):
            # define cells for rnn and dropout between layers
            x_fc = tf.contrib.layers.fully_connected(self.x, self._L, activation_fn=None)
            if rnn == 'lstm':
                cellx = tf.nn.rnn_cell.LSTMCell(self._L, activation=act)
            elif rnn == 'gru':
                cellx = tf.nn.rnn_cell.GRUCell(self._L, activation=act)
            else:
                cellx = tf.nn.rnn_cell.BasicRNNCell(self._L, activation=act)
            cellx = tf.nn.rnn_cell.ResidualWrapper(cellx)
            cellx = tf.nn.rnn_cell.DropoutWrapper(cellx, output_keep_prob=self._dropout)
            cellx = tf.nn.rnn_cell.MultiRNNCell([cellx] * self._R)

            # create rnn work with variance lengths
            x_all_output, self.x_last_state = tf.nn.dynamic_rnn(cellx, x_fc, dtype=tf.float32)
            if self._isAtt:
                x_output, self.alphas = attention(x_all_output, self._L, return_alphas=True)
            else:
                x_output = tf.contrib.layers.flatten(x_all_output)

        with tf.variable_scope('fcy'):
            # define fully connected layer for t+1 displacement vector
            y_fc = tf.contrib.layers.fully_connected(self.y, self._L, activation_fn=None)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_fc = tf.contrib.layers.fully_connected(y_dp, self._L, activation_fn=act)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_output = tf.contrib.layers.flatten(y_dp)

        with tf.variable_scope('conca'):
            # concatenate two vectors x_output and y_output
            con = tf.concat([x_output, y_output], 1)
            conca = tf.contrib.layers.fully_connected(con, self._k, activation_fn=act)
            concadp = tf.contrib.layers.dropout(conca, self._dropout)

        with tf.variable_scope('predd'):
            self.predd = wxb_layer(concadp, self._k, 1, n_layer='d', truncate=False,
                                   activation_function=None)

        with tf.name_scope('loss'):
            self.loss()

        with tf.name_scope('train'):
            # self.optimizer = tf.train.GradientDescentOptimizer(self._learningrate).minimize(self.total_error)
            self.optimizer = tf.train.AdamOptimizer(self._learningrate).minimize(self.total_error)
            tf.summary.scalar('learning_rate', self._learningrate)

        with tf.name_scope('prediction_power'):
            self.pred_power()

    def loss(self):
        with tf.name_scope('CMSE'):
            # cross entropy to give us the loss of the classification
            cmse = tf.reduce_sum(tf.squared_difference(self.predd, self.d))
            # cmse = tf.losses.mean_squared_error(self.d, self.predd)
        with tf.name_scope('TOTAL_LOSS'):
            # total error
            self.total_error = self._alpha * cmse
        tf.summary.scalar('cmse_loss', cmse)
        tf.summary.scalar('total_loss', self.total_error)
        return cmse, cmse, cmse, self.total_error

    def pred_power(self):
        with tf.name_scope('cmseloss'):
            self.cmse = tf.reduce_mean(tf.abs(self.predd - self.d))
        tf.summary.scalar('cmse', self.cmse)
        return self.cmse, self.cmse


# multi-task compare handcraft rmse
class MotionRNN1r(object):
    def __init__(self, ninput, noutput, nclasses, lstms, lstmo, R, H, L, k,
                 at=None, rnn='lstm', delta=1., alpha=1., isTest=False, isAtt=False):
        self._ninput = ninput
        self._noutput = noutput
        self._lstms = lstms
        self._lstmo = lstmo
        self._nclasses = nclasses
        self._R = R
        self._H = H
        self._L = L
        self._k = k
        self._delta = delta
        self._alpha = alpha
        self._isTest = isTest
        self._isAtt = isAtt

        if at == 'tanh':
            act = tf.nn.tanh
        elif at == 'relu':
            act = tf.nn.relu
        elif at == 'relu6':
            act = tf.nn.relu6
        elif at == 'crelu':
            act = tf.nn.crelu
        elif at == 'elu':
            act = tf.nn.elu
        elif at == 'softsign':
            act = tf.nn.softsign
        elif at == 'softplus':
            act = tf.nn.softplus
        elif at == 'sigmoid':
            act = tf.sigmoid
        elif at == 'selu':
            act = selu
        else:
            act = None

        with tf.variable_scope('inputs'):
            # tf Graph input layer
            bs = None
            self.x = tf.placeholder(tf.float32, [bs, self._lstms, self._ninput], name='x')  # input
            self.xp = tf.placeholder(tf.float32, [], name='xp')  # input
            self.y = tf.placeholder(tf.float32, [bs, self._lstmo, self._ninput], name='y')  # input
            self.xlastloc = tf.placeholder(tf.float32, [bs, self._noutput], name='last')  # input
            self.pos = tf.placeholder(tf.float32, [bs, self._noutput], name='pos')
            self.c = tf.placeholder(tf.float32, [bs, self._nclasses], name='cls')
            self.d = tf.placeholder(tf.float32, [bs, 1], name='cls')
            self._dropout = tf.placeholder(tf.float32, [], name='keep_prop')
            self._learningrate = tf.placeholder(tf.float32, [], name='lr')

        with tf.variable_scope('rnnx'):
            # define cells for rnn and dropout between layers
            x_fc = tf.contrib.layers.fully_connected(self.x, self._L, activation_fn=None)
            if rnn == 'lstm':
                cellx = tf.nn.rnn_cell.LSTMCell(self._L, activation=act)
            elif rnn == 'gru':
                cellx = tf.nn.rnn_cell.GRUCell(self._L, activation=act)
            else:
                cellx = tf.nn.rnn_cell.BasicRNNCell(self._L, activation=act)
            cellx = tf.nn.rnn_cell.ResidualWrapper(cellx)
            cellx = tf.nn.rnn_cell.DropoutWrapper(cellx, output_keep_prob=self._dropout)
            cellx = tf.nn.rnn_cell.MultiRNNCell([cellx] * self._R)

            # create rnn work with variance lengths
            x_all_output, self.x_last_state = tf.nn.dynamic_rnn(cellx, x_fc, dtype=tf.float32)

            if self._R == 1:
                if rnn == 'lstm':
                    x_last_output = self.x_last_state[0][-1]
                else:
                    x_last_output = self.x_last_state[0]
            elif self._R == 2:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1]],
                                              1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1]], 1)
            else:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1],
                                               self.x_last_state[2][-1]], 1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1],
                                               self.x_last_state[2]], 1)

        with tf.variable_scope('predy'):
            fc = tf.contrib.layers.fully_connected(x_last_output, self._k, activation_fn=None)
            dp = tf.contrib.layers.dropout(fc, self._dropout)
            self.predy = wxb_layer(dp, self._k, self._noutput, n_layer='y', truncate=False,
                                   activation_function=None)
            self.nextloc = tf.add(self.xlastloc, self.predy)

        with tf.name_scope('loss'):
            self.loss()

        with tf.name_scope('train'):
            # self.optimizer = tf.train.GradientDescentOptimizer(self._learningrate).minimize(self.total_error)
            self.optimizer = tf.train.AdamOptimizer(self._learningrate).minimize(self.total_error)
            tf.summary.scalar('learning_rate', self._learningrate)

        with tf.name_scope('prediction_power'):
            self.pred_power()

    def huberloss(self):
        x = tf.abs(tf.subtract(self.nextloc, self.pos))
        x = tf.where(x <= self._delta, 0.5 * tf.square(x), self._delta * (x - 0.5 * self._delta))
        return tf.reduce_sum(x)

    def loss(self):
        with tf.name_scope('RMSE'):
            # MSE(L2) to give us the loss of the regression
            # rmse = tf.reduce_sum(tf.squared_difference(self.predy, self.pos))
            # smoothL1 loss
            rmse = self.huberloss()
        with tf.name_scope('TOTAL_LOSS'):
            # total error
            self.total_error = rmse
        tf.summary.scalar('rmse_loss', rmse)
        tf.summary.scalar('total_loss', self.total_error)
        return rmse, rmse, rmse, self.total_error

    def pred_power(self):
        with tf.name_scope('rmseloss'):
            self.rmse = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.squared_difference(self.nextloc, self.pos), axis=1)))
        tf.summary.scalar('rmse', self.rmse)
        return self.rmse, self.rmse


# multi-task compare handcraft cmse+cross
class MotionRNN3c(object):
    def __init__(self, ninput, noutput, nclasses, lstms, lstmo, R, H, L, k,
                 at=None, rnn='lstm', delta=1., alpha=1., isTest=False, isAtt=False):
        self._ninput = ninput
        self._noutput = noutput
        self._lstms = lstms
        self._lstmo = lstmo
        self._nclasses = nclasses
        self._R = R
        self._H = H
        self._L = L
        self._k = k
        self._delta = delta
        self._alpha = alpha
        self._isTest = isTest
        self._isAtt = isAtt

        if at == 'tanh':
            act = tf.nn.tanh
        elif at == 'relu':
            act = tf.nn.relu
        elif at == 'relu6':
            act = tf.nn.relu6
        elif at == 'crelu':
            act = tf.nn.crelu
        elif at == 'elu':
            act = tf.nn.elu
        elif at == 'softsign':
            act = tf.nn.softsign
        elif at == 'softplus':
            act = tf.nn.softplus
        elif at == 'sigmoid':
            act = tf.sigmoid
        elif at == 'selu':
            act = selu
        else:
            act = None

        with tf.variable_scope('inputs'):
            # tf Graph input layer
            bs = None
            self.x = tf.placeholder(tf.float32, [bs, self._lstms, self._ninput], name='x')  # input
            self.xp = tf.placeholder(tf.float32, [], name='xp')  # input
            self.y = tf.placeholder(tf.float32, [bs, self._lstmo, self._ninput], name='y')  # input
            self.xlastloc = tf.placeholder(tf.float32, [bs, self._noutput], name='last')  # input
            self.pos = tf.placeholder(tf.float32, [bs, self._noutput], name='pos')
            self.c = tf.placeholder(tf.float32, [bs, self._nclasses], name='cls')
            self.d = tf.placeholder(tf.float32, [bs, 1], name='cls')
            self._dropout = tf.placeholder(tf.float32, [], name='keep_prop')
            self._learningrate = tf.placeholder(tf.float32, [], name='lr')

        with tf.variable_scope('rnnx'):
            # define cells for rnn and dropout between layers
            x_fc = tf.contrib.layers.fully_connected(self.x, self._L, activation_fn=None)
            if rnn == 'lstm':
                cellx = tf.nn.rnn_cell.LSTMCell(self._L, activation=act)
            elif rnn == 'gru':
                cellx = tf.nn.rnn_cell.GRUCell(self._L, activation=act)
            else:
                cellx = tf.nn.rnn_cell.BasicRNNCell(self._L, activation=act)
            cellx = tf.nn.rnn_cell.ResidualWrapper(cellx)
            cellx = tf.nn.rnn_cell.DropoutWrapper(cellx, output_keep_prob=self._dropout)
            cellx = tf.nn.rnn_cell.MultiRNNCell([cellx] * self._R)

            # create rnn work with variance lengths
            x_all_output, self.x_last_state = tf.nn.dynamic_rnn(cellx, x_fc, dtype=tf.float32)
            if self._isAtt:
                x_output, self.alphas = attention(x_all_output, self._L, return_alphas=True)
            else:
                x_output = tf.contrib.layers.flatten(x_all_output)

        with tf.variable_scope('fcy'):
            # define fully connected layer for t+1 displacement vector
            y_fc = tf.contrib.layers.fully_connected(self.y, self._L, activation_fn=None)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_fc = tf.contrib.layers.fully_connected(y_dp, self._L, activation_fn=act)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_output = tf.contrib.layers.flatten(y_dp)

        with tf.variable_scope('conca'):
            # concatenate two vectors x_output and y_output
            con = tf.concat([x_output, y_output], 1)
            conca = tf.contrib.layers.fully_connected(con, self._k, activation_fn=act)
            concadp = tf.contrib.layers.dropout(conca, self._dropout)

        with tf.variable_scope('predc'):
            self.pred = wxb_layer(concadp, self._k, self._nclasses, n_layer='c',
                                  activation_function=act)
            self.predc = tf.nn.softmax(self.pred)

        with tf.variable_scope('predd'):
            self.predd = wxb_layer(concadp, self._k, 1, n_layer='d', truncate=False,
                                   activation_function=None)

        with tf.name_scope('loss'):
            self.loss()

        with tf.name_scope('train'):
            # self.optimizer = tf.train.GradientDescentOptimizer(self._learningrate).minimize(self.total_error)
            self.optimizer = tf.train.AdamOptimizer(self._learningrate).minimize(self.total_error)
            tf.summary.scalar('learning_rate', self._learningrate)

        with tf.name_scope('prediction_power'):
            self.pred_power()

    def loss(self):
        with tf.name_scope('CROSS_ENTROPY'):
            # cross entropy to give us the loss of the classification
            # cross = -tf.reduce_sum(self.c * tf.log(self.predc))  # tf.clip_by_value(self.predc, 1e-10, 1.0)))
            cross = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.c, 1),
                                                               logits=self.pred))
        with tf.name_scope('CMSE'):
            # cross entropy to give us the loss of the classification
            cmse = tf.reduce_sum(tf.squared_difference(self.predd, self.d))
            # cmse = tf.losses.mean_squared_error(self.d, self.predd)
        with tf.name_scope('TOTAL_LOSS'):
            # total error
            self.total_error = cross + self._alpha * cmse
        tf.summary.scalar('cross_entropy_loss', cross)
        tf.summary.scalar('cmse_loss', cmse)
        tf.summary.scalar('total_loss', self.total_error)
        return cmse, cross, cmse, self.total_error

    def pred_power(self):
        predictions = tf.cast(tf.argmax(self.predc, 1), tf.float32)
        actuals = tf.cast(tf.argmax(self.c, 1), tf.float32)
        with tf.name_scope('err'):
            self.error = tf.reduce_mean(tf.cast(tf.not_equal(actuals, predictions), tf.float32))
        with tf.name_scope('cmseloss'):
            self.cmse = tf.reduce_mean(tf.abs(self.predd - self.d))
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)
        with tf.name_scope('TP'):
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('TN'):
            tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FP'):
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FN'):
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('acc'):
            accuracy = (tp + tn) / (tp + fp + fn + tn)
        with tf.name_scope('rec'):
            if tp + fn == tf.constant(0):
                recall = tf.constant(0)
            else:
                recall = tp / (tp + fn)
        with tf.name_scope('pre'):
            if tp + fp == tf.constant(0):
                precision = tf.constant(0)
            else:
                precision = tp / (tp + fp)

        tf.summary.scalar('cmse', self.cmse)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('recall', recall)
        tf.summary.scalar('precision', precision)
        return self.cmse, self.cmse


# multi-task compare handcraft rmse+cross
class MotionRNN3r(object):
    def __init__(self, ninput, noutput, nclasses, lstms, lstmo, R, H, L, k,
                 at=None, rnn='lstm', delta=1., alpha=1., isTest=False, isAtt=False):
        self._ninput = ninput
        self._noutput = noutput
        self._lstms = lstms
        self._lstmo = lstmo
        self._nclasses = nclasses
        self._R = R
        self._H = H
        self._L = L
        self._k = k
        self._delta = delta
        self._alpha = alpha
        self._isTest = isTest
        self._isAtt = isAtt

        if at == 'tanh':
            act = tf.nn.tanh
        elif at == 'relu':
            act = tf.nn.relu
        elif at == 'relu6':
            act = tf.nn.relu6
        elif at == 'crelu':
            act = tf.nn.crelu
        elif at == 'elu':
            act = tf.nn.elu
        elif at == 'softsign':
            act = tf.nn.softsign
        elif at == 'softplus':
            act = tf.nn.softplus
        elif at == 'sigmoid':
            act = tf.sigmoid
        elif at == 'selu':
            act = selu
        else:
            act = None

        with tf.variable_scope('inputs'):
            # tf Graph input layer
            bs = None
            self.x = tf.placeholder(tf.float32, [bs, self._lstms, self._ninput], name='x')  # input
            self.xp = tf.placeholder(tf.float32, [], name='xp')  # input
            self.y = tf.placeholder(tf.float32, [bs, self._lstmo, self._ninput], name='y')  # input
            self.xlastloc = tf.placeholder(tf.float32, [bs, self._noutput], name='last')  # input
            self.pos = tf.placeholder(tf.float32, [bs, self._noutput], name='pos')
            self.c = tf.placeholder(tf.float32, [bs, self._nclasses], name='cls')
            self.d = tf.placeholder(tf.float32, [bs, 1], name='cls')
            self._dropout = tf.placeholder(tf.float32, [], name='keep_prop')
            self._learningrate = tf.placeholder(tf.float32, [], name='lr')

        with tf.variable_scope('rnnx'):
            # define cells for rnn and dropout between layers
            x_fc = tf.contrib.layers.fully_connected(self.x, self._L, activation_fn=None)
            if rnn == 'lstm':
                cellx = tf.nn.rnn_cell.LSTMCell(self._L, activation=act)
            elif rnn == 'gru':
                cellx = tf.nn.rnn_cell.GRUCell(self._L, activation=act)
            else:
                cellx = tf.nn.rnn_cell.BasicRNNCell(self._L, activation=act)
            cellx = tf.nn.rnn_cell.ResidualWrapper(cellx)
            cellx = tf.nn.rnn_cell.DropoutWrapper(cellx, output_keep_prob=self._dropout)
            cellx = tf.nn.rnn_cell.MultiRNNCell([cellx] * self._R)

            # create rnn work with variance lengths
            x_all_output, self.x_last_state = tf.nn.dynamic_rnn(cellx, x_fc, dtype=tf.float32)
            if self._isAtt:
                x_output, self.alphas = attention(x_all_output, self._L, return_alphas=True)
            else:
                x_output = tf.contrib.layers.flatten(x_all_output)
            if self._R == 1:
                if rnn == 'lstm':
                    x_last_output = self.x_last_state[0][-1]
                else:
                    x_last_output = self.x_last_state[0]
            elif self._R == 2:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1]],
                                              1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1]], 1)
            else:
                if rnn == 'lstm':
                    x_last_output = tf.concat([self.x_last_state[0][-1], self.x_last_state[1][-1],
                                               self.x_last_state[2][-1]], 1)
                else:
                    x_last_output = tf.concat([self.x_last_state[0], self.x_last_state[1],
                                               self.x_last_state[2]], 1)

        with tf.variable_scope('fcy'):
            # define fully connected layer for t+1 displacement vector
            y_fc = tf.contrib.layers.fully_connected(self.y, self._L, activation_fn=None)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_fc = tf.contrib.layers.fully_connected(y_dp, self._L, activation_fn=act)
            y_dp = tf.contrib.layers.dropout(y_fc, self._dropout)
            y_output = tf.contrib.layers.flatten(y_dp)

        with tf.variable_scope('conca'):
            # concatenate two vectors x_output and y_output
            con = tf.concat([x_output, y_output], 1)
            conca = tf.contrib.layers.fully_connected(con, self._k, activation_fn=act)
            concadp = tf.contrib.layers.dropout(conca, self._dropout)

        with tf.variable_scope('predc'):
            self.pred = wxb_layer(concadp, self._k, self._nclasses, n_layer='c',
                                  activation_function=act)
            self.predc = tf.nn.softmax(self.pred)

        with tf.variable_scope('predy'):
            fc = tf.contrib.layers.fully_connected(x_last_output, self._k, activation_fn=None)
            dp = tf.contrib.layers.dropout(fc, self._dropout)
            self.predy = wxb_layer(dp, self._k, self._noutput, n_layer='y', truncate=False,
                                   activation_function=None)
            self.nextloc = tf.add(self.xlastloc, self.predy)

        with tf.name_scope('loss'):
            self.loss()

        with tf.name_scope('train'):
            # self.optimizer = tf.train.GradientDescentOptimizer(self._learningrate).minimize(self.total_error)
            self.optimizer = tf.train.AdamOptimizer(self._learningrate).minimize(self.total_error)
            tf.summary.scalar('learning_rate', self._learningrate)

        with tf.name_scope('prediction_power'):
            self.pred_power()

    def huberloss(self):
        x = tf.abs(tf.subtract(self.nextloc, self.pos))
        x = tf.where(x <= self._delta, 0.5 * tf.square(x), self._delta * (x - 0.5 * self._delta))
        return tf.reduce_sum(x)

    def loss(self):
        with tf.name_scope('RMSE'):
            # MSE(L2) to give us the loss of the regression
            # rmse = tf.reduce_sum(tf.squared_difference(self.predy, self.pos))
            # smoothL1 loss
            rmse = self.huberloss()
        with tf.name_scope('CROSS_ENTROPY'):
            # cross entropy to give us the loss of the classification
            # cross = -tf.reduce_sum(self.c * tf.log(self.predc))  # tf.clip_by_value(self.predc, 1e-10, 1.0)))
            cross = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.c, 1),
                                                               logits=self.pred))
        with tf.name_scope('TOTAL_LOSS'):
            # total error
            self.total_error = rmse + cross
        tf.summary.scalar('rmse_loss', rmse)
        tf.summary.scalar('cross_entropy_loss', cross)
        tf.summary.scalar('total_loss', self.total_error)
        return rmse, cross, rmse, self.total_error

    def pred_power(self):
        predictions = tf.cast(tf.argmax(self.predc, 1), tf.float32)
        actuals = tf.cast(tf.argmax(self.c, 1), tf.float32)
        with tf.name_scope('err'):
            self.error = tf.reduce_mean(tf.cast(tf.not_equal(actuals, predictions), tf.float32))
        with tf.name_scope('rmseloss'):
            self.rmse = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.squared_difference(self.nextloc, self.pos), axis=1)))
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)
        with tf.name_scope('TP'):
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('TN'):
            tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FP'):
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                      tf.equal(predictions, ones_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('FN'):
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                      tf.equal(predictions,
                                                               zeros_like_predictions)),
                                       dtype=tf.float32))
        with tf.name_scope('acc'):
            accuracy = (tp + tn) / (tp + fp + fn + tn)
        with tf.name_scope('rec'):
            if tp + fn == tf.constant(0):
                recall = tf.constant(0)
            else:
                recall = tp / (tp + fn)
        with tf.name_scope('pre'):
            if tp + fp == tf.constant(0):
                precision = tf.constant(0)
            else:
                precision = tp / (tp + fp)

        tf.summary.scalar('rmse', self.rmse)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('recall', recall)
        tf.summary.scalar('precision', precision)
        return self.rmse, self.rmse
