## config of RNN models
#
#

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class config(object):  # Network Parameters
    def __init__(self, scenario, lstms, lstmo, i, act, rnn, R, H, L, k, delta=0.5, alpha=1,
                 dropout=0.2,
                 bs=10000, epoch=1000, lr=1e-3):
        self.scenario = scenario
        self.lstms = lstms
        self.lstmo = lstmo
        self.act = act  # tanh, elu, softsign, relu, relu6, sigmoid, crelu, softplus
        self.rnn = rnn  # lstm, gru, rnn
        self.R = R  # multi RNN
        self.H = H  # layers
        self.L = L  # neurons lstm
        self.k = k  # neurons fc
        self.delta = delta
        self.alpha = alpha  # weight on the regression loss position estimation
        self.dropout = dropout
        self.batch_size = bs
        self.epoch = epoch

        self.n_classes = 2  # binary classifier

        self.maxlearning_rate = lr
        self.minlearning_rate = 1e-6
        self.decay_rate = 0.1
        self.learning_rate_decay_steps = 15
        self.display_step = 5
        self.no_of_batches = 0

        self.is_test = False
        self.save = False
        self.gpu = 0.7  # 0.2 ~ 0.75
        self.load_model = False  # Whether to load a saved model.
        self.feature = i
        self.fullname = scenario + 'S' + str(lstms) + 'H' + str(lstmo) + 'I' + str(i) + '_' + \
                        act + '_' + rnn + 'r' + str(R) + 'h' + str(H) + 'l' + str(L) + \
                        'k' + str(k) + 'dp' + str(int(dropout * 10)) + \
                        'al' + str(int(alpha * 100)) + 'dt' + str(int(delta * 10)) + \
                        'bs' + str(int(bs / 1000)) + 'ep' + str(int(epoch / 100))
        self.name = 'I' + str(i) + '_' + act + '_' + rnn + 'r' + str(R) + 'h' + str(H) + \
                    'l' + str(L) + 'k' + str(k) + 'dp' + str(int(dropout * 10)) + \
                    'al' + str(int(alpha * 100)) + 'dt' + str(int(delta * 10)) + \
                    'bs' + str(int(bs / 1000)) + 'ep' + str(int(epoch / 100))
        self.shortname = scenario + 'S' + str(lstms) + 'H' + str(lstmo) + 'I' + str(i)


class test_config(object):
    def __init__(self, scenario, lstms, lstmo, i, act, rnn, R, H, L, k, delta=0.5, alpha=1,
                 dropout=0.2,
                 bs=10000, epoch=1000, lr=1e-3):
        self.scenario = scenario
        self.lstms = lstms
        self.lstmo = lstmo
        self.act = act  # tanh, elu, softsign, relu, relu6, sigmoid, crelu, softplus
        self.rnn = rnn  # lstm, gru, rnn
        self.R = R  # multi RNN
        self.H = H  # layers
        self.L = L  # neurons lstm
        self.k = k  # neurons fc
        self.delta = delta
        self.alpha = alpha  # weight on the regression loss
        self.dropout = dropout
        self.batch_size = bs
        self.epoch = epoch

        self.n_classes = 2  # binary classifier

        self.maxlearning_rate = lr

        self.is_test = True
        self.save = False
        self.gpu = 0.4  # 0.3 ~ 0.75
        self.load_model = True  # Whether to load a saved model.
        self.feature = i
        self.fullname = scenario + 'S' + str(lstms) + 'H' + str(lstmo) + 'I' + str(i) + '_' + \
                        act + '_' + rnn + 'r' + str(R) + 'h' + str(H) + 'l' + str(L) + \
                        'k' + str(k) + 'dp' + str(int(dropout * 10)) + \
                        'al' + str(int(alpha * 100)) + 'dt' + str(int(delta * 10)) + \
                        'bs' + str(int(bs / 1000)) + 'ep' + str(int(epoch / 100))
        self.name = 'I' + str(i) + '_' + act + '_' + rnn + 'r' + str(R) + 'h' + str(H) + \
                    'l' + str(L) + 'k' + str(k) + 'dp' + str(int(dropout * 10)) + \
                    'al' + str(int(alpha * 100)) + 'dt' + str(int(delta * 10)) + \
                    'bs' + str(int(bs / 1000)) + 'ep' + str(int(epoch / 100))
        self.shortname = scenario + 'S' + str(lstms) + 'H' + str(lstmo) + 'I' + str(i)
