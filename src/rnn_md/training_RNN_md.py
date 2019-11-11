# ## LSTM with fully connected layers for predicting the position of a particle in the future frames
#
# based on the previous movements of a particle
#
#

import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, \
    precision_recall_curve
from rnn_md.model_config_RNN_md import config, test_config
from rnn_md.data_RNN_md import loadRNNdata
from rnn_md.model_RNN_md import MotionRNN, MotionCRNN, MotionRRNN, MotionRHRNN, MotionRHCNN, \
    MotionRNN1c, MotionRNN1r, MotionRNN2, MotionRNN3c, MotionRNN3r
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def training(nc, togit=1):
    # Training Parameters
    path_m = 'models/rnnMD/' + nc.scenario + '/model_' + nc.fullname
    path_log = 'logs/rnnMD/' + nc.scenario + '/log' + str(nc.lstms) \
               + '_H' + str(nc.lstmo) + '_I' + str(nc.feature)

    # load data
    train_x, train_xp, train_y, train_pos, train_c, train_l, train_d, \
    test_x, test_xp, test_y, test_pos, test_c, test_l, test_d = \
        loadRNNdata(nc.scenario, nc.lstms, nc.lstmo, nc.feature, True, togit)
    nc.no_of_batches = int(len(train_x) / nc.batch_size)
    no_of_batches_test = int(len(test_x) / nc.batch_size)
    # nc.no_of_batches, no_of_batches_test, _ = loadRNNdata_batchsize(nc.scenario, nc.lstms, nc.lstmo, togit)
    print('Training / validation data batches: ', nc.no_of_batches, no_of_batches_test)
    print('Epoch = ' + str(nc.epoch))
    print('batch size = ' + str(nc.batch_size))
    print('S+1 = ' + str(nc.lstms + 1))
    print('H = ' + str(nc.lstmo))
    print('i = ' + str(nc.feature))
    print('RNN = ' + nc.rnn)
    print('act = ' + nc.act)
    print('r = ' + str(nc.R))
    print('h = ' + str(nc.H))
    print('L = ' + str(nc.L))
    print('k = ' + str(nc.k))
    print('alpha = ' + str(nc.alpha))
    print('delta = ' + str(nc.delta))
    print('dropout = ' + str(nc.dropout))

    l_old = 1e10
    steps_stop = 0
    decay_init = int(nc.learning_rate_decay_steps / 2)
    # Make a path for our model to be saved in.
    if not os.path.exists(path_m):
        os.makedirs(path_m)
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    if not os.path.exists(path_m + '_final'):
        os.makedirs(path_m + '_final')

    # reset and initializing
    tf.reset_default_graph()
    if nc.feature < 6:
        if nc.scenario == 'VIRUS':
            n_input = 3
            n_output = 3
        else:
            n_input = 2
            n_output = 2
    else:
        if nc.scenario == 'VIRUS':
            n_input = 13
            n_output = 3
            n_inputp = 3
        else:
            n_input = 8
            n_output = 2
            n_inputp = 2

    if nc.feature == 0:  # RNN learned
        motion = MotionRRNN(n_input, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R, nc.H, nc.L,
                            nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, False)
    elif nc.feature == 1:
        motion = MotionRRNN(n_input, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R, nc.H, nc.L,
                            nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, True)
    elif nc.feature == 2:  # CNN learned
        motion = MotionCRNN(n_input, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R, nc.H, nc.L,
                            nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, False)
    elif nc.feature == 3:
        motion = MotionCRNN(n_input, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R, nc.H, nc.L,
                            nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, True)
    elif nc.feature == 4:  # position only no handcraft or learned
        motion = MotionRNN(n_input, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R, nc.H, nc.L,
                           nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, False)
    elif nc.feature == 5:
        motion = MotionRNN(n_input, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R, nc.H, nc.L,
                           nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, True)
    elif nc.feature == 6:  # handcraft
        motion = MotionRNN(n_input, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R, nc.H, nc.L,
                           nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, False)
    elif nc.feature == 7:
        motion = MotionRNN(n_input, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R, nc.H, nc.L,
                           nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, True)
    elif nc.feature == 8:  # combined RNN learned with handcraft
        motion = MotionRHRNN(n_input, n_inputp, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R,
                             nc.H, nc.L, nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, False)
    elif nc.feature == 9:
        motion = MotionRHRNN(n_input, n_inputp, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R,
                             nc.H, nc.L, nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, True)
    elif nc.feature == 10:  # combined CNN learned with handcraft
        motion = MotionRHCNN(n_input, n_inputp, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R,
                             nc.H, nc.L, nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, False)
    elif nc.feature == 11:
        motion = MotionRHCNN(n_input, n_inputp, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R,
                             nc.H, nc.L, nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, True)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=nc.gpu)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()
    merged = tf.summary.merge_all()
    train_log_folder = path_log + '/train_' + nc.name
    test_log_folder = path_log + '/val_' + nc.name
    model_log_folder = path_log + '/model_' + nc.name
    if os.path.exists(train_log_folder):
        subprocess.call(['rm', '-rf', train_log_folder])
    if os.path.exists(test_log_folder):
        subprocess.call(['rm', '-rf', test_log_folder])
    if nc.save:
        if os.path.exists(model_log_folder):
            subprocess.call(['rm', '-rf', model_log_folder])
        if not os.path.exists(model_log_folder):
            os.makedirs(model_log_folder)

    train_writer = tf.summary.FileWriter(train_log_folder, sess.graph)
    test_writer = tf.summary.FileWriter(test_log_folder)
    saver = tf.train.Saver()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    # load model from file
    if nc.load_model:
        if os.path.exists(path_m + '/model.index'):
            print('Loading Model...')
            # saver.restore(sess, path_m + '/model')
        else:
            print('Model not exist! Create new!')

    # Training iterations
    i = 1
    lr = nc.maxlearning_rate
    while i <=nc.epoch:
    #while i <= nc.epoch or (i <= nc.epoch * (nc.lstmo + 1) and lr >= nc.minlearning_rate):
        for k in range(nc.no_of_batches):
            ptr = k * nc.batch_size
            if nc.feature < 4:
                tx, tpos, tc, tl, td = train_x[ptr:ptr + nc.batch_size], \
                                       train_pos[ptr:ptr + nc.batch_size], \
                                       train_c[ptr:ptr + nc.batch_size], \
                                       train_l[ptr:ptr + nc.batch_size], \
                                       train_d[ptr:ptr + nc.batch_size]
                ty, txp = 1, 1
            elif nc.feature >= 8:
                tx, txp, ty, tpos, tc, tl, td = train_x[ptr:ptr + nc.batch_size], \
                                                train_xp[ptr:ptr + nc.batch_size], \
                                                train_y[ptr:ptr + nc.batch_size], \
                                                train_pos[ptr:ptr + nc.batch_size], \
                                                train_c[ptr:ptr + nc.batch_size], \
                                                train_l[ptr:ptr + nc.batch_size], \
                                                train_d[ptr:ptr + nc.batch_size]
            else:
                tx, ty, tpos, tc, tl, td = train_x[ptr:ptr + nc.batch_size], \
                                           train_y[ptr:ptr + nc.batch_size], \
                                           train_pos[ptr:ptr + nc.batch_size], \
                                           train_c[ptr:ptr + nc.batch_size], \
                                           train_l[ptr:ptr + nc.batch_size], \
                                           train_d[ptr:ptr + nc.batch_size]
                txp = 1
            sess.run(motion.optimizer,
                     feed_dict={
                         motion.x: tx, motion.xp: txp, motion.y: ty, motion.pos: tpos,
                         motion.xlastloc: tl, motion.c: tc, motion.d: td,
                         motion._dropout: nc.dropout, motion._learningrate: lr
                     })

        # logs for tensorboard
        result = sess.run(merged,
                          feed_dict={
                              motion.x: tx, motion.xp: txp, motion.y: ty, motion.pos: tpos,
                              motion.xlastloc: tl, motion.c: tc, motion.d: td,
                              motion._dropout: nc.dropout, motion._learningrate: lr
                          })
        train_writer.add_summary(result, i)

        # validation, output performance
        if i % nc.display_step == 0 and i != 0:
            rl = 0
            cl = 0
            sl = 0
            l = 0
            cmse = 0
            mse = 0
            acc = 0
            rec = 0
            pre = 0
            auc = 0
            apr = 0
            for j in range(no_of_batches_test):
                ptr = j * nc.batch_size
                if nc.feature < 4:
                    ttx, ttpos, ttc, ttl, ttd = test_x[ptr:ptr + nc.batch_size], \
                                                test_pos[ptr:ptr + nc.batch_size], \
                                                test_c[ptr:ptr + nc.batch_size], \
                                                test_l[ptr:ptr + nc.batch_size], \
                                                test_d[ptr:ptr + nc.batch_size]
                    tty, ttxp = 1, 1
                elif nc.feature >= 8:
                    ttx, ttxp, tty, ttpos, ttc, ttl, ttd = test_x[ptr:ptr + nc.batch_size], \
                                                           test_xp[ptr:ptr + nc.batch_size], \
                                                           test_y[ptr:ptr + nc.batch_size], \
                                                           test_pos[ptr:ptr + nc.batch_size], \
                                                           test_c[ptr:ptr + nc.batch_size], \
                                                           test_l[ptr:ptr + nc.batch_size], \
                                                           test_d[ptr:ptr + nc.batch_size]
                else:
                    ttx, tty, ttpos, ttc, ttl, ttd = test_x[ptr:ptr + nc.batch_size], \
                                                     test_y[ptr:ptr + nc.batch_size], \
                                                     test_pos[ptr:ptr + nc.batch_size], \
                                                     test_c[ptr:ptr + nc.batch_size], \
                                                     test_l[ptr:ptr + nc.batch_size], \
                                                     test_d[ptr:ptr + nc.batch_size]
                    ttxp = 1
                reg_loss, cls_loss, cose_loss, loss = sess.run(motion.loss(),
                                                               feed_dict={
                                                                   motion.x: ttx,
                                                                   motion.xp: ttxp,
                                                                   motion.y: tty,
                                                                   motion.pos: ttpos,
                                                                   motion.xlastloc: ttl,
                                                                   motion.c: ttc,
                                                                   motion.d: ttd,
                                                                   motion._dropout: 1,
                                                                   motion._learningrate: lr
                                                               })
                err, macc, mrec, mpre, mmse = sess.run(motion.pred_power(),
                                                       feed_dict={
                                                           motion.x: ttx, motion.xp: ttxp,
                                                           motion.y: tty,
                                                           motion.pos: ttpos,
                                                           motion.xlastloc: ttl,
                                                           motion.c: ttc, motion.d: ttd,
                                                           motion._dropout: 1,
                                                           motion._learningrate: lr
                                                       })
                # ttpredc = sess.run(motion.predc, feed_dict={motion.x: ttx, motion.xp: ttxp,
                #                                            motion.y: tty,
                #                                            motion.pos: ttpos,
                #                                            motion.xlastloc: ttl, motion.c: ttc,
                #                                            motion.d: ttd, motion._dropout: 1,
                #                                            motion._learningrate: lr})

                # aauc = roc_auc_score(ttc[:, 1], ttpredc[:, 1])
                # roc_auc_score(np.argmax(ttc, 1), np.argmax(ttpredc, 1))
                # aapr = average_precision_score(ttc[:, 1], ttpredc[:, 1])

                rl += reg_loss
                cl += cls_loss
                sl += cose_loss
                l += loss
                cmse += err
                acc += macc
                rec += mrec
                pre += mpre
                mse += mmse
                # auc += aauc
                # apr += aapr

            rl /= no_of_batches_test
            cl /= no_of_batches_test
            sl /= no_of_batches_test
            l /= no_of_batches_test
            mse /= no_of_batches_test
            cmse /= no_of_batches_test
            acc /= no_of_batches_test
            rec /= no_of_batches_test
            pre /= no_of_batches_test
            # auc /= no_of_batches_test
            # apr /= no_of_batches_test

            # logs for tensorboard
            valresult = sess.run(merged, feed_dict={
                motion.x: ttx, motion.xp: ttxp, motion.y: tty,
                motion.pos: ttpos,
                motion.xlastloc: ttl, motion.c: ttc,
                motion.d: ttd,
                motion._dropout: 1, motion._learningrate: lr
            })
            test_writer.add_summary(valresult, i)

            print(nc.shortname + 'Iter ' + str(i) + ' Loss= {:.6f}'.format(l) +
                  ' CROSS= {:.6f}'.format(cl) +
                  ' CMSE= {:.6f}'.format(sl) +
                  ' RMSE= {:.6f}'.format(rl) +
                  ' valAccu= {:.4f}'.format(acc * 100) +
                  ' valPrec= {:.4f}'.format(pre * 100) +
                  ' valReca= {:.4f}'.format(rec * 100) +
                  # ' valAUC= {:.6f}'.format(auc) +
                  # ' valAPR= {:.6f}'.format(apr) +
                  ' valRMSE= {:.6f}'.format(mse) +
                  ' valCMSE= {:.6f}'.format(cmse) +
                  ' lr= ' + str(lr))

            if l <= l_old:  # and rec >= rec_old:
                l_old = l
                rec_old = rec
                steps_stop = 0
                saver.save(sess, path_m + '/model')
                print('Better model saved!')
                if nc.save:  # Periodically save the model
                    saver.save(sess, model_log_folder + '/model', i)
            else:
                steps_stop += 1
                if i != 0 and steps_stop != 0 and steps_stop % nc.learning_rate_decay_steps == 0 \
                        and lr > nc.minlearning_rate:
                    lr = nc.decay_rate * lr
                    nc.learning_rate_decay_steps += decay_init
                    print(steps_stop, lr)
        i += 1
    saver.save(sess, path_m + '_final/model')
    sess.close()
    print('Optimization Finished!')
    os.system('curl -s -X POST https://api.telegram.org/'
              'bot459863485:AAEPUJsNI0wkf3iI8RfweGMR9u0rKiSHvuc/'
              'sendMessage -F chat_id=385302225 -F text='
              + nc.fullname + ' finished.\n')


def testing(tc, togit=1):
    # Training Parameters
    path_m = 'models/rnnMD/' + tc.scenario + '/model_' + tc.fullname
    path_log = 'logs/rnnMD/' + tc.scenario + '/log' + str(tc.lstms) \
               + '_H' + str(tc.lstmo) + '_I' + str(tc.feature)
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    # load data
    test_x, test_xp, test_y, test_pos, test_c, test_l, test_d = loadRNNdata(tc.scenario, tc.lstms,
                                                                            tc.lstmo, tc.feature,
                                                                            False, togit)
    no_of_batches_test = int(len(test_x) / tc.batch_size)
    # _, _, no_of_batches_test = loadRNNdata_batchsize(tc.scenario, tc.lstms, tc.lstmo, togit)
    # print('Validation data batches: ', no_of_batches_test)

    # Make a path for our model to be saved in.
    for isbest in [True, False]:

        if not isbest:
            path_m = path_m + '_final'
            stt = 'Last'
        else:
            stt = 'Best'

        if not tc.load_model or not os.path.exists(path_m + '/model.index'):
            print('No model in ' + path_m)
        else:
            # reset and initializing
            tf.reset_default_graph()
            if tc.feature < 6:
                if tc.scenario == 'VIRUS':
                    n_input = 3
                    n_output = 3
                else:
                    n_input = 2
                    n_output = 2
            else:
                if tc.scenario == 'VIRUS':
                    n_input = 13
                    n_output = 3
                    n_inputp = 3
                else:
                    n_input = 8
                    n_output = 2
                    n_inputp = 2

            if tc.feature == 0:
                motion = MotionRRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H,
                                    tc.L, tc.k, tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
            elif tc.feature == 1:
                motion = MotionRRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H,
                                    tc.L, tc.k, tc.act, tc.rnn, tc.delta, tc.alpha, True, True)
            elif tc.feature == 2:
                motion = MotionCRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H,
                                    tc.L, tc.k, tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
            elif tc.feature == 3:
                motion = MotionCRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H,
                                    tc.L, tc.k, tc.act, tc.rnn, tc.delta, tc.alpha, True, True)
            elif tc.feature == 4:
                motion = MotionRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H,
                                   tc.L, tc.k, tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
            elif tc.feature == 5:
                motion = MotionRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H,
                                   tc.L, tc.k, tc.act, tc.rnn, tc.delta, tc.alpha, True, True)
            elif tc.feature == 6:
                motion = MotionRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H,
                                   tc.L, tc.k, tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
            elif tc.feature == 7:
                motion = MotionRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H,
                                   tc.L, tc.k, tc.act, tc.rnn, tc.delta, tc.alpha, True, True)
            elif tc.feature == 8:
                motion = MotionRHRNN(n_input, n_inputp, n_output, tc.n_classes, tc.lstms, tc.lstmo,
                                     tc.R, tc.H, tc.L, tc.k,
                                     tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
            elif tc.feature == 9:
                motion = MotionRHRNN(n_input, n_inputp, n_output, tc.n_classes, tc.lstms, tc.lstmo,
                                     tc.R, tc.H, tc.L, tc.k,
                                     tc.act, tc.rnn, tc.delta, tc.alpha, True, True)
            elif tc.feature == 10:
                motion = MotionRHCNN(n_input, n_inputp, n_output, tc.n_classes, tc.lstms, tc.lstmo,
                                     tc.R, tc.H, tc.L, tc.k,
                                     tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
            elif tc.feature == 11:
                motion = MotionRHCNN(n_input, n_inputp, n_output, tc.n_classes, tc.lstms, tc.lstmo,
                                     tc.R, tc.H, tc.L, tc.k,
                                     tc.act, tc.rnn, tc.delta, tc.alpha, True, True)

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=tc.gpu)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess.run(init)

            # print('Loading Model...')
            saver.restore(sess, path_m + '/model')

            # batch testing
            rl = 0
            cl = 0
            sl = 0
            l = 0
            cmse = 0
            mse = 0
            acc = 0
            rec = 0
            pre = 0
            test_p_pred = []
            test_p = []
            for j in range(no_of_batches_test):
                ptr = j * tc.batch_size
                if tc.feature < 4:
                    ttx, ttpos, ttc, ttl, ttd = test_x[ptr:ptr + tc.batch_size], \
                                                test_pos[ptr:ptr + tc.batch_size], \
                                                test_c[ptr:ptr + tc.batch_size], \
                                                test_l[ptr:ptr + tc.batch_size], \
                                                test_d[ptr:ptr + tc.batch_size]
                    tty, ttxp = 1, 1
                elif tc.feature >= 8:
                    ttx, ttxp, tty, ttpos, ttc, ttl, ttd = test_x[ptr:ptr + tc.batch_size], \
                                                           test_xp[ptr:ptr + tc.batch_size], \
                                                           test_y[ptr:ptr + tc.batch_size], \
                                                           test_pos[ptr:ptr + tc.batch_size], \
                                                           test_c[ptr:ptr + tc.batch_size], \
                                                           test_l[ptr:ptr + tc.batch_size], \
                                                           test_d[ptr:ptr + tc.batch_size]
                else:
                    ttx, tty, ttpos, ttc, ttl, ttd = test_x[ptr:ptr + tc.batch_size], \
                                                     test_y[ptr:ptr + tc.batch_size], \
                                                     test_pos[ptr:ptr + tc.batch_size], \
                                                     test_c[ptr:ptr + tc.batch_size], \
                                                     test_l[ptr:ptr + tc.batch_size], \
                                                     test_d[ptr:ptr + tc.batch_size]
                    ttxp = 1

                reg_loss, cls_loss, cost_loss, loss = sess.run(motion.loss(),
                                                               feed_dict={
                                                                   motion.x: ttx,
                                                                   motion.xp: ttxp,
                                                                   motion.y: tty,
                                                                   motion.pos: ttpos,
                                                                   motion.c: ttc,
                                                                   motion.d: ttd,
                                                                   motion.xlastloc: ttl,
                                                                   motion._dropout: 1
                                                               })
                mcmse, macc, mrec, mpre, mmse = sess.run(motion.pred_power(),
                                                         feed_dict={
                                                             motion.x: ttx, motion.xp: ttxp,
                                                             motion.y: tty,
                                                             motion.pos: ttpos,
                                                             motion.c: ttc,
                                                             motion.d: ttd,
                                                             motion.xlastloc: ttl,
                                                             motion._dropout: 1
                                                         })
                ttpredc = sess.run(motion.predc,
                                   feed_dict={
                                       motion.x: ttx, motion.xp: ttxp, motion.y: tty,
                                       motion.pos: ttpos, motion.xlastloc: ttl,
                                       motion.c: ttc, motion.d: ttd, motion._dropout: 1
                                   })
                test_p.extend(ttc)
                test_p_pred.extend(ttpredc)

                rl += reg_loss
                cl += cls_loss
                sl += cost_loss
                l += loss
                cmse += mcmse
                acc += macc
                rec += mrec
                pre += mpre
                mse += mmse

            rl /= no_of_batches_test
            cl /= no_of_batches_test
            sl /= no_of_batches_test
            l /= no_of_batches_test
            mse /= no_of_batches_test
            cmse /= no_of_batches_test
            acc /= no_of_batches_test
            rec /= no_of_batches_test
            pre /= no_of_batches_test
            test_p = np.asanyarray(test_p)
            test_p_pred = np.asanyarray(test_p_pred)
            #auc = roc_auc_score(test_p[:, 1], test_p_pred[:, 1])
            #apr = average_precision_score(test_p[:, 1], test_p_pred[:, 1])

            result = (tc.shortname + stt + ' Test Accuracy= {:.4f}'.format(acc * 100) +
                      ' loss= ' + '{:.4f}'.format(l) +
                      ' Test Precision= ' + '{:.4f}'.format(pre * 100) +
                      ' Test Recall= ' + '{:.4f}'.format(rec * 100) +
                      ' Rloss= ' + '{:.4f}'.format(rl) +
                      ' CNloss= ' + '{:.4f}'.format(cl) +
                      ' CSloss= ' + '{:.4f}'.format(sl) +
                      #' Test AUC= {:.6f}'.format(auc) +
                      #' Test APR= {:.6f}'.format(apr) +
                      ' Test RMSE= ' + '{:.6f}'.format(mse) +
                      ' Test CMSE= ' + '{:.6f}'.format(cmse))

            sess.close()

            print(result)
            # print('Test Finished!')
            '''
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], thr = roc_curve(test_p[:, i], test_p_pred[:, i])
                roc_auc[i] = roc_auc_score(test_p[:, i], test_p_pred[:, i])
            plt.figure()
            plt.plot(fpr[1], tpr[1], color='darkorange', lw=2,
                     label='ROC curve (area = %0.2f)' % roc_auc[1])
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC of ' + tc.fullname)
            plt.legend(loc="lower right")
            plt.savefig(path_log + '/' + tc.fullname + stt + '.png')
            plt.clf()
            precision, recall, _ = precision_recall_curve(test_p[:, 1], test_p_pred[:, 1])
            plt.figure()
            plt.step(recall, precision, color='b', alpha=0.2, where='post')
            plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('2-class PRC: AP={0:0.2f}'.format(apr))
            # plt.show()
            plt.savefig(path_log + '/PRC' + tc.fullname + stt + '.png')
            plt.clf()
            # os.system('curl -s -X POST https://api.telegram.org/'
            #          'bot459863485:AAEPUJsNI0wkf3iI8RfweGMR9u0rKiSHvuc/'
            #          'sendMessage -F chat_id=385302225 -F text='
            #          + tc.fullname + '\n' + result)
            '''

def runRNN_md_train(scenarios, ss, oo, ff, acts, rnns, rr, hh, ll, kk,
                    delta, alpha, dropout, batch_size, epoch, lr, togit=1):
    for ft in ff:
        for st in ss:
            for ot in oo:
                for scenario in scenarios:
                    for act in acts:
                        for rnn in rnns:
                            for r in rr:
                                for h in hh:
                                    for l, k in zip(ll, kk):
                                        print('----------------------------------------')
                                        nc = config(scenario, st, ot, ft, act, rnn, r, h, l, k,
                                                    delta, alpha, dropout, batch_size, epoch, lr)
                                        stritem = 'Training ' + 'RNN ' + nc.fullname
                                        print(stritem)
                                        start = time.time()
                                        training(nc, togit)
                                        # tc = test_config(scenario, st, ot, ft, act, rnn, H, L, k,
                                        #                 delta, alpha, dropout, batch_size, epoch, lr)
                                        # testing(tc, togit)
                                        print('Time : ', time.time() - start)


def runRNN_md_test(scenarios, ss, oo, ff, acts, rnns, rr, hh, ll, kk,
                   delta, alpha, dropout, batch_size, epoch, lr, togit=1):
    for ft in ff:
        for st in ss:
            for ot in oo:
                for scenario in scenarios:
                    for act in acts:
                        for rnn in rnns:
                            for r in rr:
                                for h in hh:
                                    for l, k in zip(ll, kk):
                                        print('----------------------------------------')
                                        tc = test_config(scenario, st, ot, ft, act, rnn, r, h, l, k,
                                                         delta, alpha, dropout, batch_size, epoch, lr)
                                        stritem = 'Testing ' + 'RNN ' + tc.fullname
                                        print(stritem)
                                        start = time.time()
                                        testing(tc, togit)
                                        print('Time : ', time.time() - start)


def training_multi(nc, togit=1):
    # Training Parameters
    path_m = 'models_m/rnnMD/' + nc.scenario + '/model_' + nc.fullname
    path_log = 'logs_m/rnnMD/' + nc.scenario + '/log' + str(nc.lstms) \
               + '_H' + str(nc.lstmo) + '_I' + str(nc.feature)

    # load data
    train_x, train_xp, train_y, train_pos, train_c, train_l, train_d, \
    test_x, test_xp, test_y, test_pos, test_c, test_l, test_d = \
        loadRNNdata(nc.scenario, nc.lstms, nc.lstmo, nc.feature, True, togit)
    nc.no_of_batches = int(len(train_x) / nc.batch_size)
    no_of_batches_test = int(len(test_x) / nc.batch_size)
    # nc.no_of_batches, no_of_batches_test, _ = loadRNNdata_batchsize(nc.scenario, nc.lstms, nc.lstmo, togit)
    print('Training / validation data batches: ', nc.no_of_batches, no_of_batches_test)
    print('Epoch = ' + str(nc.epoch))
    print('batch size = ' + str(nc.batch_size))
    print('S+1 = ' + str(nc.lstms + 1))
    print('H = ' + str(nc.lstmo))
    print('i = ' + str(nc.feature))
    print('RNN = ' + nc.rnn)
    print('act = ' + nc.act)
    print('r = ' + str(nc.R))
    print('h = ' + str(nc.H))
    print('L = ' + str(nc.L))
    print('k = ' + str(nc.k))
    print('alpha = ' + str(nc.alpha))
    print('delta = ' + str(nc.delta))
    print('dropout = ' + str(nc.dropout))

    l_old = 1e10
    rec_old = 0
    steps_stop = 0
    decay_init = int(nc.learning_rate_decay_steps / 2)
    # Make a path for our model to be saved in.
    if not os.path.exists(path_m):
        os.makedirs(path_m)
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    if not os.path.exists(path_m + '_final'):
        os.makedirs(path_m + '_final')

    # reset and initializing
    tf.reset_default_graph()
    if nc.feature < 6:
        if nc.scenario == 'VIRUS':
            n_input = 3
            n_output = 3
        else:
            n_input = 2
            n_output = 2
    else:
        if nc.scenario == 'VIRUS':
            n_input = 13
            n_output = 3
            n_inputp = 3
        else:
            n_input = 8
            n_output = 2
            n_inputp = 2

    if nc.feature == 210:  # handcraft cmse
        motion = MotionRNN1c(n_input, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R, nc.H, nc.L,
                             nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, False)
    elif nc.feature == 211:  # handcraft rmse
        motion = MotionRNN1r(n_input, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R, nc.H, nc.L,
                             nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, False)
    elif nc.feature == 22:  # handcraft cmse+rmse
        motion = MotionRNN2(n_input, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R, nc.H, nc.L,
                            nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, False)
    elif nc.feature == 230:  # handcraft cmse+cross
        motion = MotionRNN3c(n_input, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R, nc.H, nc.L,
                             nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, False)
    elif nc.feature == 231:  # handcraft rmse+cross
        motion = MotionRNN3r(n_input, n_output, nc.n_classes, nc.lstms, nc.lstmo, nc.R, nc.H, nc.L,
                             nc.k, nc.act, nc.rnn, nc.delta, nc.alpha, False, False)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=nc.gpu)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()
    merged = tf.summary.merge_all()
    train_log_folder = path_log + '/train_' + nc.name
    test_log_folder = path_log + '/val_' + nc.name
    model_log_folder = path_log + '/model_' + nc.name
    if os.path.exists(train_log_folder):
        subprocess.call(['rm', '-rf', train_log_folder])
    if os.path.exists(test_log_folder):
        subprocess.call(['rm', '-rf', test_log_folder])
    if nc.save:
        if os.path.exists(model_log_folder):
            subprocess.call(['rm', '-rf', model_log_folder])
        if not os.path.exists(model_log_folder):
            os.makedirs(model_log_folder)

    train_writer = tf.summary.FileWriter(train_log_folder, sess.graph)
    test_writer = tf.summary.FileWriter(test_log_folder)
    saver = tf.train.Saver()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    # load model from file
    if nc.load_model:
        if os.path.exists(path_m + '/model.index'):
            print('Loading Model...')
            saver.restore(sess, path_m + '/model')
        else:
            print('Model not exist! Create new!')

    # Training iterations
    i = 1
    lr = nc.maxlearning_rate
    while i <= nc.epoch:
    #while i <= nc.epoch or (i <= nc.epoch * (nc.lstmo + 1) and lr >= nc.minlearning_rate):
        for k in range(nc.no_of_batches):
            ptr = k * nc.batch_size
            if nc.feature < 4:
                tx, tpos, tc, tl, td = train_x[ptr:ptr + nc.batch_size], \
                                       train_pos[ptr:ptr + nc.batch_size], \
                                       train_c[ptr:ptr + nc.batch_size], \
                                       train_l[ptr:ptr + nc.batch_size], \
                                       train_d[ptr:ptr + nc.batch_size]
                ty, txp = 1, 1
            elif 11 >= nc.feature >= 8:
                tx, txp, ty, tpos, tc, tl, td = train_x[ptr:ptr + nc.batch_size], \
                                                train_xp[ptr:ptr + nc.batch_size], \
                                                train_y[ptr:ptr + nc.batch_size], \
                                                train_pos[ptr:ptr + nc.batch_size], \
                                                train_c[ptr:ptr + nc.batch_size], \
                                                train_l[ptr:ptr + nc.batch_size], \
                                                train_d[ptr:ptr + nc.batch_size]
            else:
                tx, ty, tpos, tc, tl, td = train_x[ptr:ptr + nc.batch_size], \
                                           train_y[ptr:ptr + nc.batch_size], \
                                           train_pos[ptr:ptr + nc.batch_size], \
                                           train_c[ptr:ptr + nc.batch_size], \
                                           train_l[ptr:ptr + nc.batch_size], \
                                           train_d[ptr:ptr + nc.batch_size]
                txp = 1
            sess.run(motion.optimizer,
                     feed_dict={
                         motion.x: tx, motion.xp: txp, motion.y: ty, motion.pos: tpos,
                         motion.xlastloc: tl, motion.c: tc, motion.d: td,
                         motion._dropout: nc.dropout, motion._learningrate: lr
                     })

        # logs for tensorboard
        result = sess.run(merged,
                          feed_dict={
                              motion.x: tx, motion.xp: txp, motion.y: ty, motion.pos: tpos,
                              motion.xlastloc: tl, motion.c: tc, motion.d: td,
                              motion._dropout: nc.dropout, motion._learningrate: lr
                          })

        train_writer.add_summary(result, i)

        # validation, output performance
        if i % nc.display_step == 0 and i != 0:
            rl = 0
            cl = 0
            sl = 0
            l = 0
            cmse = 0
            mse = 0
            for j in range(no_of_batches_test):
                ptr = j * nc.batch_size
                if nc.feature < 4:
                    ttx, ttpos, ttc, ttl, ttd = test_x[ptr:ptr + nc.batch_size], \
                                                test_pos[ptr:ptr + nc.batch_size], \
                                                test_c[ptr:ptr + nc.batch_size], \
                                                test_l[ptr:ptr + nc.batch_size], \
                                                test_d[ptr:ptr + nc.batch_size]
                    tty, ttxp = 1, 1
                elif 11 >= nc.feature >= 8:
                    ttx, ttxp, tty, ttpos, ttc, ttl, ttd = test_x[ptr:ptr + nc.batch_size], \
                                                           test_xp[ptr:ptr + nc.batch_size], \
                                                           test_y[ptr:ptr + nc.batch_size], \
                                                           test_pos[ptr:ptr + nc.batch_size], \
                                                           test_c[ptr:ptr + nc.batch_size], \
                                                           test_l[ptr:ptr + nc.batch_size], \
                                                           test_d[ptr:ptr + nc.batch_size]
                else:
                    ttx, tty, ttpos, ttc, ttl, ttd = test_x[ptr:ptr + nc.batch_size], \
                                                     test_y[ptr:ptr + nc.batch_size], \
                                                     test_pos[ptr:ptr + nc.batch_size], \
                                                     test_c[ptr:ptr + nc.batch_size], \
                                                     test_l[ptr:ptr + nc.batch_size], \
                                                     test_d[ptr:ptr + nc.batch_size]
                    ttxp = 1
                reg_loss, cls_loss, cose_loss, loss = sess.run(motion.loss(),
                                                               feed_dict={
                                                                   motion.x: ttx,
                                                                   motion.xp: ttxp,
                                                                   motion.y: tty,
                                                                   motion.pos: ttpos,
                                                                   motion.xlastloc: ttl,
                                                                   motion.c: ttc,
                                                                   motion.d: ttd,
                                                                   motion._dropout: 1,
                                                                   motion._learningrate: lr
                                                               })
                mcmse, mrmse = sess.run(motion.pred_power(),
                                        feed_dict={
                                            motion.x: ttx, motion.xp: ttxp,
                                            motion.y: tty,
                                            motion.pos: ttpos,
                                            motion.xlastloc: ttl,
                                            motion.c: ttc, motion.d: ttd,
                                            motion._dropout: 1,
                                            motion._learningrate: lr
                                        })

                rl += reg_loss
                cl += cls_loss
                sl += cose_loss
                l += loss
                cmse += mcmse
                mse += mrmse

            rl /= no_of_batches_test
            cl /= no_of_batches_test
            sl /= no_of_batches_test
            l /= no_of_batches_test
            mse /= no_of_batches_test
            cmse /= no_of_batches_test

            # logs for tensorboard
            valresult = sess.run(merged, feed_dict={
                motion.x: ttx, motion.xp: ttxp, motion.y: tty,
                motion.pos: ttpos,
                motion.xlastloc: ttl, motion.c: ttc,
                motion.d: ttd,
                motion._dropout: 1, motion._learningrate: lr
            })

            test_writer.add_summary(valresult, i)

            print(nc.shortname + 'Iter ' + str(i) + ' Loss= {:.6f}'.format(l) +
                  ' CROSS= {:.6f}'.format(cl) +
                  ' CMSE= {:.6f}'.format(sl) +
                  ' RMSE= {:.6f}'.format(rl) +
                  ' valRMSE= {:.6f}'.format(mse) +
                  ' valCMSE= {:.6f}'.format(cmse) +
                  ' lr= ' + str(lr))

            if l <= l_old:
                l_old = l
                steps_stop = 0
                saver.save(sess, path_m + '/model')
                print('Better model saved!')
                if nc.save:  # Periodically save the model
                    saver.save(sess, model_log_folder + '/model', i)
            else:
                steps_stop += 1
                if i != 0 and steps_stop != 0 and steps_stop % nc.learning_rate_decay_steps == 0 \
                        and lr > nc.minlearning_rate:
                    lr = nc.decay_rate * lr
                    nc.learning_rate_decay_steps += decay_init
                    print(steps_stop, lr)
        i += 1
    saver.save(sess, path_m + '_final/model')
    sess.close()
    print('Optimization Finished!')


def testing_multi(tc, togit=1):
    # Training Parameters
    path_m = 'models_m/rnnMD/' + tc.scenario + '/model_' + tc.fullname
    path_log = 'logs_m/rnnMD/' + tc.scenario + '/log' + str(tc.lstms) \
               + '_H' + str(tc.lstmo) + '_I' + str(tc.feature)
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    # load data
    test_x, test_xp, test_y, test_pos, test_c, test_l, test_d = loadRNNdata(tc.scenario, tc.lstms,
                                                                            tc.lstmo, tc.feature,
                                                                            False, togit)
    no_of_batches_test = int(len(test_x) / tc.batch_size)
    # _, _, no_of_batches_test = loadRNNdata_batchsize(tc.scenario, tc.lstms, tc.lstmo, togit)
    # print('Validation data batches: ', no_of_batches_test)

    # Make a path for our model to be saved in.
    for isbest in [True, False]:

        if not isbest:
            path_m = path_m + '_final'
            stt = 'Last'
        else:
            stt = 'Best'

        if not tc.load_model or not os.path.exists(path_m + '/model.index'):
            print('No model in ' + path_m)
        else:
            # reset and initializing
            tf.reset_default_graph()
            if tc.feature < 6:
                if tc.scenario == 'VIRUS':
                    n_input = 3
                    n_output = 3
                else:
                    n_input = 2
                    n_output = 2
            else:
                if tc.scenario == 'VIRUS':
                    n_input = 13
                    n_output = 3
                    n_inputp = 3
                else:
                    n_input = 8
                    n_output = 2
                    n_inputp = 2

            if tc.feature == 210:
                motion = MotionRNN1c(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R,
                                     tc.H, tc.L, tc.k,
                                     tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
            elif tc.feature == 211:
                motion = MotionRNN1r(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R,
                                     tc.H, tc.L, tc.k,
                                     tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
            elif tc.feature == 22:
                motion = MotionRNN2(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H,
                                    tc.L, tc.k,
                                    tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
            elif tc.feature == 230:
                motion = MotionRNN3c(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R,
                                     tc.H, tc.L, tc.k,
                                     tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
            elif tc.feature == 231:
                motion = MotionRNN3r(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R,
                                     tc.H, tc.L, tc.k,
                                     tc.act, tc.rnn, tc.delta, tc.alpha, True, False)

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=tc.gpu)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess.run(init)

            # print('Loading Model...')
            saver.restore(sess, path_m + '/model')

            # batch testing
            l = 0
            cmse = 0
            mse = 0
            for j in range(no_of_batches_test):
                ptr = j * tc.batch_size
                if tc.feature < 4:
                    ttx, ttpos, ttc, ttl, ttd = test_x[ptr:ptr + tc.batch_size], \
                                                test_pos[ptr:ptr + tc.batch_size], \
                                                test_c[ptr:ptr + tc.batch_size], \
                                                test_l[ptr:ptr + tc.batch_size], \
                                                test_d[ptr:ptr + tc.batch_size]
                    tty, ttxp = 1, 1
                elif 11 >= tc.feature >= 8:
                    ttx, ttxp, tty, ttpos, ttc, ttl, ttd = test_x[ptr:ptr + tc.batch_size], \
                                                           test_xp[ptr:ptr + tc.batch_size], \
                                                           test_y[ptr:ptr + tc.batch_size], \
                                                           test_pos[ptr:ptr + tc.batch_size], \
                                                           test_c[ptr:ptr + tc.batch_size], \
                                                           test_l[ptr:ptr + tc.batch_size], \
                                                           test_d[ptr:ptr + tc.batch_size]
                else:
                    ttx, tty, ttpos, ttc, ttl, ttd = test_x[ptr:ptr + tc.batch_size], \
                                                     test_y[ptr:ptr + tc.batch_size], \
                                                     test_pos[ptr:ptr + tc.batch_size], \
                                                     test_c[ptr:ptr + tc.batch_size], \
                                                     test_l[ptr:ptr + tc.batch_size], \
                                                     test_d[ptr:ptr + tc.batch_size]
                    ttxp = 1

                reg_loss, cls_loss, cost_loss, loss = sess.run(motion.loss(),
                                                               feed_dict={
                                                                   motion.x: ttx,
                                                                   motion.xp: ttxp,
                                                                   motion.y: tty,
                                                                   motion.pos: ttpos,
                                                                   motion.c: ttc,
                                                                   motion.d: ttd,
                                                                   motion.xlastloc: ttl,
                                                                   motion._dropout: 1
                                                               })
                mcmse, mrmse = sess.run(motion.pred_power(),
                                        feed_dict={
                                            motion.x: ttx, motion.xp: ttxp, motion.y: tty,
                                            motion.pos: ttpos,
                                            motion.c: ttc,
                                            motion.d: ttd,
                                            motion.xlastloc: ttl,
                                            motion._dropout: 1
                                        })

                l += loss
                cmse += mcmse
                mse += mrmse

            l /= no_of_batches_test
            mse /= no_of_batches_test
            cmse /= no_of_batches_test
            result = (tc.shortname + stt + ' loss= ' + '{:.4f}'.format(l) +
                      ' Test RMSE= ' + '{:.6f}'.format(mse) + ' Test CMSE= ' + '{:.6f}'.format(
                        cmse))
            sess.close()
            print(result)
            print('Test Finished!')


def runRNN_md_train_multi(scenarios, ss, oo, ff, acts, rnns, rr, hh, ll, kk,
                          delta, alpha, dropout, batch_size, epoch, lr, togit=1):
    for ft in ff:
        for st in ss:
            for ot in oo:
                for scenario in scenarios:
                    for act in acts:
                        for rnn in rnns:
                            for r in rr:
                                for h, l, k in zip(hh, ll, kk):
                                    print('----------------------------------------')
                                    nc = config(scenario, st, ot, ft, act, rnn, r, h, l, k,
                                                delta, alpha, dropout, batch_size, epoch, lr)
                                    stritem = 'Training ' + 'RNN ' + nc.fullname
                                    print(stritem)
                                    start = time.time()
                                    training_multi(nc, togit)
                                    # tc = test_config(scenario, st, ot, ft, act, rnn, H, L, k,
                                    #                 delta, alpha, dropout, batch_size, epoch, lr)
                                    # testing(tc, togit)
                                    print('Time : ', time.time() - start)


def runRNN_md_test_multi(scenarios, ss, oo, ff, acts, rnns, rr, hh, ll, kk,
                         delta, alpha, dropout, batch_size, epoch, lr, togit=1):
    for ft in ff:
        for st in ss:
            for ot in oo:
                for scenario in scenarios:
                    for act in acts:
                        for rnn in rnns:
                            for r in rr:
                                for h, l, k in zip(hh, ll, kk):
                                    print('----------------------------------------')
                                    tc = test_config(scenario, st, ot, ft, act, rnn, r, h, l, k,
                                                     delta, alpha, dropout, batch_size, epoch, lr)
                                    stritem = 'Testing ' + 'RNN ' + tc.fullname
                                    print(stritem)
                                    start = time.time()
                                    testing_multi(tc, togit)
                                    print('Time : ', time.time() - start)
