# ## data preparation for LSTM
#
# reading and create .npy files for training and testing.
# use variable sequence lengths from zero to s
#

import math
import os
import random
import subprocess
import time
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import csv
from linking.convert_XML import writeXML, writeTXT


# load the constructed data to memory
def loadRNNdata(scenario, st, out, feature, istrain, togit=1):
    if scenario == 'VIRUS':
        n = 3
    else:
        n = 2
    nciter = 1

    if togit == 1:
        pathsave = '/home/yaoyao/IDEA/dl/data/MTJ/RNNMD/' + scenario + '/RNNmd' + str(nciter) + '/'
    elif togit == 0:
        pathsave = '/media/yyao/work/data/MTJ/RNNMD/' + scenario + '/RNNmd' + str(nciter) + '/'
    else:
        pathsave = '/nfs/home4/yao88/IDEA/dl/data/MTJ/RNNMD/' + scenario + '/RNNmd' + str(
            nciter) + '/'

    if not os.path.exists(pathsave + 'p_train' + str(st) + str(out) + '.npy'):
        s = 10
    else:
        s = st

    if istrain:
        train_x = np.load(pathsave + 'x_train' + str(s) + str(out) + '.npy')
        train_y = np.load(pathsave + 'y_train' + str(s) + str(out) + '.npy')
        train_t = np.load(pathsave + 't_train' + str(s) + str(out) + '.npy')
        train_p = np.load(pathsave + 'p_train' + str(s) + str(out) + '.npy')
        train_l = np.load(pathsave + 'l_train' + str(s) + str(out) + '.npy')
        train_d = np.load(pathsave + 'd_train' + str(s) + str(out) + '.npy')
        train_d = np.reshape(train_d, (train_d.shape[0], 1))
        x = np.load(pathsave + 'x_val' + str(s) + str(out) + '.npy')
        y = np.load(pathsave + 'y_val' + str(s) + str(out) + '.npy')
        t = np.load(pathsave + 't_val' + str(s) + str(out) + '.npy')
        p = np.load(pathsave + 'p_val' + str(s) + str(out) + '.npy')
        l = np.load(pathsave + 'l_val' + str(s) + str(out) + '.npy')
        d = np.load(pathsave + 'd_val' + str(s) + str(out) + '.npy')
        d = np.reshape(d, (d.shape[0], 1))
        train_x = train_x[:, -st:, :]
        x = x[:, -st:, :]

        time = len(np.where(p[:, 1] == 1)) / len(np.where(p[:, 0] == 1))
        print('Validation P vs. N ratio = ' + str(time))

        train_xp = 1
        xp = 1
        if feature < 4:  # 0,1,2,3 position only x+y
            train_x = np.concatenate((train_x[:, :, -n:], train_y[:, :, -n:]), axis=1)
            x = np.concatenate((x[:, :, -n:], y[:, :, -n:]), axis=1)
            train_y = 1
            y = 1
        elif 4 <= feature < 6:  # 4,5 position only x,y
            train_x = train_x[:, :, -n:]
            train_y = train_y[:, :, -n:]
            x = x[:, :, -n:]
            y = y[:, :, -n:]
        else:  # 6,7,... handcraft x,y and position only x+y
            train_xp = np.concatenate((train_x[:, :, -n:], train_y[:, :, -n:]), axis=1)
            xp = np.concatenate((x[:, :, -n:], y[:, :, -n:]), axis=1)
            train_x = train_x[:, :, :-n]
            train_y = train_y[:, :, :-n]
            x = x[:, :, :-n]
            y = y[:, :, :-n]

        return train_x, train_xp, train_y, train_t, train_p, train_l, train_d, x, xp, y, t, p, l, d
    else:
        x = np.load(pathsave + 'x_test' + str(s) + str(out) + '.npy')
        y = np.load(pathsave + 'y_test' + str(s) + str(out) + '.npy')
        t = np.load(pathsave + 't_test' + str(s) + str(out) + '.npy')
        p = np.load(pathsave + 'p_test' + str(s) + str(out) + '.npy')
        l = np.load(pathsave + 'l_test' + str(s) + str(out) + '.npy')
        d = np.load(pathsave + 'd_test' + str(s) + str(out) + '.npy')
        d = np.reshape(d, (d.shape[0], 1))
        x = x[:, -st:, :]

        time = len(np.where(p[:, 1] == 1)) / len(np.where(p[:, 0] == 1))
        # print('Test P vs. N ratio = ' + str(time))

        xp = 1
        if feature < 4:
            x = np.concatenate((x[:, :, -n:], y[:, :, -n:]), axis=1)
            y = 1
        elif 4 <= feature < 6:
            x = x[:, :, -n:]
            y = y[:, :, -n:]
        else:
            xp = np.concatenate((x[:, :, -n:], y[:, :, -n:]), axis=1)
            x = x[:, :, :-n]
            y = y[:, :, :-n]
        return x, xp, y, t, p, l, d


# distance between two detections
def distance(pos1, pos2, is3D=False):
    if is3D:
        dis = math.sqrt((math.pow(pos1[0] - pos2[0], 2) +
                         math.pow(pos1[1] - pos2[1], 2) +
                         math.pow(pos1[3] - pos2[3], 2)))
    else:
        dis = math.sqrt((math.pow(pos1[0] - pos2[0], 2) +
                         math.pow(pos1[1] - pos2[1], 2)))
    return dis


# distance between two detections
def distance2(pos1, pos2, is3D=False):
    if is3D:
        dis = math.sqrt((math.pow(pos1[0] - pos2[0], 2) +
                         math.pow(pos1[1] - pos2[1], 2) +
                         math.pow(pos1[2] - pos2[2], 2)))
    else:
        dis = math.sqrt((math.pow(pos1[0] - pos2[0], 2) +
                         math.pow(pos1[1] - pos2[1], 2)))
    return dis


# summary of the data
def track_disp_summary():
    al = []
    cl = []
    scenarios = ['VIRUS', 'VESICLE', 'RECEPTOR', 'MICROTUBULE']
    for scenario in scenarios:
        x = []
        d = []
        z = []
        t = []
        l = []
        print(scenario)
        for snrid in [1, 2, 4, 7]:
            for dens in ['high', 'mid', 'low']:
                path = ('/home/yaoyao/IDEA/dl/data/MTJ_RNN/data/' + scenario + ' snr ' + str(
                    snrid) + ' density ' + dens + '.detections.xml.txt')
                data = getData(path)
                # print(' Data loaded from ' + path[pi] + '\n. Formatting...')
                tr = np.asarray(data[:, 4])
                tot_tracks = int(tr.max() + 1)
                # zz = np.asarray(data[:, 3])
                # print(zz.min(), zz.max())
                # xx = np.asarray(data[:, 0])
                # print(xx.min(), xx.max())
                # yy = np.asarray(data[:, 1])
                # print(yy.min(), yy.max())

                for track_id in range(tot_tracks):
                    track, dataprocaug = getTrackN(track_id, data)
                    l.append(len(track))
                    if len(track) > 0:
                        for index in range(1, len(track)):
                            x.append(abs(track[index, 0] - track[index - 1, 0]))
                            x.append(abs(track[index, 1] - track[index - 1, 1]))
                            z.append(abs(track[index, 3] - track[index - 1, 3]))
                            t.append(abs(track[index, 2] - track[index - 1, 2]))
                            if scenario == 'VIRUS':
                                d.append(
                                    math.sqrt(math.pow((track[index, 0] - track[index - 1, 0]), 2) +
                                              math.pow((track[index, 1] - track[index - 1, 1]), 2) +
                                              math.pow((track[index, 3] - track[index - 1, 3]), 2)))
                            else:
                                d.append(
                                    math.sqrt(math.pow((track[index, 0] - track[index - 1, 0]), 2) +
                                              math.pow((track[index, 1] - track[index - 1, 1]), 2)))
        x = np.asarray(x)
        d = np.asarray(d)
        z = np.asarray(z)
        t = np.asarray(t)
        al.append(x)
        cl.append(d)
        l = np.asarray(l)
        print(x.min(), x.max(), x.mean(), np.median(x))
        print(d.min(), d.max(), d.mean(), np.median(d))
        print(z.min(), z.max(), z.mean(), np.median(z))
        print(t.min(), t.max(), t.mean(), np.median(t))
        print(l.min(), l.max(), l.mean(), np.median(l))
        # plt.figure()
        # figpathd = 'results/' + scenario + 'disdxy.png'
        # plt.hist(al, 20, alpha=0.75, label=scenarios)
        # plt.xlabel('Distance xy')
        # plt.ylabel('Counts')
        # plt.title(r'$\mathrm{Histogram\ of\ NN}$')
        # plt.grid(True)
        # # plt.show()
        # plt.legend(loc='upper right')
        # plt.savefig(figpathd)
        # # subprocess.call(["curl", "-s", "-X", "POST",
        # #                 "https://api.telegram.org/bot459863485:AAEPUJsNI0wkf3iI8RfweGMR9u0rKiSHvuc/sendPhoto",
        # #                 "-F", "chat_id=385302225", "-F", 'photo=@' + figpathd])
        #
        # plt.figure()
        # figpathv = 'results/' + scenario + 'disd.png'
        # plt.hist(cl, 20, alpha=0.75, label=scenarios)
        # plt.xlabel('Distance ec')
        # plt.ylabel('Counts')
        # plt.title(r'$\mathrm{Histogram\ of\ NNxy}$')
        # plt.grid(True)
        # # plt.show()
        # plt.legend(loc='upper right')
        # plt.savefig(figpathv)
        # # subprocess.call(["curl", "-s", "-X", "POST",
        # #                 "https://api.telegram.org/bot459863485:AAEPUJsNI0wkf3iI8RfweGMR9u0rKiSHvuc/sendPhoto",
        # #                 "-F", "chat_id=385302225", "-F", 'photo=@' + figpathv])


def readVR():
    file = '/home/yaoyao/Dropbox/dl/R.txt'

    s = []
    h = []
    i = []
    l = []
    k = []
    acc = []
    los = []
    pre = []
    rec = []
    mse = []

    with open(file) as f:
        lines = f.readlines()

    for li in range(0, len(lines), 2):
        line = lines[li]
        ind = line.find("RECEPTORS")
        indli = line.find("_")
        s.append(line[ind + 9])
        h.append(line[ind + 11])
        i.append(line[ind + 13:indli])
        indl = line.find("H1L")
        indk = line.find("k")
        inddt = line.find("dt")
        l.append(line[indl + 3:indk])
        k.append(line[indk + 1:inddt])
        inda = line.find("= ")
        indl = line.find(" loss= ")
        indp = line.find(" Test Precision= ")
        indr = line.find(" Test Recall= ")
        indrl = line.find(" Rloss= ")
        indm = line.find(" Test MSE= ")
        acc.append(line[inda + 2:indl])
        los.append(line[indl + 6:indp])
        pre.append(line[indp + 16:indr])
        rec.append(line[indr + 13:indrl])
        mse.append(line[indm + 10:-1])
    ft = np.vstack((s, h, i, l, k, los, acc, pre, rec, mse))
    ft = np.asanyarray(ft, dtype=np.float32)
    ft = np.matrix.transpose(ft)
    np.savetxt('/home/yaoyao/Dropbox/dl/Rtab.txt', ft, fmt='%.3f')


def readRorV():
    file = '/home/yaoyao/Dropbox/dl/Vtab.txt'
    with open(file, 'r') as csvfile:
        my_data = csv.reader(csvfile, delimiter=',')
        table = [[e for e in r] for r in my_data]
    table = table[1:]
    fig = plt.figure(figsize=(8, 18))
    set = ['L8k32', 'L64k256', 'L256k1024']
    for i in [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44]:
        acc = []
        los = []
        pre = []
        rec = []
        mse = []
        for l in table:
            if float(l[2]) == i:
                acc.append(float(l[6]))
                los.append(float(l[5]))
                pre.append(float(l[7]))
                rec.append(float(l[8]))
                mse.append(float(l[9]))
        acc = np.asanyarray(acc)
        los = np.asanyarray(los)
        pre = np.asanyarray(pre)
        rec = np.asanyarray(rec)
        mse = np.asanyarray(mse)

        xt = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44]
        xtl = ['c-l', 'v-l', 'a-l', 'v&a-l', 'all-l',
               'c-m', 'v-m', 'a-m', 'v&a-m', 'all-m',
               'c-h', 'v-h', 'a-h', 'v&a-h', 'all-h']

        ax = plt.subplot(5, 1, 1)
        line, = ax.plot([i, i, i], los, '+-')
        # ax.set_title('Loss')
        ax.set_ylabel('loss')
        # ax.set_xlabel('i')
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl)

        ax = plt.subplot(5, 1, 2)
        line, = ax.plot([i, i, i], acc, '+-')
        # ax.set_title('Accuracy')
        ax.set_ylabel('accuracy')
        # ax.set_ylim([50,95])
        # ax.set_xlabel('i')
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl)

        ax = plt.subplot(5, 1, 3)
        line, = ax.plot([i, i, i], pre, '+-')
        # ax.set_title('Precision')
        ax.set_ylabel('precision')
        # ax.set_ylim([80,100])
        # ax.set_xlabel('i')
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl)

        ax = plt.subplot(5, 1, 4)
        line, = ax.plot([i, i, i], rec, '+-')
        # ax.set_title('Recall')
        ax.set_ylabel('recall')
        # ax.set_ylim([80,100])
        # ax.set_xlabel('i')
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl)

        ax = plt.subplot(5, 1, 5)
        line, = ax.plot([i, i, i], mse, '+-')
        # ax.set_title('MSE')
        ax.set_ylabel('mse')
        # ax.set_ylim([1,4])
        ax.set_xlabel('features-data used')
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl)

    # fig.show()
    fig.savefig('/home/yaoyao/Dropbox/dl/V.png')


# simulate add false-positives, false-negtives to detections
def MTJsimulation(scenario, snr, dens, add, rem, R):
    path = '/home/yaoyao/IDEA/dl/data/MTJ_RNN/data/' + scenario + ' snr ' + str(
        snr) + ' density ' + dens + '.detections.xml.txt'
    path_save = '/home/yaoyao/IDEA/dl/data/MTJ_RNN/data/' + scenario + ' snr ' + str(
        snr) + ' density ' + dens + \
                ' R ' + str(R) + ' add ' + str(add) + ' rem ' + str(rem) + '.detections.xml.txt'

    if scenario == 'VIRUS':
        is3D = True
    else:
        is3D = False

    data = getData(path)
    newdata = []

    for fi in range(100):
        frame = data[data[:, 2] == fi]

        if rem > 0:
            tot = len(frame)
            r_size = tot - int(tot * rem * 0.01)
            frame = np.random.permutation(frame)
            frame = frame[:r_size]

        for di in range(len(frame) - 1):
            if frame[di, -1] != 1:
                for dd in range(di + 1, len(frame)):
                    if frame[dd, -1] != 1:
                        dis = distance(frame[di], frame[dd], is3D)
                        if dis < R * 0.1:
                            frame[dd, -1] = 1
        frame = frame[frame[:, -1] != 1]  # remove close detections inside R

        if add > 0:
            tot = len(frame)
            a_size = int(tot * add * 0.01)
            for ai in range(a_size):
                det = []
                det.append(np.random.uniform(0, 512))
                det.append(np.random.uniform(0, 512))
                det.append(fi)
                if is3D:
                    det.append(np.random.uniform(-0.909, 9.9091))
                else:
                    det.append(0)
                det.append(0)
                det.append(np.random.uniform(1, 10))
                det.append(0)
                det.append(0)
                det.append(0)
                det.append(0)
                det = np.array(det)
                frame = np.vstack((frame, det))

        newdata.extend(frame)
    newdata = np.array(newdata)
    np.savetxt(path_save, newdata, fmt='%.3f')


# check whether a detection is in the gates
def inSide(pn, p, delta, gate, dv='', gatez=3, is3D=False):
    if is3D:
        gateall = math.sqrt(gate * gate * 2 + gatez * gatez)
        disp = math.sqrt(
            math.pow(pn[0] - p[0], 2) + math.pow(pn[1] - p[1], 2) + math.pow(pn[3] - p[3], 2))
        td = abs(pn[2] - p[2])
        disvx = abs((pn[0] - p[0]) / td - delta[0])
        disvy = abs((pn[1] - p[1]) / td - delta[1])
        disvz = abs((pn[3] - p[3]) / td - delta[2])
        if dv == 'v':
            if disvx < gate and disvy < gate and disvz < gatez:
                return True
            else:
                return False
        elif dv == 'd':
            if disp < td * gateall:
                return True
            else:
                return False
        else:
            if (disvx < gate and disvy < gate and disvz < gatez) or disp < td * gateall:
                return True
            else:
                return False
    else:
        gateall = math.sqrt(gate * gate * 2)
        disp = math.sqrt(math.pow(pn[0] - p[0], 2) + math.pow(pn[1] - p[1], 2))
        td = abs(pn[2] - p[2])
        disvx = abs((pn[0] - p[0]) / td - delta[0])
        disvy = abs((pn[1] - p[1]) / td - delta[1])
        if dv == 'v':
            if disvx < gate and disvy < gate:
                return True
            else:
                return False
        elif dv == 'd':
            if disp < td * gateall:
                return True
            else:
                return False
        else:
            if (disvx < gate and disvy < gate) or disp < td * gateall:
                return True
            else:
                return False


# get data from file
def getData(path):
    with open(path) as f:
        data = np.loadtxt(path)
    f.close()
    return data


# get tracks and do interpolation to fill gaps
def getTrackI(index, data):
    track = data[data[:, 4] == index]
    track = track[track[:, 5] == 255]
    if len(track) > 0:
        newtrack = []
        iid = 0
        newtrack.append(track[iid])
        for iid in range(1, len(track)):
            if track[iid, 2] - track[iid - 1, 2] != 1:
                x = (track[iid, 0] - track[iid - 1, 0]) / (
                            1.0 * (track[iid, 2] - track[iid - 1, 2]))
                y = (track[iid, 1] - track[iid - 1, 1]) / (
                            1.0 * (track[iid, 2] - track[iid - 1, 2]))
                z = 255  # (track[iid, 3] - track[iid - 1, 3]) / (1.0 * (track[iid, 2] - track[iid - 1, 2]))
                for i in range(int(track[iid, 2] - track[iid - 1, 2] - 1)):
                    newtrack.append(np.asarray(
                        [(track[iid - 1, 0] + x * (i + 1)), (track[iid - 1, 1] + y * (i + 1)),
                         track[iid - 1, 2] + i + 1, z, track[iid, 4], 1.0, 1.0, 0.0, 0.0, 0.0]))
                newtrack.append(track[iid])
            else:
                newtrack.append(track[iid])
        newtrack = np.asarray(newtrack)
        return newtrack, data
    else:
        return [], data


# get tracks and random set detections to gaps
def getTrackR(index, dataorg):
    trackorg = dataorg[dataorg[:, 4] == index]
    trackorg = trackorg[trackorg[:, 5] == 255]
    track = deepcopy(trackorg)
    data = deepcopy(dataorg)
    if len(track) > 3:
        maxa = min(int(0.25 * len(track)), 10)
        a = random.randint(0, maxa)
        tag = False
        first = True
        if a > 0:
            while tag == True or first == True:
                j = random.sample(range(len(track)), a)
                j = np.sort(j)
                first = False
                tag = False
                if len(j) >= 4:
                    for k in range(3, len(j)):
                        if j[k] - j[k - 3] == 3 and j[k - 3] != 0 and j[k] != len(track) - 1:
                            tag = True
                            maxa = maxa - 1
                            a = random.randint(0, maxa)
                            break
            for aa in j:
                if aa != 0 and aa != len(track) - 1:
                    track[aa, 6] = 1
                    for i in range(len(data)):
                        if data[i, 0] == track[aa, 0] and data[i, 1] == track[aa, 1] and data[
                            i, 2] == track[aa, 2]:
                            data[i, 6] = 1
                            break
    return track, data


# get tracks as it is in file
def getTrackN(index, data):
    track = data[data[:, 4] == index]
    track = track[track[:, 5] == 255]
    return track, data


# data argumentation 4 types
def dataArgumentation(trackorg, dataorg, type, framelength=100):
    track = deepcopy(trackorg)
    data = deepcopy(dataorg)
    if type == 1:  # original forward tracks
        return track, data
    elif type == 2:  # XY->YX
        for i in range(len(track)):
            temp = track[i, 0]
            track[i, 0] = track[i, 1]
            track[i, 1] = temp
        data[:, 0], data[:, 1] = data[:, 1], data[:, 0].copy()
        return track, data
    elif type == 3:  # backward tracks
        newtrack = []
        for i in range(len(track) - 1, -1, -1):
            track[i, 2] = framelength - 1 - track[i, 2]
            newtrack.append(track[i])
        for i in range(len(data)):
            data[i, 2] = framelength - 1 - data[i, 2]
        newtrack = np.asarray(newtrack)
        return newtrack, data
    else:  # backward tracks and XY->YX
        newtrack = []
        for i in range(len(track) - 1, -1, -1):
            temp = track[i, 0]
            track[i, 0] = track[i, 1]
            track[i, 1] = temp
            track[i, 2] = framelength - 1 - track[i, 2]
            newtrack.append(track[i])
        for i in range(len(data)):
            data[i, 2] = framelength - 1 - data[i, 2]
        data[:, 0], data[:, 1] = data[:, 1], data[:, 0].copy()
        newtrack = np.asarray(newtrack)
        return newtrack, data


# the performance of simulated GT track with different FN and 0%FP
def simulateGT(gap=True):
    rr = [-1, 0, 5, 10, 15, 20]
    aa = [0]
    R = 25
    print('----------------------------------------')

    scenario = 'MICROTUBULE'
    snr = 1
    dens = 'low'
    ref = 'data/MTJ_RNN/data/XML/' + scenario + ' snr ' + str(
        snr) + ' density ' + dens + '.xml'
    saveTo = 'results/RNN/simulation' + str(gap)[0] + '/'

    for rem in rr:
        for add in aa:
            if rem == -1 and add == 0:
                path = 'data/MTJ_RNN/simulate/' + scenario + ' snr ' + str(
                    snr) + ' density ' + dens + '.detections.xml.txt'

                npy = saveTo + scenario + str(snr) + dens + 'RNNpos.npy'
                can = saveTo + scenario + str(snr) + dens + 'RNNpos.xml'
                o = saveTo + scenario + str(snr) + dens + 'RNNpos.xml.txt'

            else:
                path = 'data/MTJ_RNN/simulate/' + scenario + ' snr ' + str(
                    snr) + ' density ' + dens + ' R ' + str(R) + ' add ' + str(
                    add) + ' rem ' + str(rem) + '.detections.xml.txt'

                npy = saveTo + scenario + str(snr) + dens + 'RNN' + ' R ' + str(
                    R) + ' add ' + str(add) + ' rem ' + str(rem) + 'pos.npy'
                can = saveTo + scenario + str(snr) + dens + 'RNN' + ' R ' + str(
                    R) + ' add ' + str(add) + ' rem ' + str(rem) + 'pos.xml'
                o = saveTo + scenario + str(snr) + dens + 'RNN' + ' R ' + str(
                    R) + ' add ' + str(add) + ' rem ' + str(rem) + 'pos.xml.txt'

            data = getData(path)
            print(' 3 Data loaded from ' + path + '\n. Formatting...')
            tr = np.asarray(data[:, 4])
            tot_tracks = int(tr.max() + 1)
            poslist = []
            for track_id in range(tot_tracks):
                if gap:
                    track = getTrackN(track_id, data)
                else:
                    track = getTrackI(track_id, data)
                poslist.append(track)
            np.save(npy, poslist)
            writeXML(npy, can, scenario, snr, dens, 'GT', int(gap))
            # subprocess.call(['rm', npy])
            subprocess.call([
                'java', '-jar', 'trackingPerformanceEvaluation.jar', '-r', ref, '-c', can, '-o', o])


# perpare Ground truth data
def XMLtoTXT():
    sort = True
    path = 'data/MTJ_RNN/data/XML/'
    for s in ['MICROTUBULE', 'VESICLE', 'RECEPTOR', 'VIRUS']:
        for snr in [1, 2, 4, 7]:
            for dens in ['low', 'mid', 'high']:
                file = path + s + ' snr ' + str(snr) + ' density ' + dens + '.xml'
                writeTXT(file, sort)


# get the track without gaps
def getNoGapTrack(track):
    newtrack = []
    for i in range(0, len(track)):
        if track[i, 0] != -1:
            newtrack.append(track[i])
    newtrack = np.asarray(newtrack)
    return newtrack


# last index of real point in track
def getLastNonGap(track, s):
    tp = -1
    for i in range(len(track) - 2, -1, -1):
        if track[i, 0] != -1:
            tp = i
            break
    return tp


# first real point in track
def getFirstNonVirtual(track, s):
    tp = -1
    for i in range(len(track)):
        if track[i, 0] != -1:
            tp = i
            break
    if tp == len(track) - 1:
        tp = -1
    return tp


# feature extraction for established tracks
def featureExtractX(track, s, features, is3D=True):
    temp = []
    disp = []
    lpos = []
    track = np.asarray(track)

    if track[-1, 0] != -1:
        lpos.append(track[-1, 0])
        lpos.append(track[-1, 1])
        if is3D:
            lpos.append(track[-1, 3])
    else:
        nogap = getNoGapTrack(track)
        tx, ty, tz = avedisp(nogap, len(track))
        iid = getLastNonGap(track, s)
        lpos.append(track[iid, 0] + (len(track) - iid - 1) * tx)
        lpos.append(track[iid, 1] + (len(track) - iid - 1) * ty)
        if is3D:
            lpos.append(track[iid, 3] + (len(track) - iid - 1) * tz)

    idg = getLastNonGap(track, s)
    if track[-1, 0] != -1 and idg != -1:
        timesglng = len(track) - idg - 1
        disp.append((track[-1, 0] - track[idg, 0]) / timesglng)
        disp.append((track[-1, 1] - track[idg, 1]) / timesglng)
        if is3D:
            disp.append((track[-1, 3] - track[idg, 3]) / timesglng)
    else:
        disp.append(0)
        disp.append(0)
        if is3D:
            disp.append(0)

    for i in range(0, len(track) - 1):  # s+1 is the next true!
        glng = getLastNonGap(track[:i + 2], s)

        if glng != -1:
            glng_1 = getLastNonGap(track[:glng + 1], s)
        else:
            glng_1 = -1
        gfnv = getFirstNonVirtual(track[:i + 2], s)

        if features[0]:  # instant displacement
            if track[i + 1, 0] != -1 and glng != -1:
                timesglng = i - glng + 1
                temp.append((track[i + 1, 0] - track[glng, 0]) / timesglng)
                temp.append((track[i + 1, 1] - track[glng, 1]) / timesglng)
                if is3D:
                    temp.append((track[i + 1, 3] - track[glng, 3]) / timesglng)
            else:
                temp.append(0)
                temp.append(0)
                if is3D:
                    temp.append(0)

        if features[1]:  # displacement to start of sliding window
            if track[i + 1, 0] != -1 and gfnv != -1:
                temp.append(track[i + 1, 0] - track[gfnv, 0])
                temp.append(track[i + 1, 1] - track[gfnv, 1])
                if is3D:
                    temp.append(track[i + 1, 3] - track[gfnv, 3])
            else:
                temp.append(0)
                temp.append(0)
                if is3D:
                    temp.append(0)

        if features[2]:  # instant angle
            if track[i + 1, 0] != -1 and glng != -1:
                arpi = math.atan2(track[i + 1, 1] - track[glng, 1],
                                  track[i + 1, 0] - track[glng, 0])
                if arpi == -math.pi:
                    arpi = math.pi
                temp.append(arpi)
                if is3D:
                    arpi = math.atan2(track[i + 1, 0] - track[glng, 0],
                                      track[i + 1, 3] - track[glng, 3])
                    if arpi == -math.pi:
                        arpi = math.pi
                    temp.append(arpi)
                    arpi = math.atan2(track[i + 1, 3] - track[glng, 3],
                                      track[i + 1, 1] - track[glng, 1])
                    # sinarpi = math.sin(arpi)
                    # temp.append(sinarpi)
                    if arpi == -math.pi:
                        arpi = math.pi
                    temp.append(arpi)
            else:
                temp.append(0)
                if is3D:
                    temp.append(0)
                    temp.append(0)

        if features[3]:  # distance to start of sliding window
            if track[i + 1, 0] != -1 and gfnv != -1:
                newtrack = getNoGapTrack(track[gfnv:i + 2])
                dttemp = 0
                for dti in range(len(newtrack) - 1):
                    dttemp += math.sqrt(
                        math.pow(newtrack[dti + 1, 0] - newtrack[dti, 0], 2) +
                        math.pow(newtrack[dti + 1, 1] - newtrack[dti, 1], 2) +
                        math.pow(newtrack[dti + 1, 3] - newtrack[dti, 3], 2))
                temp.append(dttemp)
            else:
                temp.append(0)

        if features[4]:  # velocity difference
            if track[i + 1, 0] != -1 and glng != -1 and glng_1 != -1:
                gl1 = i + 1 - glng
                gl2 = glng - glng_1
                temp.append((track[i + 1, 0] - track[glng, 0]) /
                            gl1 - (track[glng, 0] - track[glng_1, 0]) / gl2)
                temp.append((track[i + 1, 1] - track[glng, 1]) /
                            gl1 - (track[glng, 1] - track[glng_1, 1]) / gl2)
                if is3D:
                    temp.append((track[i + 1, 3] - track[glng, 3]) /
                                gl1 - (track[glng, 3] - track[glng_1, 3]) / gl2)
            else:
                temp.append(0)
                temp.append(0)
                if is3D:
                    temp.append(0)

        if features[5]:  # original location of particle
            if track[i + 1, 0] != -1:
                temp.append(track[i + 1, 0])
                temp.append(track[i + 1, 1])
                if is3D:
                    temp.append(track[i + 1, 3])
            else:
                temp.append(-1)
                temp.append(-1)
                if is3D:
                    temp.append(-1)

    temp = np.asarray(temp)
    n_input = int(temp.shape[0] / s)
    temp = np.reshape(temp, (s, n_input))
    disp = np.asarray(disp)
    if is3D:
        return temp, disp, lpos
    else:
        return temp, disp, lpos


# feature extraction for candidates
def featureExtractY(tracky, s, features, is3D=True):
    i = len(tracky) - 2
    track = np.asarray(tracky)

    temp = []
    disp = []

    glng = getLastNonGap(track, s)
    if glng != -1:
        glng_1 = getLastNonGap(track[:glng + 1], s)
    else:
        glng_1 = -1
    gfnv = getFirstNonVirtual(track, s)

    if track[i + 1, 0] != -1 and glng != -1:
        timesglng = i - glng + 1
        disp.append((track[i + 1, 0] - track[glng, 0]) / timesglng)
        disp.append((track[i + 1, 1] - track[glng, 1]) / timesglng)
        if is3D:
            disp.append((track[i + 1, 3] - track[glng, 3]) / timesglng)
    else:
        disp.append(0)
        disp.append(0)
        if is3D:
            disp.append(0)

    if features[0]:
        if track[i + 1, 0] != -1 and glng != -1:
            timesglng = i - glng + 1
            temp.append((track[i + 1, 0] - track[glng, 0]) / timesglng)
            temp.append((track[i + 1, 1] - track[glng, 1]) / timesglng)
            if is3D:
                temp.append((track[i + 1, 3] - track[glng, 3]) / timesglng)
        else:
            temp.append(0)
            temp.append(0)
            if is3D:
                temp.append(0)

    if features[1]:  # displacement to start of sliding window
        if track[i + 1, 0] != -1 and gfnv != -1:
            temp.append(track[i + 1, 0] - track[gfnv, 0])
            temp.append(track[i + 1, 1] - track[gfnv, 1])
            if is3D:
                temp.append(track[i + 1, 3] - track[gfnv, 3])
        else:
            temp.append(0)
            temp.append(0)
            if is3D:
                temp.append(0)

    if features[2]:  # instant angle
        if track[i + 1, 0] != -1 and glng != -1:
            arpi = math.atan2(track[i + 1, 1] - track[glng, 1],
                              track[i + 1, 0] - track[glng, 0])
            # sinarpi = math.sin(arpi)
            # temp.append(sinarpi)
            if arpi == -math.pi:
                arpi = math.pi
            temp.append(arpi)
            if is3D:

                arpi = math.atan2(track[i + 1, 0] - track[glng, 0],
                                  track[i + 1, 3] - track[glng, 3])
                if arpi == -math.pi:
                    arpi = math.pi
                temp.append(arpi)
                arpi = math.atan2(track[i + 1, 3] - track[glng, 3],
                                  track[i + 1, 1] - track[glng, 1])
                if arpi == -math.pi:
                    arpi = math.pi
                temp.append(arpi)
        else:
            temp.append(0)
            if is3D:
                temp.append(0)
                temp.append(0)

    if features[3]:  # distance to start of sliding window
        if track[i + 1, 0] != -1 and gfnv != -1:
            newtrack = getNoGapTrack(track[gfnv:])
            dttemp = 0
            for dti in range(len(newtrack) - 1):
                dttemp += math.sqrt(
                    math.pow(newtrack[dti + 1, 0] - newtrack[dti, 0], 2) +
                    math.pow(newtrack[dti + 1, 1] - newtrack[dti, 1], 2) +
                    math.pow(newtrack[dti + 1, 3] - newtrack[dti, 3], 2))
            temp.append(dttemp)
        else:
            temp.append(0)

    if features[4]:  # velocity difference
        if track[i + 1, 0] != -1 and glng != -1 and glng_1 != -1:
            gl1 = i + 1 - glng
            gl2 = glng - glng_1
            temp.append((track[i + 1, 0] - track[glng, 0]) /
                        gl1 - (track[glng, 0] - track[glng_1, 0]) / gl2)
            temp.append((track[i + 1, 1] - track[glng, 1]) /
                        gl1 - (track[glng, 1] - track[glng_1, 1]) / gl2)
            if is3D:
                temp.append((track[i + 1, 3] - track[glng, 3]) /
                            gl1 - (track[glng, 3] - track[glng_1, 3]) / gl2)
        else:
            temp.append(0)
            temp.append(0)
            if is3D:
                temp.append(0)

    if features[5]:  # original location of particle
        if track[i + 1, 0] != -1:
            temp.append(track[i + 1, 0])
            temp.append(track[i + 1, 1])
            if is3D:
                temp.append(track[i + 1, 3])
        else:
            temp.append(-1)
            temp.append(-1)
            if is3D:
                temp.append(-1)

    return temp, disp


# rank detections
def timeRank(true, track, time, num):
    if len(time) > 0:
        for i in range(len(time)):
            time[i][8] = math.sqrt(math.pow(track[0] - time[i][0], 2)
                                   + math.pow(track[1] - time[i][1], 2)
                                   + math.pow(track[3] - time[i][3], 2))
        ts = time[:, 8].argsort()
        t = time[ts]
        o1_true = -1
        for o1 in range(len(t)):
            if t[o1, 0] == true[0] and t[o1, 1] == true[1] and t[o1, 3] == true[3]:
                o1_true = o1
                break
        if o1_true >= 0 and o1_true + num <= len(t):
            return t[:o1_true + num], o1_true
        else:
            return t, o1_true
    else:
        return time, -1


# get the true location of a future detection
def truePoint(track, t):
    if t < len(track):
        truep = track[t]
    else:
        tx, ty, tz = avedisp(track, t - 1)
        truep = [track[t - 1][0] + tx, track[t - 1][1] + ty, track[t - 1][2] + 1,
                 track[t - 1][3] + tz,
                 track[t - 1][4], track[t - 1][5], track[t - 1][6], track[t - 1][7],
                 track[t - 1][8], track[t - 1][9]]
    return truep


# get average displacement of a track
def avedisp(track, t):
    tx = 0.
    ty = 0.
    tz = 0.
    num = 0
    if t >= len(track):
        t = len(track) - 1
    for ti in range(t, t - 3, -1):
        if ti - 1 >= 0 and ti < len(track):
            tx += track[ti, 0] - track[ti - 1, 0]
            ty += track[ti, 1] - track[ti - 1, 1]
            tz += track[ti, 3] - track[ti - 1, 3]
            num += 1
    tx /= num
    ty /= num
    tz /= num
    return tx, ty, tz


def getDets(f, data, lstmo=1):
    timep = []
    for z in range(lstmo):
        tep = data[data[:, 2] == f + z]
        tep = np.asarray(tep)
        tep = np.random.permutation(tep)
        timep.append(tep)
    return timep


def trackI(track, is3D=False):
    newtrack = []
    iid = 0
    print(len(track))
    newtrack.append(track[iid])
    for iid in range(1, len(track)):
        if track[iid, 2] - track[iid - 1, 2] != 1:
            x = (track[iid, 0] - track[iid - 1, 0]) / (1.0 * (track[iid, 2] - track[iid - 1, 2]))
            y = (track[iid, 1] - track[iid - 1, 1]) / (1.0 * (track[iid, 2] - track[iid - 1, 2]))
            if is3D:
                z = (track[iid, 3] - track[iid - 1, 3]) / (
                            1.0 * (track[iid, 2] - track[iid - 1, 2]))
            else:
                z = 0
            for i in range(int(track[iid, 2] - track[iid - 1, 2] - 1)):
                newtrack.append(np.asarray(
                    [(track[iid - 1, 0] + x * (i + 1)), (track[iid - 1, 1] + y * (i + 1)),
                     track[iid - 1, 2] + i + 1, z, track[iid, 4], 1.0, 1.0, 1.0, 1.0, 1.0]))
            newtrack.append(track[iid])
        else:
            newtrack.append(track[iid])
    newtrack = np.asarray(newtrack)
    return newtrack


# create training-testing data
def createRNNdataPloc(scenario, snr, path, s, out, gate, close, iter, features, istrain,
                      dv='', gatez=3, togit=0, isspt=True):
    if scenario == 'VIRUS':
        is3D = True
        p2 = 8  # 4 virus  8
        p3 = 100
    else:
        is3D = False
        p2 = 4
        p3 = 50

    if togit == 1:
        pathsave = '/home/yaoyao/IDEA/dl/data/MTJ/RNNMD/' + scenario + '/RNNmd' + str(
            iter) + '/after comb/'
    elif togit == 0:
        pathsave = '/media/yyao/work/data/MTJ/RNNMD/' + scenario + '/RNNmd' + str(
            iter) + '/after comb/'
    else:
        pathsave = '/nfs/home4/yao88/IDEA/dl/data/MTJ/RNNMD/' + scenario + '/RNNmd' + str(
            iter) + '/after comb/'

    if not os.path.exists(pathsave):
        os.makedirs(pathsave)

    pf = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

    if snr == 1:
        augtypes = [1]
    else:
        augtypes = [4, 3, 2, 1]
    x_t = []
    y_t = []
    t_t = []
    p_t = []
    l_t = []
    x_f = []
    y_f = []
    t_f = []
    p_f = []
    l_f = []
    d_t = []
    d_f = []
    for pi in range(len(path)):
        data = getData(path[pi])
        print(' Data loaded from ' + path[pi] + '   Formatting...')
        tr = np.asarray(data[:, 4])
        tot_tracks = int(tr.max() + 1)
        for track_aug_type in augtypes:
            cho_tracks = np.arange(tot_tracks)
            for track_id in cho_tracks:
                trackorg, dataproc = getTrackR(track_id, data)
                # trackorg, dataproc = getTrackN(track_id, data)
                # trackorg, dataproc = getTrackI(track_id, data)
                track, dataprocaug = dataArgumentation(trackorg, dataproc, track_aug_type)
                for index in range(out, len(track) + out):  # the last detection needed
                    xtrack = []
                    for i in range(index - s - out,
                                   index - out + 1):  # index-out+1 is the next true!
                        if i >= 0:
                            xtrack.append(track[i])
                        else:
                            xtrack.append(pf)
                    temp, xyidp, tlastpos = featureExtractX(xtrack, s, features, is3D)

                    timeptemp = []
                    for z in range(1, out + 1):
                        tnext = dataprocaug[dataprocaug[:, 2] == track[index - out, 2] + z]
                        tnext = tnext[tnext[:, 6] != 1]
                        timeptemp.append(tnext)
                    timeptemp = np.asarray(timeptemp)

                    timep = []
                    for ti in range(out):
                        timepp = []
                        if index - out + ti + 1 < len(track):
                            fr = track[index - out + ti + 1, 2]
                            tind = [track[index - out + ti + 1, 0],
                                    track[index - out + ti + 1, 1],
                                    track[index - out + ti + 1, 3]]
                        else:
                            fr = track[len(track) - 1, 2] + (index - out + ti + 2 - len(track))
                            tx, ty, tz = avedisp(track, index - out + ti)
                            tind = [
                                track[len(track) - 1, 0] + (index - out + ti + 2 - len(track)) * tx,
                                track[len(track) - 1, 1] + (index - out + ti + 2 - len(track)) * ty,
                                track[len(track) - 1, 3] + (index - out + ti + 2 - len(track)) * tz]

                        # remove some false negative detections for the first frame
                        '''
                        for tip in range(len(timeptemp[ti])):
                            clo = math.sqrt(math.pow(timeptemp[ti][tip, 0] - tind[0], 2) + math.pow(
                                timeptemp[ti][tip, 1] - tind[1], 2) + math.pow(
                                timeptemp[ti][tip, 3] - tind[2], 2))
                            if (ti + 1) * (gate + gatez) > clo:  # > close or clo == 0:
                                timepp.append(timeptemp[ti][tip])
                                # else:
                                #    print('delete')
                        '''
                        # add some false positive detections for the first frame
                        if snr != 1 and (pi in [0, 1, 3, 4]) and ti == 0 and fr < 100:
                            for li in range(4 - out):
                                det = [
                                    tind[0] + (-1) ** random.randrange(2) * np.random.uniform(1.5,
                                                                                              5),
                                    tind[1] + (-1) ** random.randrange(2) * np.random.uniform(1.5,
                                                                                              5),
                                    fr]
                                if is3D:
                                    det.append(
                                        tind[2] + (-1) ** random.randrange(2) * np.random.uniform(
                                            0.5, 1.5))
                                else:
                                    det.append(0)
                                det.append(0)
                                det.append(np.random.uniform(1, 10))
                                det.append(0)
                                det.append(0)
                                det.append(0)
                                det.append(0)
                                det = np.array(det)
                                timepp.append(det)
                        timepp = np.asarray(timepp)
                        timep.append(timepp)
                    timep = np.asarray(timep)

                    if out == 1:
                        # for the immediate next frame
                        o1_p = []  # indexes of detections in gate
                        temp_o1 = []
                        o1_true = -1
                        for o1 in range(len(timep[0])):
                            if index - out + 1 < len(track) \
                                    and timep[0][o1, 0] == track[index - out + 1, 0] \
                                    and timep[0][o1, 1] == track[index - out + 1, 1] \
                                    and timep[0][o1, 3] == track[index - out + 1, 3]:
                                o1_true = o1
                                break

                        for o1 in range(len(timep[0])):
                            if index - out < len(track) and inSide(timep[0][o1], track[index - out],
                                                                   xyidp, gate,
                                                                   dv, gatez, is3D):
                                ytrack = []
                                ytrack.extend(xtrack)
                                ytrack.append(timep[0][o1])
                                ytemp, _ = featureExtractY(ytrack, s, features, is3D)
                                temp_o1.append([ytemp])
                                if o1 == o1_true:
                                    o1_p.append(1)
                                else:
                                    o1_p.append(0)
                        # include pass frame [out+1 times of gate] for classification :
                        # have no detections in next frame
                        ytrack = []
                        ytrack.extend(xtrack)
                        ytrack.append(pf)
                        ytemp, _ = featureExtractY(ytrack, s, features, is3D)
                        temp_o1.append([ytemp])
                        if o1_true == -1:
                            o1_p.append(1)
                        else:
                            o1_p.append(0)

                        # create output x, y, t, p
                        for i in range(len(o1_p)):
                            if o1_p[i] == 0:
                                p_f.append([1, 0])
                                x_f.append(temp)
                                l_f.append(tlastpos)
                                y_f.append(temp_o1[i])
                                truetemp = []
                                oi = 1
                                if 0 < index - out + oi < len(track):
                                    truetemp.append(track[index - out + oi, 0])
                                    truetemp.append(track[index - out + oi, 1])
                                    if is3D:
                                        truetemp.append(track[index - out + oi, 3])
                                else:
                                    tx, ty, tz = avedisp(track, index - out + oi)
                                    truetemp.append(track[index - out, 0] + oi * tx)
                                    truetemp.append(track[index - out, 1] + oi * ty)
                                    if is3D:
                                        truetemp.append(track[index - out, 3] + oi * tz)
                                t_f.append(truetemp)
                                if is3D:
                                    if temp_o1[i][0][-1] == -1:
                                        d_f.append(math.sqrt(gate * gate * 2 + gatez * gatez))
                                    else:
                                        d_f.append(distance2(temp_o1[i][0][-3:], truetemp, is3D))
                                else:
                                    if temp_o1[i][0][-1] == -1:
                                        d_f.append(math.sqrt(gate * gate * 2))
                                    else:
                                        d_f.append(distance2(temp_o1[i][0][-2:], truetemp, is3D))
                            else:
                                p_t.append([0, 1])
                                x_t.append(temp)
                                l_t.append(tlastpos)
                                y_t.append(temp_o1[i])
                                truetemp = []
                                oi = 1
                                if 0 < index - out + oi < len(track):
                                    truetemp.append(track[index - out + oi, 0])
                                    truetemp.append(track[index - out + oi, 1])
                                    if is3D:
                                        truetemp.append(track[index - out + oi, 3])
                                else:
                                    tx, ty, tz = avedisp(track, index - out + oi)
                                    truetemp.append(track[index - out, 0] + oi * tx)
                                    truetemp.append(track[index - out, 1] + oi * ty)
                                    if is3D:
                                        truetemp.append(track[index - out, 3] + oi * tz)
                                t_t.append(truetemp)
                                d_t.append(0)
                    elif out == 2:
                        # for second next frame
                        o2_p = []  # indexes of detections in gate
                        temp_o2 = []
                        o1_true = -1
                        o2_true = -1
                        for o1 in range(len(timep[0])):
                            if index - out + 1 < len(track) and \
                                    timep[0][o1, 0] == track[index - out + 1, 0] and \
                                    timep[0][o1, 1] == track[index - out + 1, 1] and \
                                    timep[0][o1, 3] == track[index - out + 1, 3]:
                                o1_true = o1
                                break
                        for o2 in range(len(timep[1])):
                            if index - out + 2 < len(track) and \
                                    timep[1][o2, 0] == track[index - out + 2, 0] and \
                                    timep[1][o2, 1] == track[index - out + 2, 1] and \
                                    timep[1][o2, 3] == track[index - out + 2, 3]:
                                o2_true = o2
                                break

                        for o1 in range(len(timep[0])):
                            if index - out < len(track) and inSide(timep[0][o1], track[index - out],
                                                                   xyidp, gate,
                                                                   dv, gatez, is3D):
                                ytrack1 = []
                                ytrack1.extend(xtrack)
                                ytrack1.append(timep[0][o1])
                                temp_o1, disp_o1 = featureExtractY(ytrack1, s, features, is3D)
                                xyidp_o1 = disp_o1
                                for o2 in range(len(timep[1])):
                                    if inSide(timep[1][o2], timep[0][o1], xyidp_o1, gate, dv, gatez,
                                              is3D):
                                        ytrack2 = []
                                        ytrack2.extend(ytrack1)
                                        ytrack2.append(timep[1][o2])
                                        ytemp, _ = featureExtractY(ytrack2, s, features, is3D)
                                        temp_o2.append([temp_o1, ytemp])
                                        if o1 == o1_true and o2 == o2_true:
                                            o2_p.append(1)
                                        else:
                                            o2_p.append(0)

                                # only pass o2 frame
                                ytrack2 = []
                                ytrack2.extend(ytrack1)
                                ytrack2.append(pf)
                                ytemp, _ = featureExtractY(ytrack2, s, features, is3D)
                                temp_o2.append([temp_o1, ytemp])
                                if o1 == o1_true and o2_true == -1:
                                    o2_p.append(1)
                                else:
                                    o2_p.append(0)

                        # only pass o1 frame
                        ytrack1 = []
                        ytrack1.extend(xtrack)
                        ytrack1.append(pf)
                        temp_o1, _ = featureExtractY(ytrack1, s, features, is3D)
                        for o2 in range(len(timep[1])):
                            if index - out < len(track) and inSide(timep[1][o2], track[index - out],
                                                                   xyidp, gate,
                                                                   dv, gatez, is3D):
                                ytrack2 = []
                                ytrack2.extend(ytrack1)
                                ytrack2.append(timep[1][o2])
                                ytemp, _ = featureExtractY(ytrack2, s, features, is3D)
                                temp_o2.append([temp_o1, ytemp])
                                if o1_true == -1 and o2 == o2_true:
                                    o2_p.append(1)
                                else:
                                    o2_p.append(0)

                        # pass all frame
                        ytrack2 = []
                        ytrack2.extend(ytrack1)
                        ytrack2.append(pf)
                        ytemp, _ = featureExtractY(ytrack2, s, features, is3D)
                        temp_o2.append([temp_o1, ytemp])
                        if o1_true == -1 and o2_true == -1:
                            o2_p.append(1)
                        else:
                            o2_p.append(0)

                        # create output x, y, t, p
                        for i in range(len(o2_p)):
                            rp = np.random.randint(0, p2)
                            if o2_p[i] == 0:
                                if (pi in [0, 3, 6]) or \
                                        (pi not in [0, 3, 6] and rp == 0):
                                    p_f.append([1, 0])
                                    x_f.append(temp)
                                    l_f.append(tlastpos)
                                    y_f.append(temp_o2[i])
                                    truetemp = []
                                    oi = 1
                                    if 0 < index - out + oi < len(track):
                                        truetemp.append(track[index - out + oi, 0])
                                        truetemp.append(track[index - out + oi, 1])
                                        if is3D:
                                            truetemp.append(track[index - out + oi, 3])
                                    else:
                                        tx, ty, tz = avedisp(track, index - out + oi)
                                        truetemp.append(track[index - out, 0] + oi * tx)
                                        truetemp.append(track[index - out, 1] + oi * ty)
                                        if is3D:
                                            truetemp.append(track[index - out, 3] + oi * tz)
                                    t_f.append(truetemp)
                                    dis = 0
                                    for jo in range(out):
                                        if is3D:
                                            if temp_o2[i][jo][-1] == -1:
                                                dis += math.sqrt(
                                                    gate * gate * 2 + gatez * gatez) * (jo + 1)
                                            else:
                                                dis += distance2(temp_o2[i][jo][-3:], truetemp,
                                                                 is3D)
                                        else:
                                            if temp_o2[i][jo][-1] == -1:
                                                dis += math.sqrt(gate * gate * 2) * (jo + 1)
                                            else:
                                                dis += distance2(temp_o2[i][jo][-2:], truetemp,
                                                                 is3D)
                                    d_f.append(dis)
                            else:
                                p_t.append([0, 1])
                                x_t.append(temp)
                                l_t.append(tlastpos)
                                y_t.append(temp_o2[i])
                                truetemp = []
                                oi = 1
                                if 0 < index - out + oi < len(track):
                                    truetemp.append(track[index - out + oi, 0])
                                    truetemp.append(track[index - out + oi, 1])
                                    if is3D:
                                        truetemp.append(track[index - out + oi, 3])
                                else:
                                    tx, ty, tz = avedisp(track, index - out + oi)
                                    truetemp.append(track[index - out, 0] + oi * tx)
                                    truetemp.append(track[index - out, 1] + oi * ty)
                                    if is3D:
                                        truetemp.append(track[index - out, 3] + oi * tz)
                                t_t.append(truetemp)
                                d_t.append(0)
                    elif out == 3:
                        # for the third next frame
                        o3_p = []  # indexes of detections in gate
                        temp_o3 = []
                        o1_true = -1
                        o2_true = -1
                        o3_true = -1
                        for o1 in range(len(timep[0])):
                            if index - out + 1 < len(track) and \
                                    timep[0][o1, 0] == track[index - out + 1, 0] and \
                                    timep[0][o1, 1] == track[index - out + 1, 1] and \
                                    timep[0][o1, 3] == track[index - out + 1, 3]:
                                o1_true = o1
                                break
                        for o2 in range(len(timep[1])):
                            if index - out + 2 < len(track) and \
                                    timep[1][o2, 0] == track[index - out + 2, 0] and \
                                    timep[1][o2, 1] == track[index - out + 2, 1] and \
                                    timep[1][o2, 3] == track[index - out + 2, 3]:
                                o2_true = o2
                                break
                        for o3 in range(len(timep[2])):
                            if index - out + 3 < len(track) and \
                                    timep[2][o3, 0] == track[index - out + 3, 0] and \
                                    timep[2][o3, 1] == track[index - out + 3, 1] and \
                                    timep[2][o3, 3] == track[index - out + 3, 3]:
                                o3_true = o3
                                break

                        for o1 in range(len(timep[0])):
                            if index - out < len(track) and inSide(timep[0][o1], track[index - out],
                                                                   xyidp, gate,
                                                                   dv, gatez, is3D):
                                ytrack1 = []
                                ytrack1.extend(xtrack)
                                ytrack1.append(timep[0][o1])
                                temp_o1, disp_o1 = featureExtractY(ytrack1, s, features, is3D)
                                xyidp_o1 = disp_o1

                                for o2 in range(len(timep[1])):
                                    if inSide(timep[1][o2], timep[0][o1], xyidp_o1, gate, dv, gatez,
                                              is3D):
                                        ytrack2 = []
                                        ytrack2.extend(ytrack1)
                                        ytrack2.append(timep[1][o2])
                                        temp_o2, disp_o2 = featureExtractY(ytrack2, s, features,
                                                                           is3D)
                                        xyidp_o2 = disp_o2
                                        for o3 in range(len(timep[2])):
                                            if inSide(timep[2][o3], timep[1][o2], xyidp_o2, gate,
                                                      dv, gatez, is3D):
                                                ytrack3 = []
                                                ytrack3.extend(ytrack2)
                                                ytrack3.append(timep[2][o3])
                                                ytemp, _ = featureExtractY(ytrack3, s, features,
                                                                           is3D)

                                                temp_o3.append([temp_o1, temp_o2, ytemp])
                                                if o1 == o1_true and o2 == o2_true and o3 == o3_true:
                                                    o3_p.append(1)
                                                else:
                                                    o3_p.append(0)
                                        # only pass o3 frame
                                        ytrack3 = []
                                        ytrack3.extend(ytrack2)
                                        ytrack3.append(pf)
                                        ytemp, _ = featureExtractY(ytrack3, s, features, is3D)

                                        temp_o3.append([temp_o1, temp_o2, ytemp])
                                        if o1 == o1_true and o2 == o2_true and o3_true == -1:
                                            o3_p.append(1)
                                        else:
                                            o3_p.append(0)

                                # only pass o2 frame
                                ytrack2 = []
                                ytrack2.extend(ytrack1)
                                ytrack2.append(pf)
                                temp_o2, _ = featureExtractY(ytrack2, s, features, is3D)
                                for o3 in range(len(timep[2])):
                                    if inSide(timep[2][o3], timep[0][o1], xyidp_o1, gate, dv, gatez,
                                              is3D):
                                        ytrack3 = []
                                        ytrack3.extend(ytrack2)
                                        ytrack3.append(timep[2][o3])
                                        ytemp, _ = featureExtractY(ytrack3, s, features, is3D)

                                        temp_o3.append([temp_o1, temp_o2, ytemp])
                                        if o1 == o1_true and o2_true == -1 and o3 == o3_true:
                                            o3_p.append(1)
                                        else:
                                            o3_p.append(0)

                                # pass o2 and o3 frame
                                ytrack3 = []
                                ytrack3.extend(ytrack2)
                                ytrack3.append(pf)
                                ytemp, _ = featureExtractY(ytrack3, s, features, is3D)

                                temp_o3.append([temp_o1, temp_o2, ytemp])
                                if o1 == o1_true and o2_true == -1 and o3_true == -1:
                                    o3_p.append(1)
                                else:
                                    o3_p.append(0)

                        # only pass o1 frame
                        ytrack1 = []
                        ytrack1.extend(xtrack)
                        ytrack1.append(pf)
                        temp_o1, _ = featureExtractY(ytrack1, s, features, is3D)

                        for o2 in range(len(timep[1])):
                            if index - out < len(track) and inSide(timep[1][o2], track[index - out],
                                                                   xyidp, gate,
                                                                   dv, gatez, is3D):
                                ytrack2 = []
                                ytrack2.extend(ytrack1)
                                ytrack2.append(timep[1][o2])
                                temp_o2, disp_o2 = featureExtractY(ytrack2, s, features, is3D)
                                xyidp_o2 = disp_o2
                                for o3 in range(len(timep[2])):
                                    if inSide(timep[2][o3], timep[1][o2], xyidp_o2, gate, dv, gatez,
                                              is3D):
                                        ytrack3 = []
                                        ytrack3.extend(ytrack2)
                                        ytrack3.append(timep[2][o3])
                                        ytemp, _ = featureExtractY(ytrack3, s, features, is3D)

                                        temp_o3.append([temp_o1, temp_o2, ytemp])
                                        if o1_true == -1 and o2 == o2_true and o3 == o3_true:
                                            o3_p.append(1)
                                        else:
                                            o3_p.append(0)
                                # pass o1 and o3 frame
                                ytrack3 = []
                                ytrack3.extend(ytrack2)
                                ytrack3.append(pf)
                                ytemp, _ = featureExtractY(ytrack3, s, features, is3D)

                                temp_o3.append([temp_o1, temp_o2, ytemp])
                                if o1_true == -1 and o2 == o2_true and o3_true == -1:
                                    o3_p.append(1)
                                else:
                                    o3_p.append(0)

                        # pass o1 and o2 frame
                        ytrack2 = []
                        ytrack2.extend(ytrack1)
                        ytrack2.append(pf)
                        temp_o2, _ = featureExtractY(ytrack2, s, features, is3D)

                        for o3 in range(len(timep[2])):
                            if index - out < len(track) and inSide(timep[2][o3], track[index - out],
                                                                   xyidp, gate,
                                                                   dv, gatez, is3D):
                                ytrack3 = []
                                ytrack3.extend(ytrack2)
                                ytrack3.append(timep[2][o3])
                                ytemp, _ = featureExtractY(ytrack3, s, features, is3D)

                                temp_o3.append([temp_o1, temp_o2, ytemp])
                                if o1_true == -1 and o2_true == -1 and o3 == o3_true:
                                    o3_p.append(1)
                                else:
                                    o3_p.append(0)

                        # pass all frame
                        ytrack2 = []
                        ytrack2.extend(ytrack1)
                        ytrack2.append(pf)
                        temp_o2, _ = featureExtractY(ytrack2, s, features, is3D)
                        ytrack3 = []
                        ytrack3.extend(ytrack2)
                        ytrack3.append(pf)
                        ytemp, _ = featureExtractY(ytrack3, s, features, is3D)

                        temp_o3.append([temp_o1, temp_o2, ytemp])
                        if o1_true == -1 and o2_true == -1 and o3_true == -1:
                            o3_p.append(1)
                        else:
                            o3_p.append(0)

                        # create output x, y, t, p
                        for i in range(len(o3_p)):
                            rp = np.random.randint(0, p3)
                            if o3_p[i] == 0:
                                if (pi in [0, 3, 6]) or \
                                        (pi not in [0, 3, 6] and rp == 0):
                                    p_f.append([1, 0])
                                    x_f.append(temp)
                                    l_f.append(tlastpos)
                                    y_f.append(temp_o3[i])
                                    truetemp = []
                                    oi = 1
                                    if 0 < index - out + oi < len(track):
                                        truetemp.append(track[index - out + oi, 0])
                                        truetemp.append(track[index - out + oi, 1])
                                        if is3D:
                                            truetemp.append(track[index - out + oi, 3])
                                    else:
                                        tx, ty, tz = avedisp(track, index - out + oi)
                                        truetemp.append(track[index - out, 0] + oi * tx)
                                        truetemp.append(track[index - out, 1] + oi * ty)
                                        if is3D:
                                            truetemp.append(track[index - out, 3] + oi * tz)
                                    t_f.append(truetemp)
                                    dis = 0
                                    for jo in range(out):
                                        if is3D:
                                            if temp_o3[i][jo][-1] == -1:
                                                dis += math.sqrt(
                                                    gate * gate * 2 + gatez * gatez) * (jo + 1)
                                            else:
                                                dis += distance2(temp_o3[i][jo][-3:], truetemp,
                                                                 is3D)
                                        else:
                                            if temp_o3[i][jo][-1] == -1:
                                                dis += math.sqrt(gate * gate * 2) * (jo + 1)
                                            else:
                                                dis += distance2(temp_o3[i][jo][-2:], truetemp,
                                                                 is3D)
                                    d_f.append(dis)
                            else:
                                p_t.append([0, 1])
                                x_t.append(temp)
                                l_t.append(tlastpos)
                                y_t.append(temp_o3[i])
                                truetemp = []
                                oi = 1
                                if 0 < index - out + oi < len(track):
                                    truetemp.append(track[index - out + oi, 0])
                                    truetemp.append(track[index - out + oi, 1])
                                    if is3D:
                                        truetemp.append(track[index - out + oi, 3])
                                else:
                                    tx, ty, tz = avedisp(track, index - out + oi)
                                    truetemp.append(track[index - out, 0] + oi * tx)
                                    truetemp.append(track[index - out, 1] + oi * ty)
                                    if is3D:
                                        truetemp.append(track[index - out, 3] + oi * tz)
                                t_t.append(truetemp)
                                d_t.append(0)
            print(len(x_t), len(x_f))

    if istrain:
        np.savez_compressed(pathsave + 'ax_F' + str(s) + str(out), x=x_f, y=y_f, t=t_f, p=p_f,
                            l=l_f, d=d_f)
        np.savez_compressed(pathsave + 'ax_T' + str(s) + str(out), x=x_t, y=y_t, t=t_t, p=p_t,
                            l=l_t, d=d_t)
    else:
        np.savez_compressed(pathsave + 'bx_F' + str(s) + str(out), x=x_f, y=y_f, t=t_f, p=p_f,
                            l=l_f, d=d_f)
        np.savez_compressed(pathsave + 'bx_T' + str(s) + str(out), x=x_t, y=y_t, t=t_t, p=p_t,
                            l=l_t, d=d_t)


# splitting data into training-testing-validation
def train_test(scenario, s, out, iter, istrain='tr', togit=0, dodelete=False, isspt=True):
    if togit == 1:
        pathsave = '/home/yaoyao/IDEA/dl/data/MTJ/RNNMD/' + scenario + '/RNNmd' + str(iter)
    elif togit == 0:
        pathsave = '/media/yyao/work/data/MTJ/RNNMD/' + scenario + '/RNNmd' + str(iter)
    else:
        pathsave = '/nfs/home4/yao88/IDEA/dl/data/MTJ/RNNMD/' + scenario + '/RNNmd' + str(iter)

    fbs = 10000

    if istrain == 'train':
        print('Combining...')
        loaded = np.load(pathsave + '/after comb/' + 'ax_T' + str(s) + str(out) + '.npz')
        x_t = loaded['x']
        y_t = loaded['y']
        t_t = loaded['t']
        p_t = loaded['p']
        l_t = loaded['l']
        d_t = loaded['d']
        loaded = np.load(pathsave + '/after comb/' + 'ax_F' + str(s) + str(out) + '.npz')
        x_f = loaded['x']
        y_f = loaded['y']
        t_f = loaded['t']
        p_f = loaded['p']
        l_f = loaded['l']
        d_f = loaded['d']
        permutation = np.random.permutation(p_t.shape[0])
        x_t = x_t[permutation]
        y_t = y_t[permutation]
        t_t = t_t[permutation]
        p_t = p_t[permutation]
        l_t = l_t[permutation]
        d_t = d_t[permutation]
        print('positive vs. negative: ' + str(x_t.shape[0]) + ':' + str(x_f.shape[0]))
        if out == 3:
            permutation = np.random.permutation(p_f.shape[0])[:int(p_f.shape[0] * 0.6)]
        else:
            permutation = np.random.permutation(p_f.shape[0])

        x_f = x_f[permutation]
        y_f = y_f[permutation]
        t_f = t_f[permutation]
        p_f = p_f[permutation]
        l_f = l_f[permutation]
        d_f = d_f[permutation]
        print('positive vs. negative: ' + str(x_t.shape[0]) + ':' + str(x_f.shape[0]))

        n = p_f.shape[0] // fbs
        nt = p_t.shape[0] // fbs
        n9 = int(n * 0.9)
        nt9 = int(nt * 0.9)

        if out == 2:
            if scenario == 'VIRUS':
                ratio = int(nt9 * 2.5)
            else:
                ratio = int(nt9 * 3)
        elif out == 3:
            if scenario == 'VIRUS':
                ratio = int(nt9 * 2.3)
            else:
                ratio = int(nt9 * 3)
        else:
            if scenario == 'VIRUS':
                ratio = int(nt9 * 3)
            else:
                ratio = int(n9 * 0.7)
        if ratio % 2 != 0:
            ratio += 1
        print(n9, nt9, ratio)

        x = []
        y = []
        t = []
        p = []
        l = []
        d = []
        for i in range(ratio):
            xt = []
            yt = []
            tt = []
            pt = []
            lt = []
            dt = []
            ptr = i * fbs
            ptrt = i % nt9 * fbs
            xt.extend(x_t[ptrt:ptrt + fbs])
            yt.extend(y_t[ptrt:ptrt + fbs])
            tt.extend(t_t[ptrt:ptrt + fbs])
            pt.extend(p_t[ptrt:ptrt + fbs])
            lt.extend(l_t[ptrt:ptrt + fbs])
            dt.extend(d_t[ptrt:ptrt + fbs])
            xt.extend(x_f[ptr:ptr + fbs])
            yt.extend(y_f[ptr:ptr + fbs])
            tt.extend(t_f[ptr:ptr + fbs])
            pt.extend(p_f[ptr:ptr + fbs])
            lt.extend(l_f[ptr:ptr + fbs])
            dt.extend(d_f[ptr:ptr + fbs])
            xt = np.asarray(xt)
            yt = np.asarray(yt)
            tt = np.asarray(tt)
            pt = np.asarray(pt)
            lt = np.asarray(lt)
            dt = np.asarray(dt)
            permutation = np.random.permutation(xt.shape[0])
            xt = xt[permutation]
            yt = yt[permutation]
            tt = tt[permutation]
            pt = pt[permutation]
            lt = lt[permutation]
            dt = dt[permutation]
            dt = np.reshape(dt, (len(dt), 1))
            if len(x) == 0:
                x = (xt)
                y = (yt)
                t = (tt)
                p = (pt)
                l = (lt)
                d = (dt)
            else:
                x = np.vstack((x, xt))
                y = np.vstack((y, yt))
                t = np.vstack((t, tt))
                p = np.vstack((p, pt))
                l = np.vstack((l, lt))
                d = np.vstack((d, dt))
            if i % 50 == 0:
                print(x.shape, y.shape, t.shape, p.shape, l.shape, d.shape)
        print(x.shape, y.shape, t.shape, p.shape, l.shape, d.shape)
        np.save(pathsave + '/x_train' + str(s) + str(out) + '.npy', x)
        np.save(pathsave + '/y_train' + str(s) + str(out) + '.npy', y)
        np.save(pathsave + '/t_train' + str(s) + str(out) + '.npy', t)
        np.save(pathsave + '/p_train' + str(s) + str(out) + '.npy', p)
        np.save(pathsave + '/l_train' + str(s) + str(out) + '.npy', l)
        np.save(pathsave + '/d_train' + str(s) + str(out) + '.npy', d)
        print('Training data sizes: ' + str(x.shape[0]))

        x = np.reshape(x, (x.shape[0] * s, x.shape[2]))
        y = np.reshape(y, (y.shape[0] * out, y.shape[2]))
        res = []
        for i in range(x.shape[1]):
            res.append([np.std(x[:, i]), np.mean(x[:, i]), np.std(y[:, i]), np.mean(y[:, i])])
            # res.append([255.5, 255.5])
            # res.append([255.5, 255.5])
            # res.append([5.409,4.5])
        np.save(pathsave + '/r' + str(s) + str(out) + '.npy', res)

        x = []
        y = []
        t = []
        p = []
        l = []
        d = []
        ptrt9 = int(nt9 * fbs)
        nt99 = (p_t.shape[0] - ptrt9) // fbs
        for i in range(n9, n):
            ptr = i * fbs
            ptrt = ptrt9 + (i - n9) % nt99 * fbs
            x.extend(x_t[ptrt:ptrt + fbs])
            y.extend(y_t[ptrt:ptrt + fbs])
            t.extend(t_t[ptrt:ptrt + fbs])
            p.extend(p_t[ptrt:ptrt + fbs])
            l.extend(l_t[ptrt:ptrt + fbs])
            d.extend(d_t[ptrt:ptrt + fbs])
            x.extend(x_f[ptr:ptr + fbs])
            y.extend(y_f[ptr:ptr + fbs])
            t.extend(t_f[ptr:ptr + fbs])
            p.extend(p_f[ptr:ptr + fbs])
            l.extend(l_f[ptr:ptr + fbs])
            d.extend(d_f[ptr:ptr + fbs])
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32)
        p = np.asarray(p, dtype=np.float32)
        l = np.asarray(l, dtype=np.float32)
        d = np.asarray(d, dtype=np.float32)
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation]
        y = y[permutation]
        t = t[permutation]
        p = p[permutation]
        l = l[permutation]
        d = d[permutation]
        np.save(pathsave + '/x_val' + str(s) + str(out) + '.npy', x)
        np.save(pathsave + '/y_val' + str(s) + str(out) + '.npy', y)
        np.save(pathsave + '/t_val' + str(s) + str(out) + '.npy', t)
        np.save(pathsave + '/p_val' + str(s) + str(out) + '.npy', p)
        np.save(pathsave + '/l_val' + str(s) + str(out) + '.npy', l)
        np.save(pathsave + '/d_val' + str(s) + str(out) + '.npy', d)
        print('Val data sizes: ' + str(x.shape[0]))
        # num = [nt9*5, 1]
        # np.save(pathsave + '/number' + str(s) + str(out) + '.npy', num)
    else:
        loaded = np.load(pathsave + '/after comb/' + 'bx_T' + str(s) + str(out) + '.npz')
        x_t = loaded['x']
        y_t = loaded['y']
        t_t = loaded['t']
        p_t = loaded['p']
        l_t = loaded['l']
        d_t = loaded['d']
        loaded = np.load(pathsave + '/after comb/' + 'bx_F' + str(s) + str(out) + '.npz')
        x_f = loaded['x']
        y_f = loaded['y']
        t_f = loaded['t']
        p_f = loaded['p']
        l_f = loaded['l']
        d_f = loaded['d']
        permutation = np.random.permutation(p_t.shape[0])
        x_t = x_t[permutation]
        y_t = y_t[permutation]
        t_t = t_t[permutation]
        p_t = p_t[permutation]
        l_t = l_t[permutation]
        d_t = d_t[permutation]
        permutation = np.random.permutation(p_f.shape[0])
        x_f = x_f[permutation]
        y_f = y_f[permutation]
        t_f = t_f[permutation]
        p_f = p_f[permutation]
        l_f = l_f[permutation]
        d_f = d_f[permutation]
        print('positive vs. negative: ' + str(x_t.shape[0]) + ':' + str(x_f.shape[0]))
        n = p_f.shape[0] // fbs
        nt = p_t.shape[0] // fbs
        x = []
        y = []
        t = []
        p = []
        l = []
        d = []
        for i in range(n):
            ptr = i * fbs
            ptrt = i % nt * fbs
            x.extend(x_t[ptrt:ptrt + fbs])
            y.extend(y_t[ptrt:ptrt + fbs])
            t.extend(t_t[ptrt:ptrt + fbs])
            p.extend(p_t[ptrt:ptrt + fbs])
            l.extend(l_t[ptrt:ptrt + fbs])
            d.extend(d_t[ptrt:ptrt + fbs])
            x.extend(x_f[ptr:ptr + fbs])
            y.extend(y_f[ptr:ptr + fbs])
            t.extend(t_f[ptr:ptr + fbs])
            p.extend(p_f[ptr:ptr + fbs])
            l.extend(l_f[ptr:ptr + fbs])
            d.extend(d_f[ptr:ptr + fbs])
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32)
        p = np.asarray(p, dtype=np.float32)
        l = np.asarray(l, dtype=np.float32)
        d = np.asarray(d, dtype=np.float32)
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation]
        y = y[permutation]
        t = t[permutation]
        p = p[permutation]
        l = l[permutation]
        d = d[permutation]
        np.save(pathsave + '/x_test' + str(s) + str(out) + '.npy', x)
        np.save(pathsave + '/y_test' + str(s) + str(out) + '.npy', y)
        np.save(pathsave + '/t_test' + str(s) + str(out) + '.npy', t)
        np.save(pathsave + '/p_test' + str(s) + str(out) + '.npy', p)
        np.save(pathsave + '/l_test' + str(s) + str(out) + '.npy', l)
        np.save(pathsave + '/d_test' + str(s) + str(out) + '.npy', d)
        print('Test data sizes: ' + str(x.shape[0]))
        # num = np.load(pathsave + 'number' + str(s) + str(out) + '.npy')
        # nums = [num[0], num[1], n]
        # np.save(pathsave + 'number' + str(s) + str(out) + '.npy', nums)


# data normalization
def dataNormalization(scenario, s, out, iter, togit=0, isspt=False):
    if togit == 1:
        pathsave = '/home/yaoyao/IDEA/dl/data/MTJ/RNNMD/' + scenario + '/RNNmd' + str(iter) + '/'
    elif togit == 0:
        pathsave = '/media/yyao/work/data/MTJ/RNNMD/' + scenario + '/RNNmd' + str(iter) + '/'
    else:
        pathsave = '/nfs/home4/yao88/IDEA/dl/data/MTJ/RNNMD/' + scenario + '/RNNmd' + str(
            iter) + '/'

    if isspt:
        res = np.load(pathsave + 'r' + str(s) + str(out) + '.npy')
    else:
        x = []
        y = []
        x.extend(np.load(pathsave + 'x_train' + str(s) + str(out) + '.npy'))
        y.extend(np.load(pathsave + 'y_train' + str(s) + str(out) + '.npy'))
        x = np.asarray(x)
        y = np.asarray(y)
        x = np.reshape(x, (x.shape[0] * s, x.shape[2]))
        y = np.reshape(y, (y.shape[0] * out, y.shape[2]))

        res = []
        for i in range(x.shape[1]):
            res.append([np.std(x[:, i]), np.mean(x[:, i]), np.std(y[:, i]), np.mean(y[:, i])])
            # res.append([255.5, 255.5])
            # res.append([255.5, 255.5])
            # res.append([5.409,4.5])
        np.save(pathsave + 'r' + str(s) + str(out) + '.npy', res)

    for name in ['train', 'val', 'test']:
        x = np.load(pathsave + 'x_' + name + str(s) + str(out) + '.npy')
        y = np.load(pathsave + 'y_' + name + str(s) + str(out) + '.npy')
        x = np.reshape(x, (x.shape[0] * s, x.shape[2]))
        y = np.reshape(y, (y.shape[0] * out, y.shape[2]))
        for i in range(x.shape[1]):
            x[:, i] -= res[i][1]
            x[:, i] /= res[i][0]
        for i in range(y.shape[1]):
            y[:, i] -= res[i][3]
            y[:, i] /= res[i][2]
        x = np.reshape(x, (int(x.shape[0] / s), s, x.shape[1]))
        y = np.reshape(y, (int(y.shape[0] / out), out, y.shape[1]))
        np.save(pathsave + 'x_' + name + str(s) + str(out) + '.npy', x)
        np.save(pathsave + 'y_' + name + str(s) + str(out) + '.npy', y)
    print('...Data Normalized!!!')


# create training-testing data for each case
def rnnMdData(scenarios, ss, oo, ff, togit=0, isspt=False):
    if togit == 1:
        pp = '/home/yaoyao/IDEA/dl/'
    elif togit == 0:
        pp = '/home/yyao/IDEA/dl/'
    else:
        pp = '/nfs/home4/yao88/IDEA/dl/'
    for scenario in scenarios:
        for iter in ff:
            close = 1
            dv = ''
            if iter == 0:
                features = [False, False, False, False, False, True]
            else:
                features = [True, True, True, True, True, True]

            for s in ss:
                for out in oo:
                    gate = 15
                    gatez = 3
                    stritem = 'Creating ' + scenario + '-' + ' S' + str(s) \
                              + ' H' + str(out) + ' I' + str(iter) \
                              + '_G' + str(gate) + '_' + str(gatez) + '_C' + str(close)
                    print(stritem)
                    start = time.time()
                    paths = []
                    for snrid in [4, 7]:
                        paths.append(pp + 'data/MTJ_RNN/data/' + scenario + ' snr ' + str(
                            snrid) + ' density low.detections.xml.txt')
                        paths.append(pp + 'data/MTJ_RNN/data/' + scenario + ' snr ' + str(
                            snrid) + ' density mid.detections.xml.txt')
                        paths.append(pp + 'data/MTJ_RNN/data/' + scenario + ' snr ' + str(
                            snrid) + ' density high.detections.xml.txt')

                    ist = True
                    createRNNdataPloc(scenario, 7, paths, s, out, gate, close, iter,
                                      features, ist, dv, gatez, togit, isspt)

                    paths = []
                    for snrid in [1, 2]:
                        paths.append(pp + 'data/MTJ_RNN/data/' + scenario + ' snr ' + str(
                            snrid) + ' density low.detections.xml.txt')
                        paths.append(pp + 'data/MTJ_RNN/data/' + scenario + ' snr ' + str(
                            snrid) + ' density mid.detections.xml.txt')
                        paths.append(pp + 'data/MTJ_RNN/data/' + scenario + ' snr ' + str(
                            snrid) + ' density high.detections.xml.txt')

                    ist = False
                    createRNNdataPloc(scenario, 1, paths, s, out, gate, close, iter,
                                      features, ist, dv, gatez, togit, isspt)

                    os.system('curl -s -X POST https://api.telegram.org/'
                              'bot459863485:AAEPUJsNI0wkf3iI8RfweGMR9u0rKiSHvuc/'
                              'sendMessage -F chat_id=385302225 -F text=' + stritem)

                    print('Time : ', time.time() - start)
                    print('...Finished!!!\n')


# splitting and normalize training-testing data for each case
def rnnMdDataSplit(scenarios, ss, oo, ff, normalise, togit=0, isspt=True):
    for scenario in scenarios:
        for iter in ff:
            for s in ss:
                for out in oo:
                    stritem = 'Finishing ' + scenario + ' S' + str(s) + ' H' + str(out) \
                              + ' I' + str(iter) + ' N' + str(normalise)
                    print(stritem)
                    start = time.time()
                    train_test(scenario, s, out, iter, 'train', togit, False, isspt)
                    train_test(scenario, s, out, iter, 'test', togit, False, isspt)
                    if normalise:
                        dataNormalization(scenario, s, out, iter, togit, isspt)
                    # os.system('curl -s -X POST https://api.telegram.org/'
                    #          'bot459863485:AAEPUJsNI0wkf3iI8RfweGMR9u0rKiSHvuc/'
                    #          'sendMessage -F chat_id=385302225 -F text=' + str(stritem))
                    print('Time : ', time.time() - start)
                    print('...Finished!!!\n')


def getSequence(position, true, timep, tc, gate, gatez, dv='', togit=False):
    if tc.scenario == 'VIRUS':
        is3D = True
        n = 3
    else:
        is3D = False
        n = 2

    s = 10
    if togit:
        pathsave = '/home/yaoyao/IDEA/dl/data/MTJ/RNNMD/' + tc.scenario + '/RNNmd1' + '/'
    else:
        pathsave = '/media/yyao/work/data/MTJ/RNNMD/' + tc.scenario + '/RNNmd1' + '/'
    r = np.load(pathsave + 'r' + str(s) + str(tc.lstmo) + '.npy')  # normalization

    if tc.feature < 6:
        nciter = 0
    else:
        nciter = 1

    if nciter == 0:
        features = [False, False, False, False, False, True]
    else:
        features = [True, True, True, True, True, True]

    x = []  # track features
    y = []  # detection features
    l = []  # x's last positions
    pd = []  # detection probablility
    realy = []  # the real coordinates of detection
    realp = []  # true or not
    pf = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    track = np.asarray(position)
    index = len(track) - 1

    xtrack = []
    for i in range(index - tc.lstms, index + 1):  # index+1 is the next true!
        if i >= 0:
            xtrack.append(track[i])
        else:
            xtrack.append(pf)
    temp, xyidp, tlastpos = featureExtractX(xtrack, tc.lstms, features, is3D)

    truepoints = []
    for z in range(1, tc.lstmo + 1):
        truepoints.append(true[true[:, 2] == track[index, 2] + z])
    truepoints = np.asarray(truepoints)

    if tc.lstmo == 1:
        # for the immediate next frame
        o1_true = -1
        for o1 in range(len(timep[0])):
            if len(truepoints[0]) > 0 and \
                    timep[0][o1, 0] == truepoints[0][0, 0] and \
                    timep[0][o1, 1] == truepoints[0][0, 1] and \
                    timep[0][o1, 3] == truepoints[0][0, 3]:
                o1_true = o1
                break

        for o1 in range(len(timep[0])):
            if index < len(track) and inSide(timep[0][o1], track[index], xyidp, gate, dv, gatez,
                                             is3D):
                ytrack = []
                ytrack.extend(xtrack)
                ytrack.append(timep[0][o1])
                ytemp, _ = featureExtractY(ytrack, tc.lstms, features, is3D)
                temp_o1 = [ytemp]
                temp_o1_realy = [[timep[0][o1, 0], timep[0][o1, 1], timep[0][o1, 2],
                                  timep[0][o1, 3], timep[0][o1, 4], timep[0][o1, 5], 0, 0, 0, o1]]
                x.append(temp)
                l.append(tlastpos)
                y.append(temp_o1)
                realy.append(temp_o1_realy)
                pd.append(1)
                if o1 == o1_true:
                    realp.append(1)
                else:
                    realp.append(0)

        # include pass frame [lstmo+1 times of gate] for classification :
        # have no detections in next frame
        ytrack = []
        ytrack.extend(xtrack)
        ytrack.append(pf)
        ytemp, _ = featureExtractY(ytrack, tc.lstms, features, is3D)
        temp_o1 = [ytemp]
        temp_o1_realy = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[0])]]
        x.append(temp)
        l.append(tlastpos)
        y.append(temp_o1)
        realy.append(temp_o1_realy)
        pd.append(1)
        if o1_true == -1:
            realp.append(1)
        else:
            realp.append(0)

    elif tc.lstmo == 2:
        # for second next frame
        o1_true = -1
        o2_true = -1
        for o1 in range(len(timep[0])):
            if len(truepoints[0]) > 0 and \
                    timep[0][o1, 0] == truepoints[0][0, 0] and \
                    timep[0][o1, 1] == truepoints[0][0, 1] and \
                    timep[0][o1, 3] == truepoints[0][0, 3]:
                o1_true = o1
                break
        for o2 in range(len(timep[1])):
            if len(truepoints[1]) > 0 and \
                    timep[1][o2, 0] == truepoints[1][0, 0] and \
                    timep[1][o2, 1] == truepoints[1][0, 1] and \
                    timep[1][o2, 3] == truepoints[1][0, 3]:
                o2_true = o2
                break

        for o1 in range(len(timep[0])):
            if index < len(track) and inSide(timep[0][o1], track[index], xyidp, gate, dv, gatez,
                                             is3D):
                ytrack1 = []
                ytrack1.extend(xtrack)
                ytrack1.append(timep[0][o1])
                temp_o1, xyidp_o1 = featureExtractY(ytrack1, tc.lstms, features, is3D)
                for o2 in range(len(timep[1])):
                    if inSide(timep[1][o2], timep[0][o1], xyidp_o1, gate, dv, gatez, is3D):
                        ytrack2 = []
                        ytrack2.extend(ytrack1)
                        ytrack2.append(timep[1][o2])
                        ytemp, _ = featureExtractY(ytrack2, tc.lstms, features, is3D)
                        temp_o2 = [temp_o1, ytemp]

                        temp_o2_realy = [[timep[0][o1, 0], timep[0][o1, 1], timep[0][o1, 2],
                                          timep[0][o1, 3],
                                          timep[0][o1, 4], timep[0][o1, 5], 0, 0, 0, o1],
                                         [timep[1][o2, 0], timep[1][o2, 1], timep[1][o2, 2],
                                          timep[1][o2, 3],
                                          timep[1][o2, 4], timep[1][o2, 5], 0, 0, 0, o2]]
                        x.append(temp)
                        l.append(tlastpos)
                        y.append(temp_o2)
                        realy.append(temp_o2_realy)
                        pd.append(1)
                        if o1 == o1_true and o2 == o2_true:
                            realp.append(1)
                        else:
                            realp.append(0)
                # only pass o2 frame
                ytrack2 = []
                ytrack2.extend(ytrack1)
                ytrack2.append(pf)
                ytemp, _ = featureExtractY(ytrack2, tc.lstms, features, is3D)
                temp_o2 = [temp_o1, ytemp]
                temp_o2_realy = [
                    [timep[0][o1, 0], timep[0][o1, 1], timep[0][o1, 2], timep[0][o1, 3],
                     timep[0][o1, 4], timep[0][o1, 5], 0, 0, 0, o1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[1])]]
                x.append(temp)
                l.append(tlastpos)
                y.append(temp_o2)
                realy.append(temp_o2_realy)
                pd.append(1)
                if o1 == o1_true and o2_true == -1:
                    realp.append(1)
                else:
                    realp.append(0)
        # only pass o1 frame
        ytrack1 = []
        ytrack1.extend(xtrack)
        ytrack1.append(pf)
        temp_o1, _ = featureExtractY(ytrack1, tc.lstms, features, is3D)
        for o2 in range(len(timep[1])):
            if index < len(track) and inSide(timep[1][o2], track[index], xyidp, gate, dv, gatez,
                                             is3D):
                ytrack2 = []
                ytrack2.extend(ytrack1)
                ytrack2.append(timep[1][o2])
                ytemp, _ = featureExtractY(ytrack2, tc.lstms, features, is3D)
                temp_o2 = [temp_o1, ytemp]
                temp_o2_realy = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[0])],
                                 [timep[1][o2, 0], timep[1][o2, 1], timep[1][o2, 2],
                                  timep[1][o2, 3],
                                  timep[1][o2, 4], timep[1][o2, 5], 0, 0, 0, o2]]
                x.append(temp)
                l.append(tlastpos)
                y.append(temp_o2)
                realy.append(temp_o2_realy)
                pd.append(1)
                if o1_true == -1 and o2 == o2_true:
                    realp.append(1)
                else:
                    realp.append(0)
        # pass all frame
        ytrack1 = []
        ytrack1.extend(xtrack)
        ytrack1.append(pf)
        temp_o1, _ = featureExtractY(ytrack1, tc.lstms, features, is3D)
        ytrack2 = []
        ytrack2.extend(ytrack1)
        ytrack2.append(pf)
        ytemp, _ = featureExtractY(ytrack2, tc.lstms, features, is3D)
        temp_o2 = [temp_o1, ytemp]
        temp_o2_realy = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[0])],
                         [-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[1])]]
        x.append(temp)
        l.append(tlastpos)
        y.append(temp_o2)
        realy.append(temp_o2_realy)
        pd.append(1)
        if o1_true == -1 and o2_true == -1:
            realp.append(1)
        else:
            realp.append(0)

    elif tc.lstmo == 3:
        # for the third next frame
        o1_true = -1
        o2_true = -1
        o3_true = -1
        for o1 in range(len(timep[0])):
            if len(truepoints[0]) > 0 and timep[0][o1, 0] == truepoints[0][0, 0] and \
                    timep[0][o1, 1] == truepoints[0][0, 1] and \
                    timep[0][o1, 3] == truepoints[0][0, 3]:
                o1_true = o1
                break
        for o2 in range(len(timep[1])):
            if len(truepoints[1]) > 0 and timep[1][o2, 0] == truepoints[1][0, 0] and \
                    timep[1][o2, 1] == truepoints[1][0, 1] and \
                    timep[1][o2, 3] == truepoints[1][0, 3]:
                o2_true = o2
                break
        for o3 in range(len(timep[2])):
            if len(truepoints[2]) > 0 and timep[2][o3, 0] == truepoints[2][0, 0] and \
                    timep[2][o3, 1] == truepoints[2][0, 1] and \
                    timep[2][o3, 3] == truepoints[2][0, 3]:
                o3_true = o3
                break

        for o1 in range(len(timep[0])):
            if index < len(track) and inSide(timep[0][o1], track[index], xyidp, gate, dv, gatez,
                                             is3D):
                ytrack1 = []
                ytrack1.extend(xtrack)
                ytrack1.append(timep[0][o1])
                temp_o1, xyidp_o1 = featureExtractY(ytrack1, tc.lstms, features, is3D)
                for o2 in range(len(timep[1])):
                    if inSide(timep[1][o2], timep[0][o1], xyidp_o1, gate, dv, gatez, is3D):
                        ytrack2 = []
                        ytrack2.extend(ytrack1)
                        ytrack2.append(timep[1][o2])
                        temp_o2, xyidp_o2 = featureExtractY(ytrack2, tc.lstms, features, is3D)
                        for o3 in range(len(timep[2])):
                            if inSide(timep[2][o3], timep[1][o2], xyidp_o2, gate, dv, gatez, is3D):
                                ytrack3 = []
                                ytrack3.extend(ytrack2)
                                ytrack3.append(timep[2][o3])
                                ytemp, _ = featureExtractY(ytrack3, tc.lstms, features, is3D)
                                temp_o3 = [temp_o1, temp_o2, ytemp]
                                temp_o3_realy = [[timep[0][o1, 0], timep[0][o1, 1], timep[0][o1, 2],
                                                  timep[0][o1, 3], timep[0][o1, 4], timep[0][o1, 5],
                                                  0, 0, 0, o1],
                                                 [timep[1][o2, 0], timep[1][o2, 1], timep[1][o2, 2],
                                                  timep[1][o2, 3],
                                                  timep[1][o2, 4], timep[1][o2, 5], 0, 0, 0, o2],
                                                 [timep[2][o3, 0], timep[2][o3, 1], timep[2][o3, 2],
                                                  timep[2][o3, 3],
                                                  timep[2][o3, 4], timep[2][o3, 5], 0, 0, 0, o3]]
                                x.append(temp)
                                l.append(tlastpos)
                                y.append(temp_o3)
                                realy.append(temp_o3_realy)
                                pd.append(1)
                                if o1 == o1_true and o2 == o2_true and o3 == o3_true:
                                    realp.append(1)
                                else:
                                    realp.append(0)

                        # only pass o3 frame
                        ytrack3 = []
                        ytrack3.extend(ytrack2)
                        ytrack3.append(pf)
                        ytemp, _ = featureExtractY(ytrack3, tc.lstms, features, is3D)
                        temp_o3 = [temp_o1, temp_o2, ytemp]
                        temp_o3_realy = [[timep[0][o1, 0], timep[0][o1, 1], timep[0][o1, 2],
                                          timep[0][o1, 3],
                                          timep[0][o1, 4], timep[0][o1, 5], 0, 0, 0, o1],
                                         [timep[1][o2, 0], timep[1][o2, 1], timep[1][o2, 2],
                                          timep[1][o2, 3],
                                          timep[1][o2, 4], timep[1][o2, 5], 0, 0, 0, o2],
                                         [-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[2])]]
                        x.append(temp)
                        l.append(tlastpos)
                        y.append(temp_o3)
                        realy.append(temp_o3_realy)
                        pd.append(1)
                        if o1 == o1_true and o2 == o2_true and o3_true == -1:
                            realp.append(1)
                        else:
                            realp.append(0)

                # only pass o2 frame
                ytrack2 = []
                ytrack2.extend(ytrack1)
                ytrack2.append(pf)
                temp_o2, _ = featureExtractY(ytrack2, tc.lstms, features, is3D)
                for o3 in range(len(timep[2])):
                    if inSide(timep[2][o3], timep[0][o1], xyidp_o1, gate, dv, gatez, is3D):
                        ytrack3 = []
                        ytrack3.extend(ytrack2)
                        ytrack3.append(timep[2][o3])
                        ytemp, _ = featureExtractY(ytrack3, tc.lstms, features, is3D)
                        temp_o3 = [temp_o1, temp_o2, ytemp]
                        temp_o3_realy = [[timep[0][o1, 0], timep[0][o1, 1], timep[0][o1, 2],
                                          timep[0][o1, 3],
                                          timep[0][o1, 4], timep[0][o1, 5], 0, 0, 0, o1],
                                         [-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[1])],
                                         [timep[2][o3, 0], timep[2][o3, 1], timep[2][o3, 2],
                                          timep[2][o3, 3],
                                          timep[2][o3, 4], timep[2][o3, 5], 0, 0, 0, o3]]
                        x.append(temp)
                        l.append(tlastpos)
                        y.append(temp_o3)
                        realy.append(temp_o3_realy)
                        pd.append(1)
                        if o1 == o1_true and o2_true == -1 and o3 == o3_true:
                            realp.append(1)
                        else:
                            realp.append(0)
                # pass o2 and o3 frame
                ytrack3 = []
                ytrack3.extend(ytrack2)
                ytrack3.append(pf)
                ytemp, _ = featureExtractY(ytrack3, tc.lstms, features, is3D)
                temp_o3 = [temp_o1, temp_o2, ytemp]
                temp_o3_realy = [
                    [timep[0][o1, 0], timep[0][o1, 1], timep[0][o1, 2], timep[0][o1, 3],
                     timep[0][o1, 4], timep[0][o1, 5], 0, 0, 0, o1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[1])],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[2])]]
                x.append(temp)
                l.append(tlastpos)
                y.append(temp_o3)
                realy.append(temp_o3_realy)
                pd.append(1)
                if o1 == o1_true and o2_true == -1 and o3_true == -1:
                    realp.append(1)
                else:
                    realp.append(0)
        # only pass o1 frame
        ytrack1 = []
        ytrack1.extend(xtrack)
        ytrack1.append(pf)
        temp_o1, _ = featureExtractY(ytrack1, tc.lstms, features, is3D)
        for o2 in range(len(timep[1])):
            if index < len(track) and inSide(timep[1][o2], track[index], xyidp, gate, dv, gatez,
                                             is3D):
                ytrack2 = []
                ytrack2.extend(ytrack1)
                ytrack2.append(timep[1][o2])
                temp_o2, xyidp_o2 = featureExtractY(ytrack2, tc.lstms, features, is3D)
                for o3 in range(len(timep[2])):
                    if inSide(timep[2][o3], timep[1][o2], xyidp_o2, gate, dv, gatez, is3D):
                        ytrack3 = []
                        ytrack3.extend(ytrack2)
                        ytrack3.append(timep[2][o3])
                        ytemp, _ = featureExtractY(ytrack3, tc.lstms, features, is3D)
                        temp_o3 = [temp_o1, temp_o2, ytemp]
                        temp_o3_realy = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[0])],
                                         [timep[1][o2, 0], timep[1][o2, 1], timep[1][o2, 2],
                                          timep[1][o2, 3],
                                          timep[1][o2, 4], timep[1][o2, 5], 0, 0, 0, o2],
                                         [timep[2][o3, 0], timep[2][o3, 1], timep[2][o3, 2],
                                          timep[2][o3, 3],
                                          timep[2][o3, 4], timep[2][o3, 5], 0, 0, 0, o3]]
                        x.append(temp)
                        l.append(tlastpos)
                        y.append(temp_o3)
                        realy.append(temp_o3_realy)
                        pd.append(1)
                        if o1_true == -1 and o2 == o2_true and o3 == o3_true:
                            realp.append(1)
                        else:
                            realp.append(0)
                # pass o1 and o3 frame
                ytrack3 = []
                ytrack3.extend(ytrack2)
                ytrack3.append(pf)
                ytemp, _ = featureExtractY(ytrack3, tc.lstms, features, is3D)
                temp_o3 = [temp_o1, temp_o2, ytemp]
                temp_o3_realy = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[0])],
                                 [timep[1][o2, 0], timep[1][o2, 1], timep[1][o2, 2],
                                  timep[1][o2, 3], timep[1][o2, 4], timep[1][o2, 5], 0, 0, 0, o2],
                                 [-1, -1, -1, -1, -1, -1, -1, -1, -1, -len(timep[2])]]
                x.append(temp)
                l.append(tlastpos)
                y.append(temp_o3)
                realy.append(temp_o3_realy)
                pd.append(1)
                if o1_true == -1 and o2 == o2_true and o3_true == -1:
                    realp.append(1)
                else:
                    realp.append(0)
        # pass o1 and o2 frame
        ytrack2 = []
        ytrack2.extend(ytrack1)
        ytrack2.append(pf)
        temp_o2, _ = featureExtractY(ytrack2, tc.lstms, features, is3D)
        for o3 in range(len(timep[2])):
            if index < len(track) and inSide(timep[2][o3], track[index], xyidp, gate, dv, gatez,
                                             is3D):
                ytrack3 = []
                ytrack3.extend(ytrack2)
                ytrack3.append(timep[2][o3])
                ytemp, _ = featureExtractY(ytrack3, tc.lstms, features, is3D)
                temp_o3 = [temp_o1, temp_o2, ytemp]
                temp_o3_realy = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[0])],
                                 [-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[1])],
                                 [timep[2][o3, 0], timep[2][o3, 1], timep[2][o3, 2],
                                  timep[2][o3, 3], timep[2][o3, 4], timep[2][o3, 5], 0, 0, 0, o3]]
                x.append(temp)
                l.append(tlastpos)
                y.append(temp_o3)
                realy.append(temp_o3_realy)
                pd.append(1)
                if o1_true == -1 and o2_true == -1 and o3 == o3_true:
                    realp.append(1)
                else:
                    realp.append(0)

        # pass all frame
        ytrack3 = []
        ytrack3.extend(ytrack2)
        ytrack3.append(pf)
        ytemp, _ = featureExtractY(ytrack3, tc.lstms, features, is3D)
        temp_o3 = [temp_o1, temp_o2, ytemp]
        temp_o3_realy = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[0])],
                         [-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[1])],
                         [-1, -1, -1, -1, -1, -1, -1, -1, -1, len(timep[2])]]
        x.append(temp)
        l.append(tlastpos)
        y.append(temp_o3)
        realy.append(temp_o3_realy)
        pd.append(1)
        if o1_true == -1 and o2_true == -1 and o3_true == -1:
            realp.append(1)
        else:
            realp.append(0)

    x = np.asarray(x, dtype=np.float32)
    realx = deepcopy(x)
    y = np.asarray(y, dtype=np.float32)
    pd = np.asarray(pd, dtype=np.float32)
    l = np.asarray(l, dtype=np.float32)
    realy = np.asarray(realy, dtype=np.float32)
    realp = np.asarray(realp, dtype=np.float32)

    x = np.reshape(x, (x.shape[0] * tc.lstms, x.shape[2]))
    y = np.reshape(y, (y.shape[0] * tc.lstmo, y.shape[2]))
    for i in range(x.shape[1]):
        x[:, i] -= r[i][1]
        x[:, i] /= r[i][0]
    for i in range(y.shape[1]):
        y[:, i] -= r[i][3]
        y[:, i] /= r[i][2]
    x = np.reshape(x, (int(x.shape[0] / tc.lstms), tc.lstms, x.shape[1]))
    y = np.reshape(y, (int(y.shape[0] / tc.lstmo), tc.lstmo, y.shape[1]))

    xp = 1
    if tc.feature < 4:
        x = np.concatenate((x[:, :, -n:], y[:, :, -n:]), axis=1)
        y = 1
    elif 4 <= tc.feature < 6:
        x = x[:, :, -n:]
        y = y[:, :, -n:]
    else:
        xp = np.concatenate((x[:, :, -n:], y[:, :, -n:]), axis=1)
        x = x[:, :, :-n]
        y = y[:, :, :-n]

    if tc.feature < 4:
        y, xp = 1, 1
    elif 4 <= tc.feature < 8:
        xp = 1
    # x, y, l for LSTM; realy gives the real position of detection in question
    return x, xp, y, l, pd, realy, realp, truepoints, realx


if __name__ == '__main__':
    scenarios = ['MICROTUBULE', 'VESICLE', 'RECEPTOR', 'VIRUS']
    R = 25
    # track_disp_summary()
    for scenario in scenarios:
        for snr in [1, 2, 4, 7]:
            for dens in ['low', 'mid', 'high']:
                for add in [50, 40, 30]:
                    for rem in [0]:
                        MTJsimulation(scenario, snr, dens, add, rem, R)
