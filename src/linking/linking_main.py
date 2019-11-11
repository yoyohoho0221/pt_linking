# ## linking using MAP methods
#
#

import time
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from linking.convert_XML import writeXML, writeXMLilya
from linking.linking_analysis import readFile, plotCompare_rems, plotCompare_adds
from linking.linking_MAP_RNN_md import linkingMAP
from rnn_md.model_config_RNN_md import test_config
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def plotThrsCompare(sc, snr, den, x, a, b, jp, jt, mx, my):
    fig = plt.figure(100)
    fig.suptitle(sc + ' ' + str(snr) + ' ' + den + ' thresholds comparison')

    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x, a)
    ax.plot(mx[0], my[0], 'o')
    ax.set_ylabel(r'$\alpha$')
    ax.set_xlabel('Thresholds')

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x, b)
    ax.plot(mx[1], my[1], 'o')
    ax.set_ylabel(r'$\beta$')
    ax.set_xlabel('Thresholds')

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x, jp)
    ax.plot(mx[2], my[2], 'o')
    ax.set_ylabel(r'$JSC$')
    ax.set_xlabel('Thresholds')

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x, jt)
    ax.plot(mx[3], my[3], 'o')
    ax.set_ylabel(r'$JSC\gamma$')
    ax.set_xlabel('Thresholds')

    fig.set_size_inches(9.5, 6.5)
    return fig


def deleteAll(method, rem, add, scenario, snr, den, tt, tc, gatexy, gatez, usepred, interp, isbest,
              togit=1):
    if togit == 1:
        pp = '/home/yaoyao/IDEA/dl/'
        ppsave = '/home/yaoyao/IDEA/dl/'
    elif togit == 0:
        pp = '/home/yyao/IDEA/dl/'
        ppsave = '/media/yyao/work/'
    else:
        pp = '/nfs/home4/yao88/IDEA/dl/'
        ppsave = '/home/yyao/IDEA/dl/'

    if scenario == 'VIRUS':
        is3D = True
    else:
        is3D = False

    if not isbest:
        stt = 'Last'
    else:
        stt = 'Best'
    if len(tt) == 1:
        isthrs = False
    else:
        isthrs = True
    R = 25

    saveTo = ppsave + 'results/' + 'S' + str(tc.lstms) + 'H' + str(tc.lstmo) + 'I' + str(
        tc.feature) + '/' \
             + scenario + '/' + scenario + str(snr) + den + str(add) + \
             method + '-CLRE' + str(tc.lstms) + str(tc.lstmo) + str(usepred)[0] + str(interp)[0] + \
             str(
                 isthrs)[0] + str(tc.feature) + stt + '/'

    for thrs in tt:
        if rem != -1:
            if is3D:
                npy = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + '_' + str(gatez) + 'posR' + str(R) + 'add' + str(
                    add) + 'rem' + str(
                    rem) + '.npy'
                xml = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + '_' + str(gatez) + 'posR' + str(R) + 'add' + str(
                    add) + 'rem' + str(
                    rem) + '.xml'
                o = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + '_' + str(gatez) + 'posR' + str(R) + 'add' + str(
                    add) + 'rem' + str(
                    rem) + '.xml.txt'
            else:
                npy = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(
                    rem) + '.npy'
                xml = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(
                    rem) + '.xml'
                o = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(
                    rem) + '.xml.txt'
        else:
            if is3D:
                npy = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + '_' + str(gatez) + 'pos.npy'
                xml = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + '_' + str(gatez) + 'pos.xml'
                o = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + '_' + str(gatez) + 'pos.xml.txt'
            else:
                npy = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + 'pos.npy'
                xml = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + 'pos.xml'
                o = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + 'pos.xml.txt'

        subprocess.call(['rm', npy])
        subprocess.call(['rm', xml])
        subprocess.call(['rm', o])


def readThrsfiles(method, rem, add, scenario, snr, den, tt, tc, gatexy, gatez, usepred, interp,
                  isbest, togit=1):
    if togit == 1:
        pp = '/home/yaoyao/IDEA/dl/'
        ppsave = '/home/yaoyao/IDEA/dl/'
    elif togit == 0:
        pp = '/home/yyao/IDEA/dl/'
        ppsave = '/media/yyao/work/'
    else:
        pp = '/nfs/home4/yao88/IDEA/dl/'
        ppsave = '/home/yyao/IDEA/dl/'

    if scenario == 'VIRUS':
        is3D = True
    else:
        is3D = False

    if not isbest:
        stt = 'Last'
    else:
        stt = 'Best'
    if len(tt) == 1:
        isthrs = False
    else:
        isthrs = True
    R = 25

    saveTo = ppsave + 'results/' + 'S' + str(tc.lstms) + 'H' + str(tc.lstmo) + 'I' + str(
        tc.feature) + '/' \
             + scenario + '/' + scenario + str(snr) + den + str(add) + \
             method + '-CLRE' + str(tc.lstms) + str(tc.lstmo) + str(usepred)[0] + str(interp)[0] + \
             str(
                 isthrs)[0] + str(tc.feature) + stt + '/'

    alpha = []
    beta = []
    jp = []
    jt = []
    onpy = []
    oxml = []
    oo = []
    for thrs in tt:
        if rem != -1:
            if is3D:
                npy = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + '_' + str(gatez) + 'posR' + str(R) + 'add' + str(
                    add) + 'rem' + str(
                    rem) + '.npy'
                xml = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + '_' + str(gatez) + 'posR' + str(R) + 'add' + str(
                    add) + 'rem' + str(
                    rem) + '.xml'
                o = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + '_' + str(gatez) + 'posR' + str(R) + 'add' + str(
                    add) + 'rem' + str(
                    rem) + '.xml.txt'
            else:
                npy = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(
                    rem) + '.npy'
                xml = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(
                    rem) + '.xml'
                o = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(
                    rem) + '.xml.txt'
        else:
            if is3D:
                npy = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + '_' + str(gatez) + 'pos.npy'
                xml = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + '_' + str(gatez) + 'pos.xml'
                o = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + '_' + str(gatez) + 'pos.xml.txt'
            else:
                npy = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + 'pos.npy'
                xml = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + 'pos.xml'
                o = saveTo + scenario + str(snr) + den + 'RNN' + str(thrs) + '_' + str(
                    tc.lstms) + str(
                    tc.lstmo) + str(gatexy) + 'pos.xml.txt'
        if len(npy) > 0:
            onpy.append(npy)
            oxml.append(xml)
            oo.append(o)
            a, b, j_p, j_t = readFile(o)
            alpha.append(a)
            beta.append(b)
            jp.append(j_p)
            jt.append(j_t)
    return alpha, beta, jp, jt, onpy, oxml, oo


def compareThrs(method, rem, add, scenario, snr, den, tt, tc, gatexy, gatez, usepred, interp,
                isbest, togit=1, isplot=True):
    if togit == 1:
        pp = '/home/yaoyao/IDEA/dl/'
        ppsave = '/home/yaoyao/IDEA/dl/'
    elif togit == 0:
        pp = '/home/yyao/IDEA/dl/'
        ppsave = '/media/yyao/work/'
    else:
        pp = '/nfs/home4/yao88/IDEA/dl/'
        ppsave = '/home/yyao/IDEA/dl/'

    if scenario == 'VIRUS':
        is3D = True
    else:
        is3D = False

    if not isbest:
        stt = 'Last'
    else:
        stt = 'Best'

    if len(tt) == 1:
        isthrs = False
    else:
        isthrs = True

    x = []
    y = []

    alpha, beta, jp, jt, onpy, oxml, oouts = readThrsfiles(method, rem, add, scenario, snr, den, tt,
                                                           tc,
                                                           gatexy, gatez, usepred, interp, isbest,
                                                           togit)

    for optimize in ['a', 'b', 'jp', 'jt']:
        R = 25
        if optimize == 'a' or optimize == 'b' or optimize == 'jp' or optimize == 'jt':
            ms = 0
        else:
            ms = 3
        mthrs = 0
        ma = 0
        mb = 0
        mjp = 0
        mjt = 0
        mo = ''
        mn = ''
        mx = ''

        saveTo = ppsave + 'results/' + 'S' + str(tc.lstms) + 'H' + str(tc.lstmo) + 'I' + str(
            tc.feature) + '/' \
                 + scenario + '/' + scenario + str(snr) + den + str(add) + \
                 method + '-CLRE' + str(tc.lstms) + str(tc.lstmo) + str(usepred)[0] + str(interp)[
                     0] + str(
            isthrs)[0] + str(tc.feature) + stt + '/' + optimize + '/'

        # Make a path to be saved in.
        if not os.path.exists(saveTo):
            os.makedirs(saveTo)

        for a, b, j_p, j_t, npy, xml, o, thrs in zip(alpha, beta, jp, jt, onpy, oxml, oouts, tt):
            if optimize == 'a' and a > ms:
                mn = npy
                mx = xml
                mo = o
                mthrs = thrs
                ma = a
                mb = b
                mjp = j_p
                mjt = j_t
                ms = a
            elif optimize == 'b' and b > ms:
                mn = npy
                mx = xml
                mo = o
                mthrs = thrs
                ma = a
                mb = b
                mjp = j_p
                mjt = j_t
                ms = b
            elif optimize == 'jp' and j_p > ms:
                mn = npy
                mx = xml
                mo = o
                mthrs = thrs
                ma = a
                mb = b
                mjp = j_p
                mjt = j_t
                ms = j_p
            elif optimize == 'jt' and j_t > ms:
                mn = npy
                mx = xml
                mo = o
                mthrs = thrs
                ma = a
                mb = b
                mjp = j_p
                mjt = j_t
                ms = j_t
        x.append(mthrs)
        y.append(ms)

        print(scenario + str(snr) + den + '_' + method + str(tc.lstms) + str(tc.lstmo) + '_' + str(
            rem) + '_' + str(add) + ' T: ' + str(mthrs) + ' A: ' + '{:.5f}'.format(
            ma) + ' B: ' + '{:.5f}'.format(mb) + ' Jp: ' + '{:.5f}'.format(
            mjp) + ' Jt: ' + '{:.5f}'.format(mjt))

        if rem != -1:
            if is3D:
                nnpy = saveTo + scenario + str(snr) + den + 'RNN' + '_' + str(tc.lstms) + str(
                    tc.lstmo) + str(
                    gatexy) + '_' + str(gatez) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(
                    rem) + '.npy'
                nxml = saveTo + scenario + str(snr) + den + 'RNN' + '_' + str(tc.lstms) + str(
                    tc.lstmo) + str(
                    gatexy) + '_' + str(gatez) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(
                    rem) + '.xml'
                no = saveTo + scenario + str(snr) + den + 'RNN' + '_' + str(tc.lstms) + str(
                    tc.lstmo) + str(
                    gatexy) + '_' + str(gatez) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(
                    rem) + '.xml.txt'
            else:
                nnpy = saveTo + scenario + str(snr) + den + 'RNN' + '_' + str(tc.lstms) + str(
                    tc.lstmo) + str(
                    gatexy) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(rem) + '.npy'
                nxml = saveTo + scenario + str(snr) + den + 'RNN' + '_' + str(tc.lstms) + str(
                    tc.lstmo) + str(
                    gatexy) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(rem) + '.xml'
                no = saveTo + scenario + str(snr) + den + 'RNN' + '_' + str(tc.lstms) + str(
                    tc.lstmo) + str(
                    gatexy) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(rem) + '.xml.txt'
        else:
            if is3D:
                nnpy = saveTo + scenario + str(snr) + den + 'RNN' + '_' + str(tc.lstms) + str(
                    tc.lstmo) + str(
                    gatexy) + '_' + str(gatez) + 'pos.npy'
                nxml = saveTo + scenario + str(snr) + den + 'RNN' + '_' + str(tc.lstms) + str(
                    tc.lstmo) + str(
                    gatexy) + '_' + str(gatez) + 'pos.xml'
                no = saveTo + scenario + str(snr) + den + 'RNN' + '_' + str(tc.lstms) + str(
                    tc.lstmo) + str(
                    gatexy) + '_' + str(gatez) + 'pos.xml.txt'
            else:
                nnpy = saveTo + scenario + str(snr) + den + 'RNN' + '_' + str(tc.lstms) + str(
                    tc.lstmo) + str(
                    gatexy) + 'pos.npy'
                nxml = saveTo + scenario + str(snr) + den + 'RNN' + '_' + str(tc.lstms) + str(
                    tc.lstmo) + str(
                    gatexy) + 'pos.xml'
                no = saveTo + scenario + str(snr) + den + 'RNN' + '_' + str(tc.lstms) + str(
                    tc.lstmo) + str(
                    gatexy) + 'pos.xml.txt'

        if len(nnpy) > 0:
            shutil.copyfile(mn, nnpy)
            shutil.copyfile(mx, nxml)
            shutil.copyfile(mo, no)
    print('--------------------------------------------------------------------\n')
    if isplot:
        saveTop = ppsave + 'results/' + 'S' + str(tc.lstms) + 'H' + str(tc.lstmo) + 'I' + str(
            tc.feature) + '/' \
                  + scenario + '/' + scenario + str(snr) + den + str(add) + \
                  method + '-CLRE' + str(tc.lstms) + str(tc.lstmo) + str(usepred)[0] + str(interp)[
                      0] + \
                  str(isthrs)[0] + str(tc.feature) + stt + '/'
        pl = plotThrsCompare(scenario, snr, den, tt, alpha, beta, jp, jt, x, y)
        pl.savefig(saveTop + 'Linking_' + method + str(tc.feature) + scenario + str(snr) + den +
                   '_thrscompare' + str(tc.lstms) + str(tc.lstmo) + str(usepred)[0] + str(interp)[
                       0] +
                   str(isthrs)[0] + stt + '_' + str(rem) + '_' + str(add) + '.png')
        pl.clf()

    deleteAll(method, rem, add, scenario, snr, den, tt, tc, gatexy, gatez, usepred, interp, isbest,
              togit)


def run(method, scenario, snr, den, tt, tc, add, imagesize=512, gatexy=30, gatez=5,
        usepred=True, interp=False, isbest=False, isprint=False, togit=1):
    start = time.time()

    if togit == 1:
        pp = '/home/yaoyao/IDEA/dl/'
        ppsave = '/home/yaoyao/IDEA/dl/'
    elif togit == 0:
        pp = '/home/yyao/IDEA/dl/'
        ppsave = '/media/yyao/work/'
    else:
        pp = '/nfs/home4/yao88/IDEA/dl/'
        ppsave = '/home/yyao/IDEA/dl/'

    max_epLength = 100
    R = 25

    if scenario == 'VIRUS':
        is3D = True
    else:
        is3D = False

    if add == 0:
        rr = [-1, 0, 5, 10, 15, 20]
    elif add == 20:
        rr = [0, 5, 10, 15, 20]
    else:
        rr = [0]

    path_m = 'models/rnnMD/' + tc.scenario + '/model_' + tc.fullname

    if not isbest:
        path_m = path_m + '_final'
        stt = 'Last'
    else:
        stt = 'Best'

    if len(tt) == 1:
        isthrs = False
    else:
        isthrs = True

    saveTo = ppsave + 'results/' + 'S' + str(tc.lstms) + 'H' + str(tc.lstmo) + 'I' + str(
        tc.feature) + '/' \
             + scenario + '/' + scenario + str(snr) + den + str(add) + method + \
             '-CLRE' + str(tc.lstms) + str(tc.lstmo) + str(usepred)[0] + str(interp)[0] + \
             str(isthrs)[0] + \
             str(tc.feature) + stt + '/'
    ref = pp + 'data/MTJ_RNN/data/XML/' + scenario + ' snr ' + str(snr) + ' density ' + den + '.xml'

    # Make a path to be saved in.
    if not os.path.exists(saveTo):
        os.makedirs(saveTo)

    for rem in rr:
        for thrs in tt:
            thrslink = 0.01 * thrs
            if len(tt) == 1:
                thrslink = thrs

            if rem != -1:
                print(
                    scenario + str(snr) + den + str(usepred)[0] + str(interp)[0] + str(isthrs)[0] +
                    str(tc.feature) + ' R' + str(R) + ' add' + str(add) + ' rem' + str(
                        rem) + '_' + str(thrs))
                path = pp + 'data/MTJ_RNN/data/' + scenario + ' snr ' + str(
                    snr) + ' density ' + den + ' R ' + str(
                    R) + ' add ' + str(add) + ' rem ' + str(rem) + '.detections.xml.txt'
                if is3D:
                    npy = saveTo + scenario + str(
                        snr) + den + 'RNN' + str(thrs) + '_' + str(
                        tc.lstms) + str(tc.lstmo) + str(gatexy) + '_' + str(
                        gatez) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(
                        rem) + '.npy'
                    can = saveTo + scenario + str(snr) + den + 'RNN' + str(
                        thrs) + '_' + str(tc.lstms) + str(tc.lstmo) + str(
                        gatexy) + '_' + str(gatez) + 'posR' + str(
                        R) + 'add' + str(add) + 'rem' + str(rem) + '.xml'
                    o = saveTo + scenario + str(snr) + den + 'RNN' + str(
                        thrs) + '_' + str(tc.lstms) + str(tc.lstmo) + str(
                        gatexy) + '_' + str(gatez) + 'posR' + str(
                        R) + 'add' + str(add) + 'rem' + str(rem) + '.xml.txt'
                else:
                    npy = saveTo + scenario + str(
                        snr) + den + 'RNN' + str(thrs) + '_' + str(
                        tc.lstms) + str(tc.lstmo) + str(
                        gatexy) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(
                        rem) + '.npy'
                    can = saveTo + scenario + str(snr) + den + 'RNN' + str(
                        thrs) + '_' + str(tc.lstms) + str(tc.lstmo) + str(
                        gatexy) + 'posR' + str(
                        R) + 'add' + str(add) + 'rem' + str(rem) + '.xml'
                    o = saveTo + scenario + str(snr) + den + 'RNN' + str(
                        thrs) + '_' + str(tc.lstms) + str(tc.lstmo) + str(
                        gatexy) + 'posR' + str(
                        R) + 'add' + str(add) + 'rem' + str(rem) + '.xml.txt'
            else:
                print(scenario + str(snr) + den + str(usepred)[0] + str(interp)[0] +
                      str(isthrs)[0] + str(tc.feature) + ' GT_' + str(thrs))
                path = pp + 'data/MTJ_RNN/data/' + scenario + ' snr ' + str(
                    snr) + ' density ' + den + '.detections.xml.txt'
                if is3D:
                    npy = saveTo + scenario + str(snr) + den + 'RNN' + str(
                        thrs) + '_' + str(tc.lstms) + str(tc.lstmo) + str(gatexy) + '_' + str(
                        gatez) + 'pos.npy'
                    can = saveTo + scenario + str(snr) + den + 'RNN' + str(
                        thrs) + '_' + str(tc.lstms) + str(tc.lstmo) + str(gatexy) + '_' + str(
                        gatez) + 'pos.xml'
                    o = saveTo + scenario + str(snr) + den + 'RNN' + str(
                        thrs) + '_' + str(tc.lstms) + str(tc.lstmo) + str(
                        gatexy) + '_' + str(gatez) + 'pos.xml.txt'
                else:
                    npy = saveTo + scenario + str(snr) + den + 'RNN' + str(
                        thrs) + '_' + str(tc.lstms) + str(tc.lstmo) + str(gatexy) + 'pos.npy'
                    can = saveTo + scenario + str(snr) + den + 'RNN' + str(
                        thrs) + '_' + str(tc.lstms) + str(tc.lstmo) + str(gatexy) + 'pos.xml'
                    o = saveTo + scenario + str(snr) + den + 'RNN' + str(
                        thrs) + '_' + str(tc.lstms) + str(tc.lstmo) + str(
                        gatexy) + 'pos.xml.txt'

            if not tc.load_model or not os.path.exists(path_m + '/model.index') or len(path) <= 0:
                print('No model in ' + path_m)
            else:
                linkingMAP(path, npy, path_m, tc, thrslink, usepred, interp, isthrs, gatexy, gatez,
                           max_epLength, imagesize, isprint)
                writeXML(npy, can, scenario, snr, den, method, thrslink, interp)
                if os.path.exists(o):
                    subprocess.call(['rm', o])
                subprocess.call(
                    ['java', '-jar', 'trackingPerformanceEvaluation.jar', '-r', ref, '-c', can,
                     '-o', o])
        compareThrs(method, rem, add, scenario, snr, den, tt, tc, gatexy, gatez, usepred, interp,
                    isbest, togit, isplot=True)
    print('Time : ', time.time() - start)


def runMethods(method, scenarios, dens, ss, oo, ff, acts, rnns, rr, hh, ll, kk, delta, alpha,
               dropout, batch_size, epoch, lr, adds, tt, usepred=True, interp=False, isbest=False,
               isprint=False, togit=1):
    snrs = [1, 2]
    imagesize = 512
    gatexy = 15
    gatez = 3

    for scenario in scenarios:
        for den in dens:
            for add in adds:
                for snr in snrs:
                    for st in ss:
                        for ot in oo:
                            for ft in ff:
                                for act, rnn, r, h, l, k in zip(acts, rnns, rr, hh, ll, kk):
                                    print(
                                        '\n====================================================================\n')
                                    tc = test_config(scenario, st, ot, ft, act, rnn, r, h, l, k,
                                                     delta, alpha, dropout, batch_size, epoch, lr)
                                    stritem = 'Linking ' + method + ' RNN ' + tc.fullname
                                    print(stritem)
                                    run(method, scenario, snr, den, tt, tc, add, imagesize,
                                        gatexy, gatez, usepred, interp, isbest, isprint, togit)


def runPlots(method, scenarios, dens, ss, oo, ff, adds, tt, usepred=True, interp=False,
             isbest=False, togit=1):
    gatexy = 15
    gatez = 3
    rem = 0
    if len(tt) == 1:
        isthrs = False
    else:
        isthrs = True

    for scenario in scenarios:
        for den in dens:
            for add in adds:
                for st in ss:
                    for ot in oo:
                        for ft in ff:
                            if add == 20 or add == 0:
                                plotCompare_rems(scenario, den, add, method, [st], [ot],
                                                 [ft], gatexy, gatez, usepred, interp, isthrs,
                                                 isbest, togit)
                if add == 20 or add == 0:
                    plotCompare_rems(scenario, den, add, method, ss, oo, ff, gatexy, gatez,
                                     usepred, interp, isthrs, isbest, togit)
            for st in ss:
                for ot in oo:
                    for ft in ff:
                        plotCompare_adds(scenario, den, rem, method, [st], [ot], [ft], gatexy,
                                         gatez, usepred, interp,
                                         isthrs, isbest, togit)
            plotCompare_adds(scenario, den, rem, method, ss, oo, ff, gatexy, gatez, usepred, interp,
                             isthrs, isbest, togit)


def runilya(method, typepath, type, no, tt, tc, imagesize=512, gatexy=30, gatez=5,
            usepred=True, interp=False, isbest=False, isprint=False, togit=1):
    print(type +' '+ no)
    if togit == 1:
        pp = '/home/yaoyao/IDEA/dl/'
        ppsave = '/home/yaoyao/IDEA/dl/'
    elif togit == 0:
        pp = '/home/yyao/IDEA/dl/'
        ppsave = '/media/yyao/work/'
    else:
        pp = '/nfs/home4/yao88/IDEA/dl/'
        ppsave = '/home/yyao/IDEA/dl/'

    max_epLength = 500

    is3D = False

    path_m = 'models/rnnMD/' + tc.scenario + '/model_' + tc.fullname

    if not isbest:
        path_m = path_m + '_final'
        stt = 'Last'
    else:
        stt = 'Best'

    if len(tt) == 1:
        isthrs = False
    else:
        isthrs = True
    saveTo = ppsave + 'results_ilya/' + type +'/' + tc.scenario + '/' + no + 'S' + str(tc.lstms) \
             + 'H' + str(tc.lstmo) + 'I' + str(tc.feature) \
             + '/' + method + '-CLRE' + str(tc.lstms) + str(tc.lstmo) + str(usepred)[0] + \
             str(interp)[0] + str(isthrs)[0] + str(tc.feature) + stt + '/'

    ref = pp + 'data/ilya/'+type+'/'+typepath + (no) + '-isbi.xml'
    if os.path.exists(ref):
        # Make a path to be saved in.
        if not os.path.exists(saveTo):
            os.makedirs(saveTo)

        start = time.time()

        for thrs in tt:
            thrslink = 0.01 * thrs
            if len(tt) == 1:
                thrslink = thrs

            print(type + no + str(usepred)[0] + str(interp)[0] + str(isthrs)[0] + str(tc.feature)
                  + ' GT_' + str(thrs))
            path = pp + 'data/ilya/'+type+'/'+typepath + (no) + '-.detections.xml.txt'
            npy = saveTo + 'RNN' + str(
                thrs) + '_' + str(tc.lstms) + str(tc.lstmo) + str(gatexy) + 'pos.npy'
            can = saveTo + 'RNN' + str(
                thrs) + '_' + str(tc.lstms) + str(tc.lstmo) + str(gatexy) + 'pos.xml'
            o = saveTo + 'RNN' + str(
                thrs) + '_' + str(tc.lstms) + str(tc.lstmo) + str(
                gatexy) + 'pos.xml.txt'

            if not tc.load_model or not os.path.exists(path_m + '/model.index') or len(path) <= 0:
                print('No model in ' + path_m)
            else:
                linkingMAP(path, npy, path_m, tc, thrslink, usepred, interp, isthrs, gatexy, gatez,
                           max_epLength, imagesize, isprint)
                writeXMLilya(npy, can, no, method, thrslink, interp, 'npy')
                if os.path.exists(o):
                    subprocess.call(['rm', o])
                subprocess.call(
                    ['java', '-jar', 'trackingPerformanceEvaluation.jar', '-r', ref, '-c', can,
                     '-o', o])
        compareThrsilya(method, type, no, tt, tc, gatexy, gatez, usepred, interp, isbest, togit,
                        isplot=True)
        print('Time : ', time.time() - start)
    else:
        print('No SOS tracking!!!')


def deleteAllilya(method, type, no, tt, tc, gatexy, gatez, usepred, interp, isbest, togit=1):
    if togit == 1:
        pp = '/home/yaoyao/IDEA/dl/'
        ppsave = '/home/yaoyao/IDEA/dl/'
    elif togit == 0:
        pp = '/home/yyao/IDEA/dl/'
        ppsave = '/media/yyao/work/'
    else:
        pp = '/nfs/home4/yao88/IDEA/dl/'
        ppsave = '/home/yyao/IDEA/dl/'

    is3D = False

    if not isbest:
        stt = 'Last'
    else:
        stt = 'Best'
    if len(tt) == 1:
        isthrs = False
    else:
        isthrs = True
    saveTo = ppsave + 'results_ilya/' + type +'/' + tc.scenario + '/' + no + 'S' + str(tc.lstms) \
             + 'H' + str(tc.lstmo) + 'I' + str(tc.feature) \
             + '/' + method + '-CLRE' + str(tc.lstms) + str(tc.lstmo) + str(usepred)[0] + \
             str(interp)[0] + str(isthrs)[0] + str(tc.feature) + stt + '/'

    for thrs in tt:
        npy = saveTo + 'RNN' + str(thrs) + '_' + str(tc.lstms) + str(
            tc.lstmo) + str(gatexy) + 'pos.npy'
        xml = saveTo + 'RNN' + str(thrs) + '_' + str(tc.lstms) + str(
            tc.lstmo) + str(gatexy) + 'pos.xml'
        o = saveTo + 'RNN' + str(thrs) + '_' + str(tc.lstms) + str(
            tc.lstmo) + str(gatexy) + 'pos.xml.txt'

        subprocess.call(['rm', npy])
        subprocess.call(['rm', xml])
        subprocess.call(['rm', o])


def readThrsfilesilya(method, type, no, tt, tc, gatexy, gatez, usepred, interp, isbest, togit=1):
    if togit == 1:
        pp = '/home/yaoyao/IDEA/dl/'
        ppsave = '/home/yaoyao/IDEA/dl/'
    elif togit == 0:
        pp = '/home/yyao/IDEA/dl/'
        ppsave = '/media/yyao/work/'
    else:
        pp = '/nfs/home4/yao88/IDEA/dl/'
        ppsave = '/home/yyao/IDEA/dl/'

    is3D = False

    if not isbest:
        stt = 'Last'
    else:
        stt = 'Best'
    if len(tt) == 1:
        isthrs = False
    else:
        isthrs = True
    saveTo = ppsave + 'results_ilya/' + type +'/' + tc.scenario + '/' + no + 'S' + str(tc.lstms) \
             + 'H' + str(tc.lstmo) + 'I' + str(tc.feature) \
             + '/' + method + '-CLRE' + str(tc.lstms) + str(tc.lstmo) + str(usepred)[0] + \
             str(interp)[0] + str(isthrs)[0] + str(tc.feature) + stt + '/'

    alpha = []
    beta = []
    jp = []
    jt = []
    onpy = []
    oxml = []
    oo = []
    for thrs in tt:
        npy = saveTo + 'RNN' + str(thrs) + '_' + str(tc.lstms) + str(
            tc.lstmo) + str(gatexy) + 'pos.npy'
        xml = saveTo + 'RNN' + str(thrs) + '_' + str(tc.lstms) + str(
            tc.lstmo) + str(gatexy) + 'pos.xml'
        o = saveTo + 'RNN' + str(thrs) + '_' + str(tc.lstms) + str(
            tc.lstmo) + str(gatexy) + 'pos.xml.txt'
        if len(npy) > 0:
            onpy.append(npy)
            oxml.append(xml)
            oo.append(o)
            a, b, j_p, j_t = readFile(o)
            alpha.append(a)
            beta.append(b)
            jp.append(j_p)
            jt.append(j_t)
    return alpha, beta, jp, jt, onpy, oxml, oo


def compareThrsilya(method, type, no, tt, tc, gatexy, gatez, usepred, interp, isbest, togit=1,
                    isplot=True):
    if togit == 1:
        pp = '/home/yaoyao/IDEA/dl/'
        ppsave = '/home/yaoyao/IDEA/dl/'
    elif togit == 0:
        pp = '/home/yyao/IDEA/dl/'
        ppsave = '/media/yyao/work/'
    else:
        pp = '/nfs/home4/yao88/IDEA/dl/'
        ppsave = '/home/yyao/IDEA/dl/'

    is3D = False

    if not isbest:
        stt = 'Last'
    else:
        stt = 'Best'

    if len(tt) == 1:
        isthrs = False
    else:
        isthrs = True

    x = []
    y = []

    alpha, beta, jp, jt, onpy, oxml, oouts = readThrsfilesilya(method, type, no, tt, tc,
                                                               gatexy, gatez, usepred, interp,
                                                               isbest, togit)

    for optimize in ['a', 'b', 'jp', 'jt']:
        R = 25
        if optimize == 'a' or optimize == 'b' or optimize == 'jp' or optimize == 'jt':
            ms = 0
        else:
            ms = 3
        mthrs = 0
        ma = 0
        mb = 0
        mjp = 0
        mjt = 0
        mo = ''
        mn = ''
        mx = ''
        saveTo = ppsave + 'results_ilya/' + type +'/' + tc.scenario + '/' + no + 'S' + str(
            tc.lstms) + 'H' + str(tc.lstmo) + 'I' + str(tc.feature) \
                 + '/' + method + '-CLRE' + str(tc.lstms) + str(tc.lstmo) + str(usepred)[0] + \
                 str(interp)[0] + str(isthrs)[0] + str(tc.feature) + stt + '/' + optimize + '/'

        # Make a path to be saved in.
        if not os.path.exists(saveTo):
            os.makedirs(saveTo)

        for a, b, j_p, j_t, npy, xml, o, thrs in zip(alpha, beta, jp, jt, onpy, oxml, oouts, tt):
            if optimize == 'a' and a > ms:
                mn = npy
                mx = xml
                mo = o
                mthrs = thrs
                ma = a
                mb = b
                mjp = j_p
                mjt = j_t
                ms = a
            elif optimize == 'b' and b > ms:
                mn = npy
                mx = xml
                mo = o
                mthrs = thrs
                ma = a
                mb = b
                mjp = j_p
                mjt = j_t
                ms = b
            elif optimize == 'jp' and j_p > ms:
                mn = npy
                mx = xml
                mo = o
                mthrs = thrs
                ma = a
                mb = b
                mjp = j_p
                mjt = j_t
                ms = j_p
            elif optimize == 'jt' and j_t > ms:
                mn = npy
                mx = xml
                mo = o
                mthrs = thrs
                ma = a
                mb = b
                mjp = j_p
                mjt = j_t
                ms = j_t
        x.append(mthrs)
        y.append(ms)

        print(method + str(tc.lstms) + str(tc.lstmo) + '_' + ' T: ' + str(
            mthrs) + ' A: ' + '{:.5f}'.format(
            ma) + ' B: ' + '{:.5f}'.format(mb) + ' Jp: ' + '{:.5f}'.format(
            mjp) + ' Jt: ' + '{:.5f}'.format(mjt))

        nnpy = saveTo + 'RNN' + '_' + str(tc.lstms) + str(tc.lstmo) + str(
            gatexy) + 'pos.npy'
        nxml = saveTo + 'RNN' + '_' + str(tc.lstms) + str(tc.lstmo) + str(
            gatexy) + 'pos.xml'
        nno = saveTo + 'RNN' + '_' + str(tc.lstms) + str(tc.lstmo) + str(
            gatexy) + 'pos.xml.txt'

        if len(nnpy) > 0:
            shutil.copyfile(mn, nnpy)
            shutil.copyfile(mx, nxml)
            shutil.copyfile(mo, nno)
    print('--------------------------------------------------------------------\n')

    deleteAllilya(method, type, no, tt, tc, gatexy, gatez, usepred, interp, isbest, togit)


if __name__ == '__main__':

    ss = ['RECEPTOR']#, 'VESICLE', 'VIRUS']#['RECEPTOR', 'MICROTUBULE', 'VESICLE', 'VIRUS']
    lss = [3]
    loo = [1]
    ff = [10]
    t = 1

    for s in ss:
        for ls in lss:
            for lo in loo:
                for f in ff:
                    tc = test_config(scenario=s, lstms=ls, lstmo=lo, i=f, act='tanh', rnn='lstm',
                                     R=1, H=128, L=256, k=2048, delta=0.5, alpha=0.1, dropout=0.5,
                                     bs=80000, epoch=800, lr=1e-3)

                    if t == 1:
                        tt = [0.15, 0.2, 0.25, 0.3, 0.35]#[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                    else:
                        tt = [-1]

                    for type in ['rab11','eb3']:# ['rab5', 'rab11', 'eb3']:#['rab5', 'rab6', 'rab11', 'eb3']:
                        if type == 'rab5':
                            for no in ['01', '02', '03', '04']:
                                typepath = '20180712 HeLa control mCherry-Rab5 ILAS2 x100 100ms -'
                                runilya('MAP', typepath, type, no, tt, tc, imagesize=512, gatexy=10,
                                        gatez=5, usepred=True, interp=True, isbest=False,
                                        isprint=False, togit=0)
                        elif type == 'rab6':
                            for no in ['04', '05', '08', '10']:
                                typepath = '20180831 HeLa Control Rab6-mCherry ILAS2 x100 100ms -'
                                #runilya('MAP', typepath, type, no, tt, tc, imagesize=512, gatexy=10,
                                #        gatez=5, usepred=True, interp=True, isbest=False,
                                #        isprint=False, togit=0)
                        elif type == 'rab11':
                            for no in ['01', '02', '03']:#, '04', '05', '06']:
                                typepath = '20180509 Rab11-GFP control ILAS2 x100 100ms -'
                                runilya('MAP', typepath, type, no, tt, tc, imagesize=512, gatexy=10,
                                        gatez=5, usepred=True, interp=True, isbest=False,
                                        isprint=False, togit=0)
                        else:
                            for no in ['02', '03', '04', '05']:
                                typepath = '20151021 ILAS2 HeLa EB3-GFP control x100 500ms -'
                                runilya('MAP', typepath, type, no, tt, tc, imagesize=512, gatexy=10,
                                        gatez=5, usepred=True, interp=True, isbest=False,
                                        isprint=False, togit=0)