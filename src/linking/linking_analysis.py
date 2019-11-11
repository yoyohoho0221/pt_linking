## plot linking results
#
#

import csv
import matplotlib.pyplot as plt
import numpy as np
import os


# visualize performance fix FP
def plotPerfFN(sc, dens, add, x, a, b, j_p, j_t, ticks, marker='.-', color='b'):
    fig = plt.figure(300)
    fig.suptitle(sc + ' ' + dens + ' ' + str(add) + '% of false-positives')
    # fig.tight_layout()

    ax = plt.subplot(4, 1, 1)
    line, = ax.plot(x, a, marker=marker, color=color)
    ax.set_ylabel(r'$\alpha$')
    # ax.set_xlabel('% of false-negatives')
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.grid(b=True, which='major', color='grey', linestyle='--', alpha=0.5)
    # ax.tick_params(right=True, labelright=True)

    ax = plt.subplot(4, 1, 2)
    ax.plot(x, b, marker=marker, color=color)
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.set_ylabel(r'$\beta$')
    # ax.set_xlabel('% of false-negatives')
    ax.grid(b=True, which='major', color='grey', linestyle='--', alpha=0.5)
    # ax.tick_params(right=True, labelright=True)

    ax = plt.subplot(4, 1, 3)
    ax.plot(x, j_p, marker=marker, color=color)
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.set_ylabel(r'$JSC$')
    # ax.set_xlabel('% of false-negatives')
    ax.grid(b=True, which='major', color='grey', linestyle='--', alpha=0.5)
    # ax.tick_params(right=True, labelright=True)

    ax = plt.subplot(4, 1, 4)
    ax.plot(x, j_t, marker=marker, color=color)
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.set_ylabel(r'$JSC\gamma$')
    ax.set_xlabel('% of false-negatives')
    ax.grid(b=True, which='major', color='grey', linestyle='--', alpha=0.5)
    # ax.tick_params(right=True, labelright=True)
    # plt.show()
    # fig.set_size_inches(13.5, 10.5)
    return line, fig


# visualize performance fix FN
def plotPerfFP(sc, dens, rem, x, a, b, j_p, j_t, ticks, marker='.-', color='b'):
    fig = plt.figure(200)
    fig.suptitle(sc + ' ' + dens + ' ' + str(rem) + '% of false-negatives')

    ax = plt.subplot(4, 1, 1)
    line, = ax.plot(x, a, marker=marker, color=color)
    ax.set_ylabel(r'$\alpha$')
    # ax.set_xlabel('% of false-positives')
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.grid(b=True, which='major', color='grey', linestyle='--', alpha=0.5)
    # ax.tick_params(right=True, labelright=True)

    ax = plt.subplot(4, 1, 2)
    ax.plot(x, b, marker=marker, color=color)
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.set_ylabel(r'$\beta$')
    # ax.set_xlabel('% of false-positives')
    ax.grid(b=True, which='major', color='grey', linestyle='--', alpha=0.5)
    # ax.tick_params(right=True, labelright=True)

    ax = plt.subplot(4, 1, 3)
    ax.plot(x, j_p, marker=marker, color=color)
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.set_ylabel(r'$JSC$')
    # ax.set_xlabel('% of false-positives')
    ax.grid(b=True, which='major', color='grey', linestyle='--', alpha=0.5)
    # ax.tick_params(right=True, labelright=True)

    ax = plt.subplot(4, 1, 4)
    ax.plot(x, j_t, marker=marker, color=color)
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.set_ylabel(r'$JSC\gamma$')
    ax.set_xlabel('% of false-positives')
    ax.grid(b=True, which='major', color='grey', linestyle='--', alpha=0.5)
    # ax.tick_params(right=True, labelright=True)
    # plt.show()
    return line, fig


def readIhorFiles(scenario, dens, R=0, add=0, rem=0, togit=0):
    # read it
    if togit == 1:
        path = '/home/yaoyao/IDEA/dl/data/ihor/LP_Results/' + scenario + '/'
    elif togit == 0:
        path = '/home/yyao/IDEA/dl/data/ihor/LP_Results/' + scenario + '/'
    else:
        path = '/nfs/home4/yao88/IDEA/dl/data/ihor/LP_Results/' + scenario + '/'

    alpha = []
    beta = []
    jaccard_t = []
    jaccard_p = []
    if R == 0:
        sr = '00'
    else:
        sr = str(R)
    if add == 0 or add == 5:
        sadd = '0' + str(add)
    else:
        sadd = str(add)
    if rem == 0 or rem == 5:
        srem = '0' + str(rem)
    else:
        srem = str(rem)

    filename = path + 'res_' + sr + '_' + sadd + '_' + srem + '.txt'

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        table = [[e for e in r] for r in reader]
    if dens == 'low':
        t = 1
        for i in range(1, len(table[t])):
            alpha.append(float(table[t][i]))
        t = 3
        for i in range(1, len(table[t])):
            beta.append(float(table[t][i]))
        t = 5
        for i in range(1, len(table[t])):
            jaccard_t.append(float(table[t][i]))
        t = 7
        for i in range(1, len(table[t])):
            jaccard_p.append(float(table[t][i]))
    elif dens == 'mid':
        t = 2
        for i in range(1, len(table[t])):
            alpha.append(float(table[t][i]))
        t = 4
        for i in range(1, len(table[t])):
            beta.append(float(table[t][i]))
        t = 6
        for i in range(1, len(table[t])):
            jaccard_t.append(float(table[t][i]))
        t = 8
        for i in range(1, len(table[t])):
            jaccard_p.append(float(table[t][i]))
    methods = []
    for i in range(1, len(table[0])):
        methods.append(table[0][i])
    m = ['GNN-D', 'GNN-V', 'NGA-2D-CC', 'NGA-4D-CC', '4LP', 'MAP-4D-CC', 'IMM-2D', 'LAP',
         'MAP-4D-IMM', 'MAP-3D-IMM',
         'MAP-3D-CC', 'NGA-3D-CC', '12', '13']
    return alpha, beta, jaccard_p, jaccard_t, m


def readFile(file):
    with open(file) as f:
        lines = f.readlines()

    test = lines[1]
    ind = test.find("\t")
    score1 = float(test[:ind])
    test = lines[2]
    ind = test.find("\t")
    score2 = float(test[:ind])
    test = lines[5]
    ind = test.find("\t")
    score4 = float(test[:ind])
    test = lines[11]
    ind = test.find("\t")
    score3 = float(test[:ind])
    return score1, score2, score3, score4


def readresultFiles_rems(add, method, feature, alpha, beta, jaccard_p, jaccard_t, m, s, o, usepred,
                         interp, isthrs,
                         scenario, snrs, dens, gatexy, gatez, isbest, togit=1):
    if togit == 1:
        pp = '/home/yaoyao/IDEA/dl/'
        ppsave = '/home/yaoyao/IDEA/dl/'
    elif togit == 0:
        pp = '/home/yyao/IDEA/dl/'
        ppsave = '/media/yyao/work/'
    else:
        pp = '/nfs/home4/yao88/IDEA/dl/'
        ppsave = '/home/yyao/IDEA/dl/'

    R = 25
    rems = [-1, 0, 5, 10, 15, 20]

    if not isbest:
        stt = 'Last'
    else:
        stt = 'Best'
    if scenario == 'VIRUS':
        is3D = True
    else:
        is3D = False

    name = 'RNN' + '-I' + str(feature) + '-S' + str(s) + '-H' + str(o) + str(usepred)[0] + \
           str(interp)[0] + \
           str(isthrs)[0] + stt[0]
    m.append(name)
    i = 0
    for rem in rems:
        al = []
        be = []
        jp = []
        jt = []
        for snr in snrs:
            for optimize in ['a', 'b', 'jp', 'jt']:
                if rem != -1:
                    path = ppsave + 'results/' + 'S' + str(s) + 'H' + str(o) + 'I' + str(
                        feature) + '/' \
                           + scenario + '/' + scenario + str(snr) + dens + str(
                        add) + method + '-CLRE' + str(
                        s) + str(o) + str(usepred)[0] + str(interp)[0] + str(isthrs)[0] + str(
                        feature) + stt + '/' + optimize + '/'
                    if is3D:
                        no = path + scenario + str(snr) + dens + 'RNN' + '_' + str(s) + str(
                            o) + str(
                            gatexy) + '_' + str(gatez) + 'posR' + str(R) + 'add' + str(
                            add) + 'rem' + str(
                            rem) + '.xml.txt'
                    else:
                        no = path + scenario + str(snr) + dens + 'RNN' + '_' + str(s) + str(
                            o) + str(
                            gatexy) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(
                            rem) + '.xml.txt'
                else:
                    path = ppsave + 'results/' + 'S' + str(s) + 'H' + str(o) + 'I' + str(
                        feature) + '/' \
                           + scenario + '/' + scenario + str(snr) + dens + str(
                        0) + method + '-CLRE' + str(
                        s) + str(o) + str(usepred)[0] + str(interp)[0] + str(isthrs)[0] + str(
                        feature) + stt + '/' + optimize + '/'
                    if is3D:
                        no = path + scenario + str(snr) + dens + 'RNN' + '_' + str(s) + str(
                            o) + str(
                            gatexy) + '_' + str(gatez) + 'pos.xml.txt'
                    else:
                        no = path + scenario + str(snr) + dens + 'RNN' + '_' + str(s) + str(
                            o) + str(
                            gatexy) + 'pos.xml.txt'

                a, b, j_p, j_t = readFile(no)
                if optimize == 'a':
                    al.append(a)
                elif optimize == 'b':
                    be.append(b)
                elif optimize == 'jp':
                    jp.append(j_p)
                elif optimize == 'jt':
                    jt.append(j_t)

        al = np.asanyarray(al)
        be = np.asanyarray(be)
        jp = np.asanyarray(jp)
        jt = np.asanyarray(jt)
        al = np.mean(al)
        be = np.mean(be)
        jp = np.mean(jp)
        jt = np.mean(jt)
        alpha[i].append(al)
        beta[i].append(be)
        jaccard_p[i].append(jp)
        jaccard_t[i].append(jt)
        i += 1
    m = np.asanyarray(m)
    return alpha, beta, jaccard_p, jaccard_t, m


def plotCompare_rems(scenario, den, add, method, ss, oo, ff, gatexy, gatez, usepred, interp, isthrs,
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

    rems = [0, 5, 10, 15, 20]
    alpha = []
    beta = []
    jaccard_p = []
    jaccard_t = []
    a, b, j_p, j_t, m = readIhorFiles(scenario, den)  # add GT case as base
    alpha.append(a)
    beta.append(b)
    jaccard_p.append(j_p)
    jaccard_t.append(j_t)
    for rem in rems:
        a, b, j_p, j_t, _ = readIhorFiles(scenario, den, 25, add, rem, togit)
        alpha.append(a)
        beta.append(b)
        jaccard_p.append(j_p)
        jaccard_t.append(j_t)
    choose = [0, 1, 7, 6, 2, 11, 3, 9, 8, 10, 5]

    snrs = [1, 2]

    for s in ss:
        for o in oo:
            for feature in ff:
                alpha, beta, jaccard_p, jaccard_t, m = readresultFiles_rems(add, method, feature,
                                                                            alpha, beta,
                                                                            jaccard_p, jaccard_t, m,
                                                                            s, o, usepred, interp,
                                                                            isthrs, scenario,
                                                                            snrs, den, gatexy,
                                                                            gatez, isbest, togit)
                choose.append(len(alpha[0]) - 1)

    # all
    alpha = np.asanyarray(alpha)
    beta = np.asanyarray(beta)
    jaccard_p = np.asanyarray(jaccard_p)
    jaccard_t = np.asanyarray(jaccard_t)
    choose = np.asanyarray(choose)
    m = np.asanyarray(m)

    ls = []
    ll = []
    xx = [1, 2, 3, 4, 5, 6]
    xxticks = ['GT', '0', '5', '10', '15', '20']

    markers = np.asanyarray(['v', 'v', '>', '>', '', 'x', '+', 'X', 's', 's',
                             'x', '>', '', '', 'o', '*', '*', 'P', 'D', 'd', '+', 'x', '<', '^',
                             's', 'v', 'o'])

    colors = np.asanyarray(
        ['steelblue', 'royalblue', 'c', 'darkcyan', '', 'm', 'grey', 'g', 'y', 'olive',
         'plum', 'cyan', '', '', 'r', 'coral', 'brown', 'pink', 'orange', 'crimson',
         'darksalmon', 'rosybrown', 'lightCoral', 'salmon', 'lightpink'])

    if not isbest:
        stt = 'LastM'
    else:
        stt = 'BestM'

    fig = []
    for i in choose:
        l, fig = plotPerfFN(scenario, den, add, xx, alpha[:, i], beta[:, i], jaccard_p[:, i],
                            jaccard_t[:, i],
                            xxticks, markers[i], colors[i])
        ls.append(l)
        ll.append(m[i])
    fig.legend(ls, ll, loc=8, ncol=3)
    # fig.tight_layout()
    fig.set_size_inches(5, 12)
    if len(ss) == 1 and len(oo) == 1 and len(ff) == 1:
        saveTo = ppsave + 'results_linking/' + 'S' + str(ss[0]) + 'H' + str(oo[0]) + 'I' + str(
            ff[0]) + '/' \
                 + scenario + '/' + scenario + den + method + '_FP' + str(add) + \
                 '-S' + str(ss[0]) + '-H' + str(oo[0]) + '-I' + str(ff[0])
        if not os.path.exists(saveTo):
            os.makedirs(saveTo)
        fig.savefig(saveTo + '/Linking_' + scenario + den + method + '_FP' + str(add) +
                    '-S' + str(ss[0]) + '-H' + str(oo[0]) + '-I' + str(ff[0]) +
                    str(usepred)[0] + str(interp)[0] + str(isthrs)[0] + stt + '.eps', format='eps',
                    dpi=300)
    else:
        saveTo = ppsave + 'results_linking/' + scenario + '/' + scenario + den + method + '_FP' + str(
            add)
        if not os.path.exists(saveTo):
            os.makedirs(saveTo)
        fig.savefig(saveTo + '/Linking_' + scenario + den + method + '_FP' + str(add) +
                    str(usepred)[0] + str(interp)[0] + str(isthrs)[0] + stt + '.eps', format='eps',
                    dpi=300)
    plt.close(fig)


def readresultFiles_adds(rem, method, feature, alpha, beta, jaccard_p, jaccard_t, m, s, o, usepred,
                         interp, isthrs,
                         scenario, snrs, dens, gatexy, gatez, isbest, togit=1):
    if togit == 1:
        pp = '/home/yaoyao/IDEA/dl/'
        ppsave = '/home/yaoyao/IDEA/dl/'
    elif togit == 0:
        pp = '/home/yyao/IDEA/dl/'
        ppsave = '/media/yyao/work/'
    else:
        pp = '/nfs/home4/yao88/IDEA/dl/'
        ppsave = '/home/yyao/IDEA/dl/'

    R = 25
    adds = [-1, 0, 10, 20, 30, 40, 50]

    if not isbest:
        stt = 'Last'
    else:
        stt = 'Best'
    if scenario == 'VIRUS':
        is3D = True
    else:
        is3D = False

    name = 'RNN' + '-I' + str(feature) + '-S' + str(s) + '-H' + str(o) + str(usepred)[0] + \
           str(interp)[0] + \
           str(isthrs)[0] + stt[0]
    m.append(name)
    i = 0
    for add in adds:
        al = []
        be = []
        jp = []
        jt = []
        for snr in snrs:
            for optimize in ['a', 'b', 'jp', 'jt']:
                if add == -1:
                    rem2 = -1
                    path = ppsave + 'results/' + 'S' + str(s) + 'H' + str(o) + 'I' + str(
                        feature) + '/' \
                           + scenario + '/' + scenario + str(snr) + dens + str(0) + \
                           method + '-CLRE' + str(s) + str(o) + str(usepred)[0] + str(interp)[0] + \
                           str(
                               isthrs)[0] + str(feature) + stt + '/' + optimize + '/'
                else:
                    rem2 = rem
                    path = ppsave + 'results/' + 'S' + str(s) + 'H' + str(o) + 'I' + str(
                        feature) + '/' \
                           + scenario + '/' + scenario + str(snr) + dens + str(add) + \
                           method + '-CLRE' + str(s) + str(o) + str(usepred)[0] + str(interp)[0] + \
                           str(
                               isthrs)[0] + str(feature) + stt + '/' + optimize + '/'
                if rem2 != -1:
                    if is3D:
                        no = path + scenario + str(snr) + dens + 'RNN' + '_' + str(s) + str(
                            o) + str(
                            gatexy) + '_' + str(gatez) + 'posR' + str(R) + 'add' + str(
                            add) + 'rem' + str(
                            rem2) + '.xml.txt'
                    else:
                        no = path + scenario + str(snr) + dens + 'RNN' + '_' + str(s) + str(
                            o) + str(
                            gatexy) + 'posR' + str(R) + 'add' + str(add) + 'rem' + str(
                            rem2) + '.xml.txt'
                else:
                    if is3D:
                        no = path + scenario + str(snr) + dens + 'RNN' + '_' + str(s) + str(
                            o) + str(
                            gatexy) + '_' + str(gatez) + 'pos.xml.txt'
                    else:
                        no = path + scenario + str(snr) + dens + 'RNN' + '_' + str(s) + str(
                            o) + str(
                            gatexy) + 'pos.xml.txt'

                a, b, j_p, j_t = readFile(no)
                if optimize == 'a':
                    al.append(a)
                elif optimize == 'b':
                    be.append(b)
                elif optimize == 'jp':
                    jp.append(j_p)
                elif optimize == 'jt':
                    jt.append(j_t)

        al = np.asanyarray(al)
        be = np.asanyarray(be)
        jp = np.asanyarray(jp)
        jt = np.asanyarray(jt)
        al = np.mean(al)
        be = np.mean(be)
        jp = np.mean(jp)
        jt = np.mean(jt)
        alpha[i].append(al)
        beta[i].append(be)
        jaccard_p[i].append(jp)
        jaccard_t[i].append(jt)
        i += 1
    m = np.asanyarray(m)
    return alpha, beta, jaccard_p, jaccard_t, m


def plotCompare_adds(scenario, den, rem, method, ss, oo, ff, gatexy, gatez, usepred, interp, isthrs,
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

    adds = [0, 10, 20, 30, 40, 50]
    alpha = []
    beta = []
    jaccard_p = []
    jaccard_t = []
    a, b, j_p, j_t, m = readIhorFiles(scenario, den)  # add GT case as base
    alpha.append(a)
    beta.append(b)
    jaccard_p.append(j_p)
    jaccard_t.append(j_t)
    for add in adds:
        a, b, j_p, j_t, m = readIhorFiles(scenario, den, 25, add, rem, togit)
        alpha.append(a)
        beta.append(b)
        jaccard_p.append(j_p)
        jaccard_t.append(j_t)
    choose = [0, 1, 7, 6, 2, 11, 3, 9, 8, 10, 5]

    snrs = [1, 2]

    for s in ss:
        for o in oo:
            for feature in ff:
                alpha, beta, jaccard_p, jaccard_t, m = readresultFiles_adds(rem, method, feature,
                                                                            alpha, beta,
                                                                            jaccard_p, jaccard_t, m,
                                                                            s, o, usepred, interp,
                                                                            isthrs, scenario,
                                                                            snrs, den, gatexy,
                                                                            gatez, isbest, togit)
                choose.append(len(alpha[0]) - 1)

    # all
    alpha = np.asanyarray(alpha)
    beta = np.asanyarray(beta)
    jaccard_p = np.asanyarray(jaccard_p)
    jaccard_t = np.asanyarray(jaccard_t)
    choose = np.asanyarray(choose)
    m = np.asanyarray(m)

    ls = []
    ll = []
    xx = [1, 2, 3, 4, 5, 6, 7]
    xxticks = ['GT', '0', '10', '20', '30', '40', '50']

    markers = np.asanyarray(['v', 'v', '>', '>', '', 'x', '+', 'X', 's', 's',
                             'x', '>', '', '', 'o', '*', '*', 'P', 'D', 'd', '+', 'x', '<', '^',
                             's', 'v', 'o'])

    colors = np.asanyarray(
        ['steelblue', 'royalblue', 'c', 'darkcyan', '', 'm', 'grey', 'g', 'y', 'olive',
         'plum', 'cyan', '', '', 'r', 'coral', 'brown', 'pink', 'orange', 'crimson',
         'darksalmon', 'rosybrown', 'lightCoral', 'salmon', 'lightpink'])

    if not isbest:
        stt = 'LastM'
    else:
        stt = 'BestM'

    fig = []
    for i in choose:
        l, fig = plotPerfFP(scenario, den, rem, xx, alpha[:, i], beta[:, i], jaccard_p[:, i],
                            jaccard_t[:, i],
                            xxticks, markers[i], colors[i])
        ls.append(l)
        ll.append(m[i])
    fig.legend(ls, ll, loc=8, ncol=3)
    # fig.tight_layout()
    fig.set_size_inches(5, 12)
    if len(ss) == 1 and len(oo) == 1 and len(ff) == 1:
        saveTo = ppsave + 'results_linking/' + 'S' + str(ss[0]) + 'H' + str(oo[0]) + 'I' + str(
            ff[0]) + '/' \
                 + scenario + '/' + scenario + den + method + '_FN' + str(rem) + \
                 '-S' + str(ss[0]) + '-H' + str(oo[0]) + '-I' + str(ff[0])
        if not os.path.exists(saveTo):
            os.makedirs(saveTo)
        fig.savefig(saveTo + '/Linking_' + scenario + den + method + '_FN' + str(rem) +
                    '-S' + str(ss[0]) + '-H' + str(oo[0]) + '-I' + str(ff[0]) +
                    str(usepred)[0] + str(interp)[0] + str(isthrs)[0] + stt + '.eps', format='eps',
                    dpi=300)
    else:
        saveTo = ppsave + 'results_linking/' + scenario + '/' + scenario + den + method + '_FN' + str(
            rem)
        if not os.path.exists(saveTo):
            os.makedirs(saveTo)
        fig.savefig(saveTo + '/Linking_' + scenario + den + method + '_FN' + str(rem) +
                    str(usepred)[0] + str(interp)[0] + str(isthrs)[0] + stt + '.eps', format='eps',
                    dpi=300)
    plt.close(fig)
