# ## metric score - MAP linking
# using cost scores from RNN
#
import time
import re
import numpy as np
import math
import tensorflow as tf
import gurobipy as grb
from rnn_md.data_RNN_md import getData, getSequence, getDets, trackI
from rnn_md.model_RNN_md import MotionRNN, MotionCRNN, MotionRRNN, MotionRHRNN, MotionRHCNN
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def grbsolver(costs, poslist, tdsno, dets, lstmo, nextt, is3D, isprint=False):
    # Create a new model
    model = grb.Model("mip1")
    model.Params.outputflag = 0

    # Create variables
    ro = []
    for i in range(len(costs)):
        if lstmo == 1:
            ro.append(model.addVar(0.0, 1.0, 0.0, grb.GRB.BINARY,
                                   'z_' + str(costs[i][1]) + '_' + str(costs[i][2])))
        elif lstmo == 2:
            ro.append(
                model.addVar(0.0, 1.0, 0.0, grb.GRB.BINARY, 'z_' + str(costs[i][1]) + '_' + str(
                    costs[i][2]) + '_' + str(costs[i][3])))
        elif lstmo == 3:
            ro.append(
                model.addVar(0.0, 1.0, 0.0, grb.GRB.BINARY, 'z_' + str(costs[i][1]) + '_' + str(
                    costs[i][2]) + '_' + str(costs[i][3]) + '_' + str(costs[i][4])))
    model.update()

    # Set objective
    expr = grb.LinExpr()
    for i in range(len(ro)):
        expr.add(ro[i], costs[i][0])
    model.setObjective(expr, grb.GRB.MINIMIZE)

    # Add constraint
    nrConstraint = 0
    exprcs = []
    for i in range(1, lstmo + 2):
        for j in range(tdsno[i - 1] - 1):
            exprc = grb.LinExpr()
            flag = False
            for cc in range(len(costs)):
                if costs[cc][i] == j:
                    exprc.add(ro[cc], 1.0)
                    flag = True
            nrConstraint += 1
            exprcs.append(exprc)
            if flag:
                model.addConstr(exprc, grb.GRB.EQUAL, 1.0, "c_" + str(nrConstraint))
    model.optimize()

    newposlist = []
    if model.status == grb.GRB.Status.OPTIMAL:
        modstatus = 'Model solved'
        mod = True
        # print(ro[3].Xn)
        # print(model.getVars()[0].Xn)

        # solutions
        solutionIndex = []
        solutions = []
        for i in range(len(ro)):
            if ro[i].Xn > 0.5:
                solutionIndex.append(i)
                solutions.append(ro[i].VarName)

        for name in solutions:
            inds = [m.start() for m in re.finditer('_', name)]
            posid = []
            for idd in range(len(inds)):
                if idd != len(inds) - 1:
                    posid.append(int(name[inds[idd] + 1:inds[idd + 1]]))
                else:
                    posid.append(int(name[inds[idd] + 1:]))

            # established tracks
            if posid[0] < len(poslist):
                pos = poslist[posid[0]]

                # linked detections
                if posid[1] < len(dets[0]):
                    dets[0][posid[1]][9] = 1
                    pos.append(dets[0][posid[1]])
                else:
                    if is3D:
                        nextp = [nextt[posid[0]][0],
                                 nextt[posid[0]][1],
                                 pos[len(pos) - 1][2] + 1,
                                 nextt[posid[0]][2], pos[len(pos) - 1][4], 1, 6, 0, 0, 0]
                    else:
                        nextp = [nextt[posid[0]][0],
                                 nextt[posid[0]][1],
                                 pos[len(pos) - 1][2] + 1,
                                 pos[len(pos) - 1][3], pos[len(pos) - 1][4], 1, 6, 0, 0, 0]
                    nextp = np.asarray(nextp)
                    pos.append(nextp)
                newposlist.append(pos)

        for j in range(len(dets[0])):
            if dets[0][j][9] != 1:
                newposlist.append([dets[0][j]])
    else:
        mod = False
        if model.status == grb.GRB.Status.INF_OR_UNBD:
            modstatus = 'Model is infeasible or unbounded'
        elif model.status == grb.GRB.Status.INFEASIBLE:
            modstatus = 'Model is infeasible'
        elif model.status == grb.GRB.Status.UNBOUNDED:
            modstatus = 'Model is unbounded'
        else:
            modstatus = 'Optimization was stopped with status %d' % model.status

    if isprint:
        for pos in newposlist:
            if len(pos) > 1:
                if pos[len(pos) - 1][4] != pos[len(pos) - 2][4] and pos[len(pos) - 2][0] != -1:
                    print(str(pos))
                    print('\n\n')

    return mod, modstatus, newposlist


def passedFrames(posi):
    passed = 0
    for pi in range(len(posi) - 1, -1, -1):
        if posi[pi][6] != 0:
            passed += 1
        else:
            break
    return passed


def finishedTrack(posfinal, poslist, framelength):
    newposlist = []
    for pi in range(len(poslist)):
        if passedFrames(poslist[pi]) >= 3 or poslist[pi][len(poslist[pi]) - 1][
            2] == framelength - 1:
            le = len(poslist[pi])
            for psi in range(len(poslist[pi]) - 1, -1, -1):
                if poslist[pi][psi][6] == 1:
                    le -= 1
                else:
                    break
            posfinal.append(poslist[pi][:le])
        else:
            newposlist.append(poslist[pi])
    return posfinal, newposlist


def linkingMAP(path_data, path_npy, path_m, tc, thrs, usepred=True, interp=False, isthrs=False,
               gatexy=15, gatez=3,
               framelength=100, imagesize=512, isprint=False):
    start = time.time()
    print('Linking start ... ')
    if tc.scenario == 'VIRUS':
        is3D = True
    else:
        is3D = False

    data = getData(path_data)
    poslist = []
    posfinal = []
    empty = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    t = data[data[:, 2] == 0]
    t = np.random.permutation(t)
    for i in range(len(t)):
        tr = []
        if tc.feature <= 1:
            for s in range(tc.lstms - 1):
                tr.append(empty)
        else:
            for s in range(tc.lstms):
                tr.append(empty)
        tr.append(t[i])
        poslist.append(tr)

    if isprint:
        print('Frame: ' + str(1) + '/' + str(framelength))

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
        motion = MotionRRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H, tc.L,
                            tc.k,
                            tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
    elif tc.feature == 1:
        motion = MotionRRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H, tc.L,
                            tc.k,
                            tc.act, tc.rnn, tc.delta, tc.alpha, True, True)
    elif tc.feature == 2:
        motion = MotionCRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H, tc.L,
                            tc.k,
                            tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
    elif tc.feature == 3:
        motion = MotionCRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H, tc.L,
                            tc.k,
                            tc.act, tc.rnn, tc.delta, tc.alpha, True, True)
    elif tc.feature == 4:
        motion = MotionRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H, tc.L,
                           tc.k,
                           tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
    elif tc.feature == 5:
        motion = MotionRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H, tc.L,
                           tc.k,
                           tc.act, tc.rnn, tc.delta, tc.alpha, True, True)
    elif tc.feature == 6:
        motion = MotionRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H, tc.L,
                           tc.k,
                           tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
    elif tc.feature == 7:
        motion = MotionRNN(n_input, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R, tc.H, tc.L,
                           tc.k,
                           tc.act, tc.rnn, tc.delta, tc.alpha, True, True)
    elif tc.feature == 8:
        motion = MotionRHRNN(n_input, n_inputp, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R,
                             tc.H, tc.L, tc.k,
                             tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
    elif tc.feature == 9:
        motion = MotionRHRNN(n_input, n_inputp, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R,
                             tc.H, tc.L, tc.k,
                             tc.act, tc.rnn, tc.delta, tc.alpha, True, True)
    elif tc.feature == 10:
        motion = MotionRHCNN(n_input, n_inputp, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R,
                             tc.H, tc.L, tc.k,
                             tc.act, tc.rnn, tc.delta, tc.alpha, True, False)
    elif tc.feature == 11:
        motion = MotionRHCNN(n_input, n_inputp, n_output, tc.n_classes, tc.lstms, tc.lstmo, tc.R,
                             tc.H, tc.L, tc.k,
                             tc.act, tc.rnn, tc.delta, tc.alpha, True, True)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # print(tc.gpu)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=tc.gpu)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        saver.restore(sess, path_m + '/model')

        for f in range(1, framelength):
            if isprint:
                print('Frame: ' + str(f + 1) + '/' + str(framelength))

            detectionsM = getDets(f, data, tc.lstmo)  # detection position at frame f
            costs = []
            nextt = []
            detscost = []
            for o in range(len(detectionsM)):
                detscost.append(np.zeros((len(detectionsM[o]) + 1)))
            # compute which track link with which detections counts the times per detections
            for pos in range(len(poslist)):
                posi = poslist[pos]  # tracklets
                true_track = data[data[:, 4] == posi[-1][4]]
                # true track no added fake detections since added fake detections [5] != 255
                true_track = true_track[true_track[:, 5] == 255]
                if len(true_track) < 1:
                    print(true_track)
                # true_trackI = trackI(true_track, is3D)
                mx, mxp, my, mxloc, pd, realy, realp, truepoints, realx = getSequence(posi,
                                                                                      true_track,
                                                                                      detectionsM,
                                                                                      tc, gatexy,
                                                                                      gatez)
                mpc, mpt, mpd = sess.run([motion.predc, motion.nextloc, motion.predd], feed_dict={
                    motion.x: mx,
                    motion.xp: mxp,
                    motion.y: my,
                    motion.xlastloc: mxloc,
                    motion._dropout: 1
                })
                if usepred:  # use classification score
                    conf = mpc[:, 0] * np.array(pd)
                else:  # use cost score
                    conf = mpd[:, 0] * np.array(pd)

                nextt.append(mpt[0])

                if usepred:
                    df = 1
                else:
                    if is3D:
                        df = math.sqrt(gatexy * gatexy * 2 + gatez * gatez)

                    else:
                        df = math.sqrt(gatexy * gatexy * 2)

                for pi in range(len(conf)):
                    c = [conf[pi], pos]
                    for ryi in range(len(realy[pi])):
                        c.append(int(realy[pi][ryi][9]))
                        if not isthrs:
                            detscost[ryi][int(realy[pi][ryi][9])] = np.maximum((df - conf[pi]),
                                                                               detscost[ryi][int(
                                                                                   realy[pi][ryi][
                                                                                       9])])
                    costs.append(c)

            detectionN = [
                len(poslist) + 1]  # existing track number + 1 (not in any existing tracks)
            for di in range(tc.lstmo):
                detectionN.append(
                    len(detectionsM[di]) + 1)  # existing detection number + 1 (dummy detection)
            if tc.lstmo == 1:
                for ai in range(detectionN[1]):
                    if ai != detectionN[1] - 1:
                        if isthrs:
                            c = [thrs, len(poslist), ai]
                        else:
                            c = [detscost[0][ai], len(poslist), ai]
                        costs.append(c)
            # elif tc.lstmo == 2:
            #     for ai in range(detectionN[1]):
            #         if ai != detectionN[1] - 1:
            #             for aai in range(detectionN[2]):
            #                 if aai != detectionN[2] - 1:
            #                     if isthrs:
            #                         c = [thrs, len(poslist), ai, aai]
            #                     else:
            #                         c = [detscost[0][ai] + detscost[1][aai], len(poslist), ai, aai]
            #                     costs.append(c)
            # elif tc.lstmo == 3:
            #     for ai in range(detectionN[1]):
            #         if ai != detectionN[1] - 1:
            #             for aai in range(detectionN[2]):
            #                 if aai != detectionN[2] - 1:
            #                     for aaai in range(detectionN[3]):
            #                         if aaai != detectionN[3] - 1:
            #                             if isthrs:
            #                                 c = [thrs, len(poslist), ai, aai, aaai]
            #                             else:
            #                                 c = [detscost[0][ai] + detscost[1][aai] + detscost[2][aaai],
            #                                      len(poslist), ai, aai, aaai]
            #                             costs.append(c)
            elif tc.lstmo == 2:
                for ai in range(detectionN[1]):
                    for aai in range(detectionN[2]):
                        if ai != detectionN[1] - 1 or aai != detectionN[2] - 1:
                            if isthrs:
                                c = [thrs, len(poslist), ai, aai]
                            else:
                                c = [detscost[0][ai] + detscost[1][aai], len(poslist), ai, aai]
                            costs.append(c)
            elif tc.lstmo == 3:
                for ai in range(detectionN[1]):
                    for aai in range(detectionN[2]):
                        for aaai in range(detectionN[3]):
                            if ai != detectionN[1] - 1 or aai != detectionN[2] - 1 or aaai != \
                                    detectionN[3] - 1:
                                if isthrs:
                                    c = [thrs, len(poslist), ai, aai, aaai]
                                else:
                                    c = [detscost[0][ai] + detscost[1][aai] + detscost[2][aaai],
                                         len(poslist), ai, aai, aaai]
                                costs.append(c)

            # call gurobi solver
            mod, modstatus, poslist = grbsolver(costs, poslist, detectionN, detectionsM, tc.lstmo,
                                                nextt, is3D, isprint)
            if mod:
                posfinal, poslist = finishedTrack(posfinal, poslist, framelength)
            else:
                print(modstatus)
                exit(0)

    pos = []
    if interp:
        for j in range(len(posfinal)):
            posi = []
            for i in range(len(posfinal[j])):
                if posfinal[j][i][0] != -1 and imagesize - 1 >= posfinal[j][i][0] >= 0 \
                        and imagesize - 1 >= posfinal[j][i][1] >= 0 and posfinal[j][i][6] != 6:
                    posi.append([posfinal[j][i][0], posfinal[j][i][1], posfinal[j][i][2],
                                 posfinal[j][i][3], posfinal[j][i][4]])
            if len(posi) > 0:
                pos.append(posi)
    else:
        for j in range(len(posfinal)):
            posi = []
            for i in range(len(posfinal[j])):
                if posfinal[j][i][0] != -1 and imagesize - 1 >= posfinal[j][i][0] >= 0 \
                        and imagesize - 1 >= posfinal[j][i][1] >= 0:
                    posi.append([posfinal[j][i][0], posfinal[j][i][1], posfinal[j][i][2],
                                 posfinal[j][i][3], posfinal[j][i][4]])
            if len(posi) > 0:
                pos.append(posi)

    np.save(path_npy, pos)
    print('Linking finished in ' + '{:.2f}'.format(time.time() - start) + ' sec.')
