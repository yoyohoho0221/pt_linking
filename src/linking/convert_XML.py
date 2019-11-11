# ## read saved tracking results from .npy file and convert to .xml file
#
#

import numpy as np
import os


def filterNPY(pospath, interp, is3D, length=3):
    posFinal = []
    pos = np.load(pospath)
    if interp:
        for a in range(len(pos)):
            posi = []
            id = 0
            posi.append(pos[a][id])

            for id in range(1, len(pos[a])):
                if pos[a][id][2] - pos[a][id - 1][2] != 1:
                    x = (pos[a][id][0] - pos[a][id - 1][0]) / (
                                1.0 * (pos[a][id][2] - pos[a][id - 1][2]))
                    y = (pos[a][id][1] - pos[a][id - 1][1]) / (
                                1.0 * (pos[a][id][2] - pos[a][id - 1][2]))
                    if is3D:
                        z = (pos[a][id][3] - pos[a][id - 1][3]) / (
                                    1.0 * (pos[a][id][2] - pos[a][id - 1][2]))
                    else:
                        z = 0
                    for i in range(int(pos[a][id][2] - pos[a][id - 1][2] - 1)):
                        posi.append(np.array(
                            [(pos[a][id - 1][0] + x * (i + 1)), (pos[a][id - 1][1] + y * (i + 1)),
                             pos[a][id - 1][2] + i + 1, (pos[a][id - 1][0] + z * (i + 1)),
                             pos[a][id][4], 1.0, 1.0, 0.0, 0.0, 0.0]))
                    posi.append(pos[a][id])
                else:
                    posi.append(pos[a][id])

            if len(posi) > length:
                posFinal.append(posi)
        posFinal = np.array(posFinal)
    else:
        posFinal = pos
    return posFinal


def writeXML(pospath, filepath, scenario, snr, dens, method, thrs, interp):
    if scenario == 'VIRUS':
        is3D = True
    else:
        is3D = False

    pos = filterNPY(pospath, interp, is3D)
    with open(filepath, "w+") as output:
        output.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        output.write('<root>\n')
        output.write('<TrackContestISBI2012 SNR="' + str(
            snr) + '" density="' + dens + '" scenario="' + scenario + \
                     '" ' + method + '="' + str(thrs) + '">\n')
        for i in range(len(pos)):
            output.write('<particle>\n')
            for a in range(len(pos[i])):
                if is3D:
                    output.write('<detection t="' + str(int(pos[i][a][2])) +
                                 '" x="' + str(pos[i][a][0]) +
                                 '" y="' + str(pos[i][a][1]) +
                                 '" z="' + str(pos[i][a][3]) + '"/>\n')
                else:
                    output.write('<detection t="' + str(int(pos[i][a][2])) +
                                 '" x="' + str(pos[i][a][0]) +
                                 '" y="' + str(pos[i][a][1]) + '" z="0"/>\n')
            output.write('</particle>\n')
        output.write('</TrackContestISBI2012>\n')
        output.write('</root>\n')
        output.close()


def readXML(file):
    with open(file) as f:
        lines = f.readlines()
    f.close()
    poslist = []
    p = 0
    for i in range(len(lines)):
        if '<particle>' in lines[i]:
            posi = []
        elif '<detection t=' in lines[i]:
            ind1 = lines[i].find('"')
            ind2 = lines[i].find('"', ind1 + 1)
            t = int(lines[i][ind1 + 1:ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            x = float(lines[i][ind1 + 1:ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            y = float(lines[i][ind1 + 1:ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            z = float(lines[i][ind1 + 1:ind2])
            posi.append([x, y, t, z, p])
        elif '</particle>' in lines[i]:
            p += 1
            poslist.append(posi)
    return poslist


def writeTXT(file, sort=False):
    poslist = readXML(file)
    npy = []
    for posi in poslist:
        for p in posi:
            npy.append([p[0], p[1], p[2], p[3], p[4], 255.0])

    if sort:
        filepath = file[:-3] + 'detections.xml.txt'
        npy = np.array(npy)
        ind = np.argsort(npy[:, 2])
        npy = npy[ind]
    else:
        filepath = file + '.txt'

    with open(filepath, "w+") as output:
        for p in npy:
            output.write('{:.3f}'.format(p[0]) + '\t{:.3f}'.format(p[1]) + '\t{:.3f}'.format(
                p[2]) + '\t{:.3f}'.format(p[3]) + '\t{:.3f}'.format(p[4]) + '\t{:.3f}'.format(p[5])
                         + '\t0.000\t0.000\t0.000\t0.000\n')
    output.close()


def filterXML(filepath, scenario, snr, dens, thrs=-1, imagesize=512):
    pos = readXML(filepath)
    # newfilepath = filepath[:filepath.find('.xml')]+'ftd.xml'
    with open(filepath, "w+") as output:
        output.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        output.write('<root>\n')
        if thrs > 0:
            output.write('<TrackContestISBI2012 SNR="' + str(
                snr) + '" density="' + dens + '" scenario="' + scenario + '" TS="' + str(
                thrs) + '">\n')
        elif thrs == -1:
            output.write('<TrackContestISBI2012 SNR="' + str(
                snr) + '" density="' + dens + '" scenario="' + scenario + '" KM="' + str(
                thrs) + '">\n')
        elif thrs == -2:
            output.write('<TrackContestISBI2012 SNR="' + str(
                snr) + '" density="' + dens + '" scenario="' + scenario + '" RL="True">\n')

        for i in range(len(pos)):
            if len(pos[i]) > 3:
                output.write('<particle>\n')
                for a in range(len(pos[i])):
                    if imagesize >= pos[i][a][0] >= 0 and imagesize >= pos[i][a][1] >= 0:
                        if scenario == 'VIRUS' and pos[i][a][3] > 0:
                            output.write('<detection t="' + str(int(pos[i][a][2])) +
                                         '" x="' + str(pos[i][a][0]) +
                                         '" y="' + str(pos[i][a][1]) +
                                         '" z="' + str(int(pos[i][a][3])) + '"/>\n')
                        else:
                            output.write('<detection t="' + str(int(pos[i][a][2])) +
                                         '" x="' + str(pos[i][a][0]) +
                                         '" y="' + str(pos[i][a][1]) + '" z="0"/>\n')
                output.write('</particle>\n')
        output.write('</TrackContestISBI2012>\n')
        output.write('</root>\n')
        output.close()


def filterNPYilya(pos, interp, is3D, length=3):
    posFinal = []
    if interp:
        for a in range(len(pos)):
            posi = []
            id = 0
            posi.append(pos[a][id])

            for id in range(1, len(pos[a])):
                if pos[a][id][2] - pos[a][id - 1][2] != 1:
                    x = (pos[a][id][0] - pos[a][id - 1][0]) / (
                                1.0 * (pos[a][id][2] - pos[a][id - 1][2]))
                    y = (pos[a][id][1] - pos[a][id - 1][1]) / (
                                1.0 * (pos[a][id][2] - pos[a][id - 1][2]))
                    if is3D:
                        z = (pos[a][id][3] - pos[a][id - 1][3]) / (
                                    1.0 * (pos[a][id][2] - pos[a][id - 1][2]))
                    else:
                        z = 0
                    for i in range(int(pos[a][id][2] - pos[a][id - 1][2] - 1)):
                        posi.append(np.array(
                            [(pos[a][id - 1][0] + x * (i + 1)), (pos[a][id - 1][1] + y * (i + 1)),
                             pos[a][id - 1][2] + i + 1, (pos[a][id - 1][0] + z * (i + 1)),
                             pos[a][id][4], 1.0, 1.0, 0.0, 0.0, 0.0]))
                    posi.append(pos[a][id])
                else:
                    posi.append(pos[a][id])

            if len(posi) > length:
                posFinal.append(posi)
        posFinal = np.array(posFinal)
    else:
        posFinal = pos
    return posFinal


def readXMLilya(file):
    with open(file) as f:
        lines = f.readlines()
    f.close()
    poslist = []
    p = 0
    for i in range(len(lines)):
        if '<track id=' in lines[i]:
            posi = []
        elif '<point id=' in lines[i]:
            ind1 = lines[i].find('"')
            ind2 = lines[i].find('"', ind1 + 1)
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            t = int(lines[i][ind1 + 1:ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            x = float(lines[i][ind1 + 1:ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            y = float(lines[i][ind1 + 1:ind2])
            # ind1 = lines[i].find('"', ind2 + 1)
            # ind2 = lines[i].find('"', ind1 + 1)
            # z = float(lines[i][ind1 + 1:ind2])
            z = 0
            posi.append([x, y, t - 1, z, p])
        elif '</track>' in lines[i]:
            p += 1
            poslist.append(posi)
    return poslist


def writeTXTilya(file, sort=False):
    if os.path.exists(file):
        poslist = readXMLilya(file)
        npy = []
        for posi in poslist:
            for p in posi:
                npy.append([p[0], p[1], p[2], p[3], p[4], 255.0])

        if sort:
            filepath = file[:-3] + 'detections.xml.txt'
            npy = np.array(npy)
            ind = np.argsort(npy[:, 2])
            npy = npy[ind]
        else:
            filepath = file + '.txt'

        with open(filepath, "w+") as output:
            for p in npy:
                output.write('{:.3f}'.format(p[0]) + '\t{:.3f}'.format(p[1]) + '\t{:.3f}'.format(
                    p[2]) + '\t{:.3f}'.format(p[3]) + '\t{:.3f}'.format(p[4]) + '\t{:.3f}'.format(p[5])
                             + '\t0.000\t0.000\t0.000\t0.000\n')
        output.close()


def writeXMLilya(pospath, filepath, no, method, thrs, interp, filetype='xml'):
    if os.path.exists(pospath):
        is3D = False
        if filetype == 'xml':
            pos = readXMLilya(pospath)
            pos = filterNPYilya(pos, interp, is3D, 2)
        else:
            pos = np.load(pospath)
        with open(filepath, "w+") as output:
            output.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
            output.write('<root>\n')
            # output.write('<TrackContestISBI2012 SNR="1" density="low" generationDateTime="Mon Feb 13 18:26:25 CET 2012" info="http://bioimageanalysis.org/track/" scenario="MICROTUBULE">\n')
            output.write(
                '<TrackContestISBI2012 ilya="' + (no) + '" ' + method + '="' + str(thrs) + '">\n')
            for i in range(len(pos)):
                output.write('<particle>\n')
                for a in range(len(pos[i])):
                    if is3D:
                        output.write('<detection t="' + str(int(pos[i][a][2])) +
                                     '" x="' + str(pos[i][a][0]) +
                                     '" y="' + str(pos[i][a][1]) +
                                     '" z="' + str(pos[i][a][3]) + '"/>\n')
                    else:
                        output.write('<detection t="' + str(int(pos[i][a][2])) +
                                     '" x="' + str(pos[i][a][0]) +
                                     '" y="' + str(pos[i][a][1]) + '" z="0"/>\n')
                output.write('</particle>\n')
            output.write('</TrackContestISBI2012>\n')
            output.write('</root>\n')
            output.close()


if __name__ == '__main__':
    '''
    scenarios = ['MICROTUBULE', 'VESICLE', 'RECEPTOR', 'VIRUS']
    for scenario in scenarios:
        for snrid in [1, 2, 4, 7]:
            for dens in ['high', 'mid', 'low']:
                path = ('/home/yyao/IDEA/dl/data/MTJ_RNN/data/XML/' + scenario + ' snr ' + str(
                    snrid) + ' density ' + dens + '.xml')
                writeTXT(path, sort=True)
    '''

    for type in ['rab5', 'rab6','rab11','eb3']:
        if type == 'rab5':
            for no in ['01','02','03','04']:
                path = '/home/yyao/IDEA/dl/data/ilya/'+type+\
                       '/20180712 HeLa control mCherry-Rab5 ILAS2 x100 100ms -' + (no) + '-.xml'
                filepath = '/home/yyao/IDEA/dl/data/ilya/'+type+\
                       '/20180712 HeLa control mCherry-Rab5 ILAS2 x100 100ms -' + (no) + '-isbi.xml'
                writeTXTilya(path, sort=True)
                writeXMLilya(path, filepath, no, 'ORG', 0, 0)
        elif type == 'rab6':
            for no in ['04', '05', '08', '10']:
                path = '/home/yyao/IDEA/dl/data/ilya/'+type+\
                       '/20180831 HeLa Control Rab6-mCherry ILAS2 x100 100ms -' + (no) + '-.xml'
                filepath = '/home/yyao/IDEA/dl/data/ilya/'+type+\
                       '/20180831 HeLa Control Rab6-mCherry ILAS2 x100 100ms -' + (no) + '-isbi.xml'
                writeTXTilya(path, sort=True)
                writeXMLilya(path, filepath, no, 'ORG', 0, 0)
        elif type == 'rab11':
            for no in ['01','02','03']:#,'04','05','06']:
                path = '/home/yyao/IDEA/dl/data/ilya/'+type+\
                       '/20180509 Rab11-GFP control ILAS2 x100 100ms -' + (no) + '-.xml'
                filepath = '/home/yyao/IDEA/dl/data/ilya/'+type+\
                       '/20180509 Rab11-GFP control ILAS2 x100 100ms -' + (no) + '-isbi.xml'
                writeTXTilya(path, sort=True)
                writeXMLilya(path, filepath, no, 'ORG', 0, 0)
        else:
            for no in ['02','03','04','05']:
                path = '/home/yyao/IDEA/dl/data/ilya/'+type+\
                       '/20151021 ILAS2 HeLa EB3-GFP control x100 500ms -' + (no) + '-.xml'
                filepath = '/home/yyao/IDEA/dl/data/ilya/'+type+\
                       '/20151021 ILAS2 HeLa EB3-GFP control x100 500ms -' + (no) + '-isbi.xml'
                writeTXTilya(path, sort=True)
                writeXMLilya(path, filepath, no, 'ORG', 0, 0)