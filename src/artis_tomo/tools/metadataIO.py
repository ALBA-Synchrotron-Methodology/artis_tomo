#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 12:10:16 2018

@author: joton
"""

import subprocess as sp
import numpy as np
import pandas as pd
import re
from transforms3d.euler import euler2mat, mat2euler
import copy


def getStarData(fName, tableName='', columnsList=''):

    command = 'relion_star_printtable {} {} {}'.format(fName,
                                                       tableName,
                                                       columnsList)

    subp = sp.Popen(command.split(), stdin=sp.PIPE,
                    stdout=sp.PIPE, stderr=sp.PIPE)

    lines = subp.stdout.readlines()

    nLines = len(lines)

    nCol = len(lines[0].split())

    outarray = np.zeros((nLines, nCol))

    for k in range(nLines):
        outarray[k, :] = np.array(list(map(float, lines[k].split())))

    return outarray


def getStarText(fName, tableName='', columnsList=''):

    command = 'relion_star_printtable {} {} {}'.format(fName,
                                                       tableName,
                                                       columnsList)

    subp = sp.Popen(command.split(), stdin=sp.PIPE,
                    stdout=sp.PIPE, stderr=sp.PIPE)

    lines = subp.stdout.readlines()

    outarray = [line.split() for line in lines]

    return outarray


def getTable(lines):

    nLines = len(lines)
    lId = 0
    labels = list()

    if 'loop_' in lines[lId]:
        lId += 1
        while lines[lId][0] == '_':
            labels.append(lines[lId].split()[0])
            lId += 1

        datalist = [lines[k].split() for k in range(lId, nLines)]
        data = pd.DataFrame(datalist, columns=labels)
    else:
        values = list()
        while lId < nLines and lines[lId][0] == '_':
            label, value = lines[lId].split()
            labels.append(label)
            values.append(value)
            lId += 1
        data = pd.DataFrame(1, columns=['value'], index=labels)
        data['value'] = values

    return data


def readStarFile(fName, mytable=''):

    with open(fName, 'r') as file:
        lines = file.read().splitlines()

    # Remove empty lines
    lines = [line for line in lines if line]
    # Remove single space lines
    lines = [line for line in lines if line not in ' ']
    # Remove comment lines
    lines = [line for line in lines if line[0] not in '#']
    nLines = len(lines)

    # Get tables
    tables = list()
    tLines = list()  # lines where tables start
    for k, line in enumerate(lines):
        if 'data_' in line[:10]:
            tables.append(line)
            tLines.append(k)

    nTables = len(tables)
    tRanges = [None]*nTables

    for k in range(nTables-1):
        tRanges[k] = (tLines[k] + 1, tLines[k+1])
    tRanges[-1] = (tLines[-1] + 1, nLines)

    data = dict()

    for k, table in enumerate(tables):
        if np.diff(tRanges[k]) > 1:
            data[table] = getTable(lines[slice(*tRanges[k])])
        else:
            data[table] = ''

    if mytable:
        return data[mytable]
    else:
        return data


def writeStarFile(fName, data, tableName='data_'):

    tData = type(data)

    if tData == dict:
        mydata = data
    elif tData == pd.core.frame.DataFrame:
        mydata = {}
        mydata[tableName] = data

    file = open(fName, 'w')

    for tName, table in mydata.items():
        labels = table.columns

        file.write('\n# version 30001 by artis_tomo\n\n')
        file.write(f'{tName}\n')
        file.write('\nloop_\n')

        for k, label in enumerate(labels):
            file.write(f'{label} #{k+1}\n')

        for k, line in table.iterrows():
            linesep = "\t".join(line)
            file.write(f'{linesep}\n')

    file.close()


def writeRelionStarFileSelection(fName, dataStar, selectPos):

    dataStarOut = copy.deepcopy(dataStar)
    newdataPart = dataStarOut['data_particles']

    newdataPart = newdataPart.iloc[selectPos].reset_index(drop=True)
    dataStarOut['data_particles'] = newdataPart
    writeStarFile(fName, dataStarOut)


def setLabel(data, labellist, label, value):

    if label in labellist:
        myCol = labellist.index(label)
    else:
        labellist.append(label)
        [x.append(None) for x in data]
        myCol = len(labellist) - 1

    for x in data:
        x[myCol] = value

    return myCol


def readCTFFile(fName):

    with open(fName, 'r') as file:
        lines = file.readlines()

    isCTFFind = 'CTFFind' in lines[0]

    if isCTFFind:
        kl = 0
        while '#' in lines[kl]:
            kl += 1
        cols = np.array([1, 2, 3])
        factor = np.array([1, 1, 1])  # Defocus is in Angstrom
    else:  # CTFplotter
        kl = 1
        cols = [4, 5, 6]
        factor = np.array([10, 10, 1])  # Defocus is in nm

    nL = len(lines) - kl
    nC = len(np.fromstring(lines[kl], sep=' '))
    data = np.zeros((nL, nC))
    for k in range(nL):
        data[k, :] = np.fromstring(lines[k + kl], sep=' ')

    ov = np.ones(nL)
    output = data[:, cols]*(np.column_stack((ov,)*3)*factor)
    return output


def readTiltComFile(fName):
    """Read Imod tiltcom file and convert it into dictionary."""
    with open(fName, 'r') as file:
        lines = file.readlines()

    tiltcomData = dict()
    for line in lines:
        if line[0] not in ['#', '$']:
            spl = line.split()
            tiltcomData[spl[0]] = spl[1:]

    return tiltcomData


def readTransformFile(fName):
    """Read Imod transform XF file and convert it into matrices."""
    dataArray = np.loadtxt(fName)
    nRows = len(dataArray)
    transArray = [None]*nRows

    for k in range(nRows):
        array = np.zeros((3, 3))
        array[:2, :2] = dataArray[k][:4].reshape(2, 2)
        array[:2, 2] = dataArray[k][4:]
        array[2, 2] = 1
        transArray[k] = array

    return transArray


def writeImodTransformFile(fName, transArrayList):

    lines = list()

    for array in transArrayList:
        lines.append(f'{array[0, 0]:.7f}\t{array[0, 1]:.7f}\t' +
                     f'{array[1, 0]:.7f}\t{array[1, 1]:.7f}\t' +
                     f'{array[0, 2]:.3f}\t{array[1, 2]:.3f}\n')

    with open(fName, 'w') as file:
        file.writelines(lines)



def motlToRelionStar3(motl, pixelSizeBin1, Cs=2.7, voltage=300,
                      binning=1, tomoLabel='TS_', padZeros=-1):

    nPart = len(motl)

    labels = ['_rlnTomoName', '_rlnTomoParticleId', '_rlnTomoManifoldIndex',
              # '_rlnMicrographName',
              '_rlnCoordinateX', '_rlnCoordinateY', '_rlnCoordinateZ',
              '_rlnOriginXAngst', '_rlnOriginYAngst', '_rlnOriginZAngst',
              '_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi',
              # '_rlnGroupNumber',
              '_rlnClassNumber']
    datapart = pd.DataFrame(index=range(nPart), columns=labels)

    if padZeros < 0:
        tomoNumW = len(str(int(motl[:, 6].max())))
    else:
        tomoNumW = padZeros

    for k in range(nPart):
        dOnePart = datapart.loc[k]
        motlOne = motl[k, :]
        partIdx = int(motlOne[3])
        tomoIdx = int(motlOne[6])
        tomoName = f'{tomoLabel}{tomoIdx:0{tomoNumW}d}'
        dOnePart['_rlnTomoName'] = f'{tomoName}'
        dOnePart['_rlnTomoParticleId'] = f'{partIdx}'
        dOnePart['_rlnTomoManifoldIndex'] = f'{int(motlOne[5])}'
        # dOnePart['_rlnMicrographName'] = f'{tomoName}'
        # dOnePart['_rlnGroupNumber'] = f'{tomoIdx}'
        dOnePart['_rlnClassNumber'] = f'{int(motlOne[19])}'

        coords = motlOne[7:10].astype(float) - 1
        shifts = coords + motlOne[10:13].astype(float)
        shifts = shifts*binning + binning//2
        coords = np.round(shifts)
        shifts = -(shifts - coords)*pixelSizeBin1
        # shifts = -1.*motlOne[10:13].astype(float)*pixelSize

        # Angles applied to rotate part -> ref (Relion ref.system)
        newAngles = -np.array(mat2euler(
                    euler2mat(*(np.pi/180*motlOne[[16, 18, 17]]),
                              'szxz'), 'rzyz'))*180/np.pi

        dOnePart['_rlnCoordinateX'] = f'{int(coords[0])}'
        dOnePart['_rlnCoordinateY'] = f'{int(coords[1])}'
        dOnePart['_rlnCoordinateZ'] = f'{int(coords[2])}'
        dOnePart['_rlnOriginXAngst'] = f'{shifts[0]:.3f}'
        dOnePart['_rlnOriginYAngst'] = f'{shifts[1]:.3f}'
        dOnePart['_rlnOriginZAngst'] = f'{shifts[2]:.3f}'
        dOnePart['_rlnAngleRot'] = f'{newAngles[2]:.2f}'
        dOnePart['_rlnAngleTilt'] = f'{newAngles[1]:.2f}'
        dOnePart['_rlnAnglePsi'] = f'{newAngles[0]:.2f}'

    data = dict()
    data['data_particles'] = datapart
    return data


def getFieldIfExists(dataframe, label):

    if label in dataframe:
        return dataframe[label]
    else:
        return 0


def relionStar3ToMotl(dataStar, binfactor=1, useTomoAngles=False):

    datao = dataStar['data_optics']
    datapart = dataStar['data_particles']
    pixelSize = float(datao['_rlnTomoTiltSeriesPixelSize'])

    binRatio = 1/binfactor

    nPart = len(datapart)
    motl = np.zeros((nPart, 20))

    if useTomoAngles:
        rotLabel  = '_rlnTomoSubtomogramRot'
        tiltLabel = '_rlnTomoSubtomogramTilt'
        psiLabel  = '_rlnTomoSubtomogramPsi'
    else:
        rotLabel  = '_rlnAngleRot'
        tiltLabel = '_rlnAngleTilt'
        psiLabel  = '_rlnAnglePsi'


    for k in range(nPart):
        dOnePart = datapart.iloc[k]
        motlOne = motl[k, :]

        motlOne[3] = int(getFieldIfExists(dOnePart, '_rlnTomoParticleId'))
        motlOne[5] = int(getFieldIfExists(dOnePart, '_rlnTomoManifoldIndex'))
        tomoIdx = np.array(int(re.findall('_\d+.', dOnePart['_rlnTomoName'])[0][1:]) )
        motlOne[6] = tomoIdx
        motlOne[0] = float(getFieldIfExists(dOnePart, '_rlnMaxValueProbDistribution'))
        motlOne[19] = int(getFieldIfExists(dOnePart, '_rlnClassNumber'))

        coords = dOnePart[['_rlnCoordinateX',
                           '_rlnCoordinateY',
                           '_rlnCoordinateZ']].\
                            values.astype(float)*binRatio
        # Subtom shifts are defined opposite to Relion and in pixels
        shifts = -dOnePart[['_rlnOriginXAngst',
                           '_rlnOriginYAngst',
                           '_rlnOriginZAngst']].\
                            values.astype(float)/pixelSize*binRatio

        # Angles applied to rotate part -> ref (Relion ref.system)
        angles = dOnePart[[psiLabel,
                           tiltLabel,
                           rotLabel]].\
                            values.astype(float)

        newAngles = np.array(mat2euler(
                    euler2mat(*(-np.pi/180*angles),
                              'rzyz'), 'szxz'))*180/np.pi

        # Subtom indexes start from 1
        shifts += coords + 1
        coords = np.round(shifts)
        shifts -= coords

        motlOne[7:10] = coords
        motlOne[10:13] = shifts
        motlOne[16:19] = newAngles[[0, 2, 1]]

    return motl


def getShiftMatrix(sx, sy, sz):

    shiftM = np.zeros((4, 4))
    shiftM[[0, 1, 2, 3], [0, 1, 2, 3]] = 1
    shiftM[0:3, 3] = np.array([sx, sy, sz])
    return shiftM


def getTransMatrix(angles=np.array([0, 0, 0]), shifts=np.array([0, 0, 0])):
    """
    Parameters
    ----------
    angles : 1D array
        with Rot, Tilt and Psi Relion angles
    shifts : 1D array
        Xshift, Yshift and Zshift

    Returns
    -------
    Trans matrix  : 2D array with transformation matrix which applies from
                    reference to particle
    """

    transMat = np.zeros((4, 4))
    transMat[3, 3] = 1
    transMat[:3, :3] = euler2mat(*(np.pi/180*angles), 'szyz')
    transMat[:3, 3] = shifts

    return transMat


def getTransFromMatrix(transMatrix):

    shifts = transMatrix[0:3, 3]

    angles = np.array(mat2euler(transMatrix[0:3, 0:3], 'szyz'))*180/np.pi

    if angles[1] < 0:
        angles[[0, 2]] += 180
        angles[1] *= -1

    return angles, shifts


def makeTiltPositive(anglesList):

    anglesList = np.copy(anglesList)

    if anglesList.ndim == 1:
        anglesList = anglesList[np.newaxis, :]

    for angles in anglesList:
        if angles[1] < 0:
            angles[[0, 2]] += 180
            angles[1] *= -1

    return np.squeeze(anglesList)


def applyTransform(dataPart, angles=np.array([0, 0, 0]),
                   shifts=np.array([0, 0, 0])):

    dfSize = dataPart.shape
    if len(dfSize) < 2:
        nPart = 1
    else:
        nPart = dfSize[0]

    oldAngles = dataPart[['_rlnAngleRot',
                          '_rlnAngleTilt',
                          '_rlnAnglePsi']].values.astype(float)
    oldShifts = dataPart[['_rlnOriginXAngst',
                          '_rlnOriginYAngst',
                          '_rlnOriginZAngst']].values.astype(float)

    # nPart = len(oldAngles)
    # newAngles = np.zeros(oldAngles.shape)
    # newShifts = np.zeros(oldShifts.shape)

    # shifts applied to reference
    globalShiftMat = getShiftMatrix(*shifts)
    # rotation applied to reference
    globalRotMat = getTransMatrix(angles)

    dataPartOut = copy.deepcopy(dataPart)

    if nPart == 1:
        # rotmat from Ref to particle
        rotMat = getTransMatrix(-oldAngles, oldShifts)

        # Trans from particle to ref, apply new changes to ref
        newTransMat = globalRotMat@globalShiftMat@np.linalg.inv(rotMat)
        # Back to particle
        newTransMatInv = np.linalg.inv(newTransMat)
        newAngles, newShifts = getTransFromMatrix(newTransMatInv)

        # Put back angles in Particle system
        newAngles *= -1
        newAngles = makeTiltPositive(newAngles)
        # print(dataPartOut)

        dataPartOut['_rlnAngleRot'] = f'{newAngles[0]:.3f}'
        dataPartOut['_rlnAngleTilt'] = f'{newAngles[1]:.3f}'
        dataPartOut['_rlnAnglePsi'] = f'{newAngles[2]:.3f}'
        dataPartOut['_rlnOriginXAngst'] = f'{newShifts[0]:.3f}'
        dataPartOut['_rlnOriginYAngst'] = f'{newShifts[1]:.3f}'
        dataPartOut['_rlnOriginZAngst'] = f'{newShifts[2]:.3f}'

    else:

        for k in range(nPart):
            dOnePart = dataPartOut.iloc[k]
            # rotmat from Ref to particle
            rotMat = getTransMatrix(-oldAngles[k, :], oldShifts[k, :])

            # Trans from particle to ref, apply new changes to ref
            newTransMat = globalRotMat@globalShiftMat@np.linalg.inv(rotMat)
            # Back to particle
            newTransMatInv = np.linalg.inv(newTransMat)
            newAngles, newShifts = getTransFromMatrix(newTransMatInv)
            # print(f'newAngles = {newAngles}')

            # Put back angles in Particle system
            newAngles *= -1
            newAngles = makeTiltPositive(newAngles)

            # print(dOnePart)

            dOnePart['_rlnAngleRot'] = f'{newAngles[0]:.3f}'
            dOnePart['_rlnAngleTilt'] = f'{newAngles[1]:.3f}'
            dOnePart['_rlnAnglePsi'] = f'{newAngles[2]:.3f}'
            dOnePart['_rlnOriginXAngst'] = f'{newShifts[0]:.3f}'
            dOnePart['_rlnOriginYAngst'] =  f'{newShifts[1]:.3f}'
            dOnePart['_rlnOriginZAngst'] =  f'{newShifts[2]:.3f}'


        # dataPartOut['_rlnAngleRot'] = [f'{x:.3f}' for x in newAngles[:, 0]]
        # dataPartOut['_rlnAngleTilt'] = [f'{x:.3f}' for x in newAngles[:, 1]]
        # dataPartOut['_rlnAnglePsi'] = [f'{x:.3f}' for x in newAngles[:, 2]]
        # dataPartOut['_rlnOriginXAngst'] = [f'{x:.3f}' for x in newShifts[:, 0]]
        # dataPartOut['_rlnOriginYAngst'] = [f'{x:.3f}' for x in newShifts[:, 1]]
        # dataPartOut['_rlnOriginZAngst'] = [f'{x:.3f}' for x in newShifts[:, 2]]

    return dataPartOut


def splitParticleStarFileByTomoName(fName, fnOutRoot, nSplit):

    data = readStarFile(fName)
    datap = data['data_particles']
    nPart = len(datap)
    tomoV = datap['_rlnTomoName'].values
    tomoVList = np.unique(tomoV)

    nTomos = len(tomoVList)

                # Using the total number of jobs
    jobParts = np.ones(nSplit + 1, 'i') * int(nTomos/nSplit)
    jobParts[:np.mod(nTomos, nSplit)] += 1
    idxEndT = jobParts.cumsum()[:nSplit].astype(int)
    idxIniT = (idxEndT - jobParts[:nSplit]).astype(int)

    for ks in range(nSplit):
        pos = np.full(nPart, False, bool)
        for kt in range(idxIniT[ks], idxEndT[ks]):
            print(f'ks={ks}, kt={kt}')

            pos = np.logical_or(pos, tomoV == tomoVList[kt])

        fnOut = fnOutRoot + f'_{ks+1}.star'
        writeRelionStarFileSelection(fnOut, data, pos)


def joinParticlesStarFiles(fNameList, fnOut, label=None):

    data = readStarFile(fNameList[0])
    datapList = list()
    datapList.append(data['data_particles'])

    for file in fNameList[1:]:
        datapList.append(readStarFile(file, 'data_particles'))

    datap = pd.concat(datapList)

    if label is not None:
        datap = datap.sort_values(by=[label])

    data['data_particles'] = datap
    writeStarFile(fnOut, data)



def splitTomogramsStarFileByTomoName(fName, fnOutRoot, nSplit):

    data = readStarFile(fName)
    tomoVdo = data['data_optics']['_rlnTomoName'].values
    tomoV = data['data_particles']['_rlnTomoName'].values
    tomoVList = np.unique(tomoV)
    nPart = len(tomoV)

    nTomos = len(tomoVList)

                # Using the total number of jobs
    jobParts = np.ones(nSplit + 1, 'i') * int(nTomos/nSplit)
    jobParts[:np.mod(nTomos, nSplit)] += 1
    idxEndT = jobParts.cumsum()[:nSplit].astype(int)
    idxIniT = (idxEndT - jobParts[:nSplit]).astype(int)

    for ks in range(nSplit):
        pos = np.full(nPart, False, bool)
        for kt in range(idxIniT[ks], idxEndT[ks]):
            posDo = tomoVdo == tomoVList[kt]
            pos = np.logical_or(pos, tomoV == tomoVList[kt])

        dataStarOut = copy.deepcopy(data)
        newdataPart = dataStarOut['data_particles']
        newdataO = dataStarOut['data_optics']

        dataStarOut['data_particles'] = newdataPart.iloc[pos].reset_index(drop=True)
        dataStarOut['data_optics'] = newdataO.iloc[posDo].reset_index(drop=True)

        fnOut = fnOutRoot + f'_{ks+1}.star'
        writeStarFile(fnOut, dataStarOut)


