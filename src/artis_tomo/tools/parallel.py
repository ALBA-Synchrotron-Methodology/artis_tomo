#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:58:28 2019

@author: joton
"""

import sys
import os
import time
import socket
import itertools
from tqdm import tqdm
# import SharedArray as sa
import numpy as np
from joblib import Parallel, delayed


class mpiNode():
    """Class to handle MPI stuff."""

    def __init__(self, mode=None, verb=True):

        if mode is None:
            self.mpiMode = self.getMpiMode()
        else:
            self.mpiMode = mode

        self.host = socket.gethostname()

        if self.mpiMode:
            from mpi4py import MPI
            import dill
            MPI.pickle.__init__(dill.dumps, dill.loads)

            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.isMaster = bool(self.rank == 0)

            self.printDescr(verb)
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
            self.isMaster = True

    def printDescr(self, verb=True):
        """Print the processes list."""
        if verb:
            if self.isMaster:
                print(f'=== MPI Setup: {self.size} processes ===', flush=True)
                print(f'Master {0:3d} @ {self.host}', flush=True)

                for k in range(1, self.size):
                    slaveHost = self.comm.recv(source=k)
                    print(f'Slave  {k:3d} @ {slaveHost}', flush=True)

                print('===============================', flush=True)

            else:
                self.comm.send(self.host, dest=0)

    @staticmethod
    def getMpiMode():
        """Check if it's running by a mpirun process."""
        macrosList = ['PMI_SIZE',
                      'OMPI_COMM_WORLD_SIZE']

        for macro in macrosList:
            value = os.getenv(macro)
            if value is not None and int(value) > 0:
                return True
        return False


# class parallelPBar():
#     """Progress bar used in parallel."""

#     def __init__(self, progressFile=''):



def monitor(params):

    mpiMode = params.get('mpiMode')
    attachProgress = params.get('attachProgress')
    mpiMaster = True

    host = socket.gethostname()
    print(f'DEBUG: MONITOR: Host={host}, mpiMode={mpiMode}', flush=True)

    if mpiMode:
        from mpi4py import MPI
        # import dill
        # MPI.pickle.__init__(dill.dumps, dill.loads)
        comm = MPI.COMM_WORLD
        mpiRank = comm.Get_rank()
        mpiMaster = mpiRank == 0
        # print(f'DEBUG: MONITOR: Host={host} Rank={mpiRank}. mpiMaster={mpiMaster}', flush=True)

    progress = sa.attach(attachProgress)
    nPart = len(progress)
    nDone = np.sum(progress == 1)
    myProgress = np.zeros((nPart), 'd')
    myProgressTot = np.zeros((nPart), 'd')

    if mpiMaster:
        pbar = tqdm(total=nPart, unit="Part", initial=nDone,
                    dynamic_ncols=True, file=sys.stdout)

        while nDone < nPart:
            pbar.n = nDone
            pbar.refresh()
            if mpiMode:
                myProgress[...] = progress[...]
                comm.Reduce(myProgress, myProgressTot, op=MPI.MAX, root=0)
                nDone = np.sum(myProgressTot)
                comm.bcast(nDone, root=0)
                #  We just update the completed jobs
                pos = np.where((myProgressTot-progress) == 1)[0]
                progress[pos] = 1
            else:
                nDone = np.sum(progress)
            time.sleep(1)
        pbar.n = nDone
        pbar.refresh()
    else:
        while nDone < nPart:
            myProgress[...] = progress[...]
            comm.Reduce(myProgress, myProgressTot, op=MPI.MAX, root=0)
            comm.bcast(nDone, root=0)
            time.sleep(1)


class ProgressParallel(Parallel):
    def __init__(self, desc=None, total=None, unit='it', use_tqdm=True,
                 *args, **kwargs):

        self._desc = desc
        self._total = total
        self._unit = unit
        self._use_tqdm = use_tqdm
        self._tqdmArgs = [None]
        self._tqdmKwargs = dict()

        super().__init__(*args, **kwargs)

    def setProgressBar(self, desc=None, total=None, unit='it', use_tqdm=True,
                       *args, **kwargs):
        self._desc = desc
        self._total = total
        self._unit = unit
        self._use_tqdm = use_tqdm
        self._tqdmArgs = args
        self._tqdmKwargs = kwargs

    def __call__(self, iterable):
        if self._total is None:
            iterable, itertmp = itertools.tee(iterable)
            self._total = len(list(itertmp))
        if self._total > 1:
            with tqdm(desc=self._desc, total=self._total, unit=self._unit,
                      disable=not self._use_tqdm,
                      *self._tqdmArgs, **self._tqdmKwargs) as self._pbar:
                return Parallel.__call__(self, iterable)
        else:
            return Parallel.__call__(self, iterable)

    def print_progress(self):
        if self._total is None:
            self._total = self.n_dispatched_tasks
            self._pbar.total = self.n_dispatched_tasks
        elif self._total > 1:
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()


def splitTasks(nTasks, nProcs):

    jobParts = np.ones(nProcs + 1) * int(nTasks/nProcs)
    jobParts[:np.mod(nTasks, nProcs)] += 1
    idxEnd = jobParts.cumsum()[:nProcs].astype(int)
    idxIni = (idxEnd - jobParts[:nProcs]).astype(int)

    return idxIni, idxEnd


def taskDispatcherProcess(nTasks, nProcs, procID):

    idxIni, idxEnd = splitTasks(nTasks, nProcs)

    for k in range(idxIni[procID], idxEnd[procID]):
        yield k


# TODO def taskDispatcherMPIProcess(nTasks, nProcs, worker, pendingPos=None):

#     if pendingPos is None:
#         pendingPos = np.arange(nTasks)

#     nPend = len(pendingPos)


#     jobParts = np.ones(nProcs + 1, 'i') * int(nPend/nProcs)
#     jobParts[:np.mod(nPend, nProcs)] += 1
#     idxEndT = jobParts.cumsum()[:nProcs].astype(int)
#     idxIniT = (idxEndT - jobParts[:nProcs]).astype(int)
#     jobIdxEnd = np.cumsum(nJobsV)
#     jobIdxIni = jobIdxEnd - nJobsV
