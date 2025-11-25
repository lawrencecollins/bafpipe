import sqlite3
import numpy as np
from ctypes import *
import sys
import os
from deconvolawrence import baf2sql
import matplotlib.pyplot as plt


class BafSpectrum():

    """Contains data from a baf file
    """
    def __init__(self):
        self.spec_vals = {'profile_mz':[], 'profile_int':[],
                          'line_mz':[], 'line_int':[]}

    def open_baf_tic(self, path):

        # add checks for .d

        self.path = path
        if sys.version_info.major == 2:
        # note: assuming a european Windows here...
            self.path = unicode(path, 'cp1252')

        self.baf_fn = os.path.join(path, "analysis.baf")
        sqlite_fn = baf2sql.getSQLiteCacheFilename(self.baf_fn)
        self.conn = sqlite3.connect(sqlite_fn)

        # --- Count spectra
        q = self.conn.execute("SELECT COUNT(*) FROM Spectra "
                        "WHERE LineMzId NOT NULL AND ProfileMzId NOT NULL")
        row = q.fetchone()
        N = row[0]
        print("Specified BAF has {} spectra with line and profile data.".format(N))

        # --- Plot TIC and BPC over MS^1 spectra
        q = self.conn.execute("SELECT Rt, SumIntensity, MaxIntensity FROM Spectra s "
                        "JOIN AcquisitionKeys ak ON s.AcquisitionKey = ak.Id "
                        "WHERE ak.MsLevel = 0 "
                        "ORDER BY s.ROWID")
        self.data = [ row for row in q ]
        self.rt = [ row[0] for row in self.data ]
        self.tic = [ row[1] for row in self.data ]
        self.bpc = [ row[2] for row in self.data ]

        return self.rt, self.tic, self.bpc
    def rt_iter(self, baf_fn=None, conn = None, rt = None, scanstart = None, scanend = None):
        if conn == None:
            conn = self.conn
        if baf_fn is None:
            baf_fn = self.baf_fn
        if rt == None:
            rt = self.rt
        if scanstart is not None and scanend is not None:
            rt = rt[scanstart:scanend]
        elif scanstart is not None:
            rt = rt[scanstart:]

        alldata = []
        scans = []
        for n, i in enumerate(rt):
            q = conn.execute("SELECT LineMzId, LineIntensityId, ProfileMzId, ProfileIntensityId FROM Spectra "
                        "WHERE ABS(Rt - {}) < 1e-8".format(i))
            row = q.fetchone()

            bs = baf2sql.BinaryStorage(baf_fn)

            if not all(row) == False: # check for None values

                bs = baf2sql.BinaryStorage(baf_fn)

                profile_mz = np.array(bs.readArrayDouble(row[2]))
                profile_int = np.array(bs.readArrayDouble(row[3]))

                scan = np.transpose([profile_mz, profile_int])
                alldata.append(scan)
                scans.append(n)
        return alldata, scans







    def extract_scans(self, scanstart=None, scanend=None, rt = None,
                      conn = None, baf_fn = None, mean=True):
        if rt is None:
            rt = self.rt
        if conn is None:
            conn = self.conn
        if baf_fn is None:
            baf_fn = self.baf_fn
        if scanstart is not None:
            rt = rt[scanstart:scanend]
        # if scanend is not None:
        #     rt = rt[:scanend]

        for i in rt:
            q = conn.execute("SELECT LineMzId, LineIntensityId, ProfileMzId, ProfileIntensityId FROM Spectra "
                        "WHERE ABS(Rt - {}) < 1e-8".format(i))
            row = q.fetchone()

            bs = baf2sql.BinaryStorage(baf_fn)

            if not all(row) == False: # check for None values

                bs = baf2sql.BinaryStorage(baf_fn)

                profile_mz = bs.readArrayDouble(row[2])
                profile_int = bs.readArrayDouble(row[3])

                self.spec_vals['profile_mz'].append(profile_mz)
                self.spec_vals['profile_int'].append(profile_int)


                line_mz = bs.readArrayDouble(row[0])
                line_int = bs.readArrayDouble(row[1])

                self.spec_vals['line_mz'].append(line_mz)
                self.spec_vals['line_int'].append(line_int)

        # convert spectra into arrays and average
        self.profile_mz = np.array(self.spec_vals['profile_mz']).mean(axis=0)
        self.profile_int = np.array(self.spec_vals['profile_int']).mean(axis=0)
        # self.line_mz = np.array(self.spec_vals['line_mz']).mean(axis=0)
        # self.line_int = np.array(self.spec_vals['line_int']).mean(axis=0)

        # transpose profile spectra
        self.data2 = np.transpose([self.profile_mz, self.profile_int])

        return self.data2

    def export_scans_from_file(self, path, scanstart=None, scanend=None, name = None):

        self.scanstart = scanstart
        self.scanend = scanend
        if name is None:
            directory, name = os.path.split(path)
        self.name = name

        self.open_baf_tic(path)
        data = self.extract_scans(scanstart=scanstart, scanend=scanend)
        self.data = data
        return self.name, data


    def plot_tic(self, rt=None, tic=None, name = None, show_scans = False):
        if rt == None:
            rt = self.rt
        if tic == None:
            tic = self.tic
        if name == None:
            name = self.name
        plt.plot(rt, tic)
        if show_scans == True:
            plt.axvspan(rt[self.scanstart], rt[self.scanend])
        plt.title(name)
        plt.xlabel("Time")
        plt.show()
