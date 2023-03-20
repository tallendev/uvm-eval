#!/usr/bin/env python
import csv
import re
import sys
import os
from os.path import basename, splitext, dirname
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from numpy import std
import numpy as np
import heapq
import itertools

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (10, 20),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         #'text.usetex' : True,
         #'legend.fontsize': 'xx-large'
         }
pylab.rcParams.update(params)

labs = ['dir', 'uvm']

def getprocnum(csv):
    return int(csv.split(".")[1].split("-")[0])

def getProcFiles(csvs, device):
    return [csv for csv in csvs if getprocnum(csv) == device]

def getDevNum(lab):
    # Tesla V100-DGXS-32GB (0)
    return int(lab.split()[2][1])

class Device:
    labs=["H2D", "T2D", "TFD"]
    def __init__(self):
        self.h2d = []
        self.t2d = []
        self.tfd = []


def probSize(f):
    return int(f.split("-")[1].split(".")[0])

def probSizes(files):
    sizes = []
    for f in files:
        sizes.append(probSize(f))
    sizes.sort()
    return sizes


size_map = {
            "b" : 1.024e3 ** -2, 
            "kb" : 1.024e3 ** -1, 
            "mb" : 1, 
            "gb" : 1.024e3 ** 1
            }

def parseAndPlotProc(files, proc):
    numFigs = len(files)
    devices = dict()
    files.sort(key=probSize)
    for i,f in enumerate(files):
        with open(f, 'r') as of:
            for k in devices:
                devices[k].h2d.append(0)
                devices[k].t2d.append(0)
                devices[k].tfd.append(0)
            for line in of:
                if "Unified Memory profiling result:" in line:
                    break
            of.readline()
            for line in of:
                r = line.split(",")
                if len(r) < 7:
                    continue
                device = getDevNum(r[0])
                if device not in devices:
                    devices[device] = Device()
                    for k in range(i+1):
                        devices[device].h2d.append(0)
                        devices[device].t2d.append(0)
                        devices[device].tfd.append(0)
                if "Host To Device" in r[7]:
                    unit = size_map[r[5][-2:].lower()]
                    devices[device].h2d[i] = (float(r[5][:-2]) * unit)
                elif "Transfers to Device" in r[7]:
                    unit = size_map[r[5][-2:].lower()]
                    devices[device].t2d[i] = (float(r[5][:-2]) * unit)
                elif "Transfers from Device" in r[7]:
                    unit = size_map[r[5][-2:].lower()]
                    devices[device].tfd[i] = (float(r[5][:-2]) * unit)
    print(devices)
    print(probSizes(files))

    def devIdx(i, device):
        if i == 0:
            return device.h2d
        if i == 1:
            return device.t2d
        if i == 2:
            return device.tfd
    for i in range(len(Device.labs)):
        plt.figure(i + 1)
        plt.subplot(4, 1, proc+1)
        for key in devices:
            print(files)
            print(devIdx(i, devices[key]))
            linewidth=3
            #if key == 0:
                #linewidth=5
            plt.plot(probSizes(files), devIdx(i, devices[key]), alpha=.5, marker="*", linewidth=linewidth, label="Dev" + str(key))
        plt.legend(loc="upper left")
        plt.ylabel(Device.labs[i] + " Transfers (MB)")
        plt.xlabel("Problem Size")
        #plt.yscale('log')
        plt.xscale('log')
        plt.title("Tealeaf " +  Device.labs[i] + " By Size/Device")
        plt.grid(True, axis="y")
        plt.tight_layout()
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Full path to CSV file', nargs='+')

    args = parser.parse_args()
    mdir = ""
    fname = ""
    
    procs = set([getprocnum(csv) for csv in args.csv])
    print(procs)


    for i, proc in enumerate(procs):
        files = getProcFiles(args.csv, proc)
        parseAndPlotProc(files, proc)
    for i in range(len(Device.labs)):
        plt.figure(i + 1)
        plt.subplot(4, 1, proc+1)
        figname = "tea-" + Device.labs[i] + ".png"
        print ("Figure Name:", figname)
        plt.savefig(figname, dpi=500)
        


if __name__ == "__main__":
        main()
