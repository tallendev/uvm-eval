#!/usr/bin/python3
import sys
import argparse
import os

from os.path import basename, splitext, dirname
import re

import itertools, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

import numpy as np

import itertools

from natsort import humansorted

from Fault import Experiment

from multiprocessing import Pool

NUM_THREADS=12

def parse_bsize(t):
    bsize, lis = t
    x = []
    y = []
    for csv in humansorted(lis):
        e = Experiment(csv)
        size = None
        total_dups = None
        if "pf" in csv:
            size = basename(dirname(csv)).split("_")[2]
        else:
            size = basename(dirname(csv)).split("_")[1]
        size = size
        x.append(size)
        y.append(np.mean([len(b) for b in e.batches]))
    return (x, y, bsize)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs="+", type=str, help='Full path to CSV file')
    parser.add_argument('-o', required=True, type=str, default="", help='output filename')
    args = parser.parse_args()
    c = args.csv
    m = "*"
    
    

    for csv in args.csv:
        if ".txt" not in csv:
            print("Suspicious input file:", csv)
            exit(1)

    
    matplotlib.rcParams['agg.path.chunksize'] = 10000
    fig = plt.figure()
    
    bsize_labels = []
    bsize_xs = []
    bsize_ys = []

    bsize_dicts = []
    
    csvs = humansorted(args.csv)

    bsize_dict = dict()
    for csv in csvs:
        bsize = int(basename(dirname(csv)).split("_")[-1])
        if bsize not in bsize_dict:
            bsize_dict[bsize] = []

        bsize_dict[bsize].append(csv)

    with Pool(processes=NUM_THREADS) as pool:
        #x, y, bsize = pool.map(parse_bsize, bsize_dict.items())
        ret = pool.map(parse_bsize, bsize_dict.items())
        for x, y, bsize in ret:
            bsize_xs.append(x)
            bsize_ys.append(y)
            bsize_labels.append(bsize)
    #for bsize, lis in bsize_dict.items():
        #x = []
        #y = []
        #bsize_labels.append(bsize)
        #for csv in lis:
        #    e = Experiment(csv)
        #    size = None
        #    total_dups = None
        #    if "pf" in csv:
        #        size = basename(dirname(csv)).split("_")[2]
        #        total_dups = e.print_duplicate_faults_64k()
        #    else:
        #        size = basename(dirname(csv)).split("_")[1]
        #        total_dups = e.print_duplicate_faults_4k()
        #    size = int(size)
        #    x.append(size)
        #    y.append(total_dups)
        
        #bsize_xs.append(x)
        #bsize_ys.append(y)
        

    evenly_spaced_interval = np.linspace(0, 1, len(bsize_xs))
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]

    for x, y, l, c in zip(bsize_xs, bsize_ys, bsize_labels, colors):
        plt.plot(x, y, "b+", label=l, marker=".", color=c, linestyle="-")


    plt.xlabel("Problem Size")
    plt.ylabel("avg batch size")
    plt.legend()

    figname = args.o
    if ".png" not in figname:
        figname += ".png"

    
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)

if __name__== "__main__":
  main()

