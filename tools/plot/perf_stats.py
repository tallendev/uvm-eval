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

#NUM_THREADS=12


appname = None

class ExperimentStats:
    # TODO should size, bsize be standardized into Fault.py?
    def __init__(self, e, size, bsize):
        self.avg_batches = np.mean([len(b) for b in e.batches])
        self.num_batches = len(e.batches)
        self.num_faults = len(e.faults)
        self.num_4k_dups = sum(e.get_duplicate_faults_4k())
        self.num_64k_dups = sum(e.get_duplicate_faults_64k())
        self.size = size
        self.bsize = bsize
        self.pf = e.pf

def parse_bsize(t):
    bsize, lis = t
    es = []
    for csv in humansorted(lis):
        e = Experiment(csv)
        size = None
        total_dups = None
        if "pf" in csv:
            size = basename(dirname(csv)).split("_")[2]
        else:
            size = basename(dirname(csv)).split("_")[1]
        es.append(ExperimentStats(e, size, bsize))
    return (es, bsize)


def plot_duplicates(es, bsize_labels):

    evenly_spaced_interval = np.linspace(0, 1, len(es))
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]

    matplotlib.rcParams['agg.path.chunksize'] = 10000
    fig = plt.figure()

    for e, l, c in zip(es, bsize_labels, colors):
        x = [exp.size for exp in e]
        y = None
        if e[0].pf:
            y = [exp.num_64k_dups for exp in e]
        else:
            y = [exp.num_4k_dups for exp in e]
        plt.plot(x, y, "b+", label=l, marker=".", color=c, linestyle="-")

    plt.xlabel("Problem Size")
    plt.ylabel("# Duplicate Faults")
    plt.legend()

    figname = appname + "-dups.png"
    
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)

def plot_avg_batch(es, bsize_labels):

    evenly_spaced_interval = np.linspace(0, 1, len(es))
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]

    matplotlib.rcParams['agg.path.chunksize'] = 10000
    fig = plt.figure()

    for e, l, c in zip(es, bsize_labels, colors):
        x = [exp.size for exp in e]
        y = [exp.avg_batches for exp in e]
        plt.plot(x, y, "b+", label=l, marker=".", color=c, linestyle="-")

    plt.xlabel("Problem Size")
    plt.ylabel("Avg Batch Size")
    plt.legend()

    figname = appname + "-avgbatch.png"
    
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)


def plot_faults(es, bsize_labels):

    evenly_spaced_interval = np.linspace(0, 1, len(es))
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]

    matplotlib.rcParams['agg.path.chunksize'] = 10000
    fig = plt.figure()

    for e, l, c in zip(es, bsize_labels, colors):
        x = [exp.size for exp in e]
        y = [exp.num_faults for exp in e]
        plt.plot(x, y, "b+", label=l, marker=".", color=c, linestyle="-")

    plt.xlabel("Problem Size")
    plt.ylabel("# Faults")
    plt.legend()

    figname = appname + "-faults.png"
    
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)

def plot_batches(es, bsize_labels):

    evenly_spaced_interval = np.linspace(0, 1, len(es))
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]

    matplotlib.rcParams['agg.path.chunksize'] = 10000
    fig = plt.figure()

    for e, l, c in zip(es, bsize_labels, colors):
        x = [exp.size for exp in e]
        y = [exp.num_batches for exp in e]
        plt.plot(x, y, "b+", label=l, marker=".", color=c, linestyle="-")

    plt.xlabel("Problem Size")
    plt.ylabel("# Batches")
    plt.legend()

    figname = appname + "-batches.png"
    
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)


def main():

    global appname
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs="+", type=str, help='Full path to CSV file')
    parser.add_argument('-o', required=True, type=str, default="", help='output filename')
    args = parser.parse_args()
    c = args.csv
    m = "*"
    appname = args.o
    
    for csv in args.csv:
        if ".txt" not in csv:
            print("Suspicious input file:", csv)
            exit(1)

    
    bsize_labels = []
    expstats = []

    bsize_dicts = []
    
    csvs = humansorted(args.csv)

    bsize_dict = dict()
    for csv in csvs:
        bsize = int(basename(dirname(csv)).split("_")[-1])
        if bsize not in bsize_dict:
            bsize_dict[bsize] = []

        bsize_dict[bsize].append(csv)

    with Pool() as pool:
        #x, y, bsize = pool.map(parse_bsize, bsize_dict.items())
        ret = pool.map(parse_bsize, bsize_dict.items())
        for e, bsize in ret:
            expstats.append(e)
            bsize_labels.append(bsize)

    plot_batches(expstats, bsize_labels)
    plot_faults(expstats, bsize_labels)
    plot_duplicates(expstats, bsize_labels)
    plot_avg_batch(expstats, bsize_labels)


if __name__== "__main__":
  main()

