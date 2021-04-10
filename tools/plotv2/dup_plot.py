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

from Batch import Experiment


def main():
    
    plt.rcParams.update({"font.family": "sans-serif", 
                        "font.sans-serif": ["Helvetica"], 
                        "font.size": 20})

    parser = argparse.ArgumentParser()
    parser.add_argument('csvs', nargs="+", type=str, help='Full path to CSV file')
    parser.add_argument('-o', required=True, type=str, default="", help='output filename')
    args = parser.parse_args()
    m = "*"
    
    #input_re = re.compile(".*quant-.+\.csv")
    
    matplotlib.rcParams['agg.path.chunksize'] = 10000
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax12 = ax.twinx()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    
    bsizes = set()
    bsize_xs = []
    bsize_ys = []
    bsize_ys2 = []
    bsize_ys3 = []

    bsize_dicts = dict()

    for csv in humansorted(args.csvs):
        #label = basename(splitext(csv)[0])[6:]
        size = None
        print("csv:", csv)
        if "pf" in basename(dirname(csv)):
            size = int(basename(dirname(csv)).split("_")[2])
        else:
            size = int(basename(dirname(csv)).split("_")[1])

        bsize = int(basename(dirname(csv)).split("_")[-1])
        if bsize > 1024:
            continue
        bsizes.add(bsize)
        if bsize not in bsize_dicts:
            bsize_dicts[bsize] = dict()
        
        e = Experiment(csv)
        batches = e.len_batches()
        total = sum(e.get_total_faults())
        dups = sum(e.get_duplicates())
        #dups = sum([c - a for c, a in zip(e.get_total_faults(), e.get_no_duplicates())])

        #dups = sum(e.get_duplicates())
        #total = sum(e.get_total_faults())
        #batches = len(e.get_duplicates())
        
        assert(size not in bsize_dicts[bsize])
        #bsize_dicts[bsize][size] = (dups, batches, total)
        bsize_dicts[bsize][size] = (dups / total, batches, total)
        print(f"{args.o} faults/batch:", dups / batches)
        

    for size, d in sorted(bsize_dicts.items(), key=lambda t: t[0]):
        print(d)
        xs, ys = zip(*humansorted(d.items(), key=lambda t: t[0]))
        ys1, ys2, ys3 = zip(*ys)
        bsize_xs.append(xs)
        bsize_ys.append(ys1)
        bsize_ys2.append(ys2)
        bsize_ys3.append(ys3)

    

    evenly_spaced_interval = np.linspace(0, 1, len(bsize_xs))
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]
    #ax2 = ax.twinx()

    for x, y, y2, y3, l, c in zip(bsize_xs, bsize_ys, bsize_ys2, bsize_ys3, sorted(bsizes), colors):
        ax.plot(x, y, "b+", label=l, marker=".", color=c, linestyle="-")
        ax2.plot(x, y2, "b+", label=l, marker="*", color=c, linestyle="--")
        ax3.plot(x, y3, "b+", label=l, marker="*", color=c, linestyle="--")


    ax.set_xlabel("Problem Size")
    ax.set_ylabel("% of Duplicates/Batch")
    ax.set_ylim(0,1)
    ax.legend()

    figname = args.o
    figname2 = "batches-" + args.o
    figname3 = "faults-" + args.o
    if ".png" not in figname:
        figname += ".png"
        figname2 += "-dup.png"

    
    #plt.tight_layout()
    fig.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)
    

    ax2.set_xlabel("Problem Size")
    ax2.legend()
    ax2.set_ylabel("# Batches")

    #plt.tight_layout()
    fig2.tight_layout()
    fig2.savefig(figname2, dpi=500)
    plt.close(fig2)

    ax3.set_xlabel("Problem Size")
    ax3.legend()
    ax3.set_ylabel("Total Faults")

    #plt.tight_layout()
    fig3.tight_layout()
    fig3.savefig(figname3, dpi=500)
    plt.close(fig3)


if __name__== "__main__":
  main()

