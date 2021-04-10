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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs="+", type=str, help='Full path to CSV file')
    parser.add_argument('-o', required=True, type=str, default="", help='output filename')
    args = parser.parse_args()
    c = args.csv
    m = "*"
    
    input_re = re.compile(".*quant-.+\.csv")
    

    for csv in args.csv:
        if input_re.match(csv) is None:
            print("Suspicious input file:", csv)
            exit(1)
    
    matplotlib.rcParams['agg.path.chunksize'] = 10000
    fig = plt.figure()
    
    bsize_labels = []
    bsize_xs = []
    bsize_ys = []

    bsize_dicts = []

    for csv in humansorted(args.csv):
        label = basename(splitext(csv)[0])[6:]
        bsize_labels.append(label)
        
        bsize_dict = dict()
        with open(csv, 'r') as c:
            while True:
                print (csv)
                #FIXME won't work with string-based prob sizes
                size = c.readline().split(",")[1]
                #size = int(c.readline().split(",")[1])
                alloced = c.readline().split(",")[1]
                perf = float(c.readline().split(",")[1])
                bsize_dict[size] = perf
                #bs_x.append(size)
                #bs_y.append(perf)
                if c.tell() == os.fstat(c.fileno()).st_size:
                    break
        
        bsize_dicts.append(bsize_dict)    
        #bsize_xs.append(bs_x)
        #bsize_ys.append(bs_y)

    for d in bsize_dicts:
        xs, ys = zip(*humansorted(d.items(), key=lambda t: t[0]))
        bsize_xs.append(xs)
        bsize_ys.append(ys)

    

    evenly_spaced_interval = np.linspace(0, 1, len(bsize_xs))
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]

    for x, y, l, c in zip(bsize_xs, bsize_ys, bsize_labels, colors):
        plt.plot(x, y, "b+", label=l, marker=".", color=c, linestyle="-")


    plt.xlabel("Problem Size")
    plt.ylabel("GFLOPS")
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

