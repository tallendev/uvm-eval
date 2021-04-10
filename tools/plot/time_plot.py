#!/usr/bin/python3
#import experimental_models
#from experimental_models import opts, metrics
import sys
import argparse
import sys
import os
import argparse
from os.path import basename, splitext, dirname

import itertools, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import Fault
from Fault import Experiment


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 16})

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Full path to CSV file')
    args = parser.parse_args()
    c = args.csv
    m = "*"

    if not ".txt" in args.csv:
        print("Suspicious input file with no/wrong extension:", args.csv)
        exit(1)
    
    e = Experiment(c)
    
    ranges, faults, batches, num_batches = e.get_raw_data()
    e.print_info()

    matplotlib.rcParams['agg.path.chunksize'] = 10000
    fig = plt.figure()
    #fig = plt.gcf()
    ax = fig.add_subplot(1,1,1)
    #ax2 = ax.twiny()
    
    read=[]
    rx = []
    write=[]
    wx = []
    prefetch = []
    px = []
    itot = 0
    for batch in batches:
        batch = sorted(batch, key=lambda f: f.timestamp) 
        for i,f in enumerate(batch):
            if f.access_type == "r":
                read.append(f.fault_address)
                rx.append(f.timestamp)
            elif f.access_type == "w":
                write.append(f.fault_address)
                wx.append(f.timestamp)
            elif f.access_type == "p":
                prefetch.append(f.fault_address)
                px.append(f.timestamp)
            else:
                print("unaccounted for access type:", faults.access_type)
        itot += len(batch)
    
    #for i,f in enumerate(sorted(faults, key=lambda f: f.timestamp)):
    #    if f.access_type == "r":
    #        read.append(f.fault_address)
    #        rx.append(i + itot)
    #    elif f.access_type == "w":
    #        write.append(f.fault_address)
    #        wx.append(i + itot)
    #    else:
    #        print("unaccounted for access type:", faults.access_type)
    
    base_x = min(faults, key=lambda f: f.timestamp).timestamp
    rx = [x - base_x for x in rx]
    wx = [x - base_x for x in wx]
    px = [x - base_x for x in px]

    base_addr = min(ranges)
    print("base_addr:", base_addr)
    #base_addr = min(faults, key=lambda f: f.fault_address).fault_address
    read = [(f - base_addr)//4096 for f in read]
    write = [(f - base_addr)//4096 for f in write]
    prefetch = [(f - base_addr)//4096 for f in prefetch]
    ranges = [(r - base_addr)//4096 for r in ranges]
    print(read)
    print (write)


    bl = [0] + [len(batch) for batch in batches]
    for i,b in enumerate(bl):
        if i == 0:
            continue
        bl[i] += bl[i-1]
    del bl[0]

    #print(batches)
    print("batch lengths:", bl)
    
    #print(ranges)
    ylim1 = min(read + write + prefetch)
    ylim2 = max(read + write + prefetch)
    #ax.vlines(bl, min(ranges), ylim2, label="batch")
    print(ranges)
    print(len(faults))
    ax.plot(rx, read, 'b' + m, markersize=3, label="read")
    ax.plot(wx, write, 'r' + m, markersize=3, label="write")
    if (len(px) > 0):
        ax.plot(px, prefetch, 'g' + m, markersize=3, label="prefetch")
    ax.hlines(ranges, 0, max(rx + wx + px), label="malloc", linestyles="dotted", color="k", linewidth=1)

    #newfaults = [f.fault_address for f in sorted(faults, key=lambda f: f.timestamp)]
    #newx = [f.timestamp for f in sorted(faults, key=lambda f: f.timestamp)]
    #newx = [int(f - min(newx)) for f in newx]
    #ax2.plot(newx, newfaults, 'g'+m, markersize=1, label="time_order")
    
    #print("newx", newx)

    #fig.set_size_inches(16, 10)

    ax.set_xlabel('Fault Buffer Timestamp')
    ax.set_ylabel('Fault Fault Index')
    psize = dirname(args.csv).split("_")[1]
    figname = splitext(basename(args.csv))[0].split("_")[0] + "-" + psize +  "-time.png"
    
    #ylabels = map(lambda t: '0x%013X' % int(t), ax.get_yticks())
    #ax.set_yticklabels(ylabels)

    ax.legend(loc="upper left")
    #h1, l1 = ax.get_legend_handles_labels()
    #h2, l2 = ax2.get_legend_handles_labels()
    #ax.legend(h1+h2, l1+l2, loc="upper left")
    
    #ax2.set_xlabel("Time (NS)");

    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)

if __name__== "__main__":
  main()

