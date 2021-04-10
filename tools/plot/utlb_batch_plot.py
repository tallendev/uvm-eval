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

import numpy as np

import itertools

import Fault
from Fault import Experiment


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
    ax2 = ax.twiny()
    
    read=[]
    rx = []
    write=[]
    wx = []
    for i,f in enumerate(faults):
        if f.access_type == "r":
            read.append(f.fault_address)
            rx.append(i)
        elif f.access_type == "w":
            write.append(f.fault_address)
            wx.append(i)
        else:
            print("unaccounted for access type:", faults.access_type)


    bl = [0] + [len(batch) for batch in batches]
    for i,b in enumerate(bl):
        if i == 0:
            continue
        bl[i] += bl[i-1]
    del bl[0]

    #print(batches)
    print("batch lengths:", bl)
    
    sm_ids = sorted(e.count_utlb_client_pairs().keys())
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(sm_ids)))
    
    id_f_map = {sm_id:[] for sm_id in sm_ids}
    id_fx_map = {sm_id:[] for sm_id in sm_ids}
    x = 1
    for fault in faults:
        mapping = (fault.utlb_id, fault.client_id)
        id_f_map[mapping].append(fault.fault_address)
        id_fx_map[mapping].append(x)
        x += 1


    bl = bl#[0:20]
    cap = bl[-1]

    print("plotting vlines")
    ylim2 = max(itertools.chain.from_iterable(id_f_map.values()))
    ax.vlines(bl, min(ranges), ylim2, label="batch", linewidth=.1)
    print("plotting hlines")
    #ax.hlines(ranges, 0, len(faults), label="cudaMallocManaged()", linestyles="dotted")
    ax.hlines(ranges, 0, cap, label="cudaMallocManaged()", linestyles="dotted")
    
    total_xs = []
    for i, (sm_id, color) in enumerate(zip(sm_ids, colors)):
        print("plotting sm", sm_id)
        #ax.scatter(id_fx_map[sm_id], id_f_map[sm_id] , color=color, label=str(sm_id), s=.01)
        x = id_fx_map[sm_id]
        y = id_f_map[sm_id]
        x = [f for f in x if f < cap]
        y = y[0:len(x)]
        total_xs += x

        print("faults len:", len(x))
        
        #if (sm_id == (46,36)):
            #print("x, y for weird SM_ID:", x, ",", y)
            #continue
        ax.plot(x, y, color=color, label=str(sm_id), linestyle="None", markersize=1, marker="*")
        #ax.plot(id_fx_map[sm_id], id_f_map[sm_id] , color=color, label=str(sm_id), linestyle="None", markersize=1)
        #if i == 0:
        #    break



    #STARTO = 1606348807401491424
    STARTO = 1606352856887328736
    ENDO = 1606352856889447392

    
    print ("sorting times lol")
    sorted_faults = sorted(faults, key=lambda f: f.timestamp)
    print ('building time lists')
    newfaults = [f.fault_address for f in sorted_faults]
    newx = [f.timestamp for f in sorted_faults]
    #newx = [int(f - STARTO) for f in newx]
    newx = [int(f - newx[0]) for f in newx]
        
    print("plotting times")
    ax2.plot(newx[0: max(total_xs)], newfaults[0:max(total_xs)], 'g'+m, markersize=1, label="time_order")
    #ax2.plot([0, ENDO - STARTO], [min(newfaults), min(newfaults)], 'b+', markersize=5, linestyle='None', label="kernel-ret")

    temp = newx[0: max(total_xs)]
    temp2 = []
    for i, f in enumerate(temp):
        if i == 0:
            temp2.append(f)
        else:
            if f > temp[i-1] + 1e5:
                print ("temporal batch:", len(temp2))
                temp2 = [f]
            else:
                temp2.append(f)
    print ("temporal batch:", len(temp2))
    
    for batch in batches:
    #for batch in batches[0:20]:
        print ("real batchlen:", len(batch))
    print ("total batches:", len(batches))

    #print([ENDO - STARTO])
    
    #print("newx", newx)

    #fig.set_size_inches(16, 10)

    ax.set_xlabel('Fault Occurence')
    ax.set_ylabel('Fault Fault Index')
    psize = dirname(args.csv).split("_")[1]
    figname = splitext(basename(args.csv))[0].split("_")[0] + "-" + psize +  "-tlb-batch.png"
    
    ylabels = map(lambda t: '0x%013X' % int(t), ax.get_yticks())
    ax.set_yticklabels(ylabels)
    
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper left")
    
    ax2.set_xlabel("Time (NS)");

    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)

if __name__== "__main__":
  main()

