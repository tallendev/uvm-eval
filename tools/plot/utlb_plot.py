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
        
    #print(ranges)
    ax.hlines(ranges, 0, len(faults))

    for i, (sm_id, color) in enumerate(zip(sm_ids, colors)):
        #ax.scatter(id_fx_map[sm_id], id_f_map[sm_id] , color=color, label=str(sm_id), s=.01)
        ax.plot(id_fx_map[sm_id], id_f_map[sm_id] , color=color, label=str(sm_id), linestyle="None", markersize=1, marker=",")
        #if i == 0:
        #    break
    

    #fig.set_size_inches(16, 10)

    #plt.legend()
    ax.set_xlabel('Fault Occurence')
    ax.set_ylabel('Fault Fault Index')
    psize = dirname(args.csv).split("_")[1]
    figname = splitext(basename(args.csv))[0].split("_")[0] + "-" + psize +  "-utlb.png"
    
    ylabels = map(lambda t: '0x%013X' % int(t), ax.get_yticks())
    ax.set_yticklabels(ylabels)
    plt.tight_layout()

    print('saving figure:', figname)
    fig.savefig(figname, dpi=800)
    plt.close(fig)

if __name__== "__main__":
  main()

