#!/usr/bin/python3
# Use this when converting fault addresses reported by nvidia-uvm
# example:
# assuming you have run dmesg_to_txt first - 
# python3 txt_to_npy_bulk.sh ../cublas/*.txt

import numpy as np
import sys
sys.path.append('../../cat/')
from os.path import basename, splitext
import opts
import multiprocessing as mp
import math

def txt_to_binary(in_filename: str):
    """takes a textfile produced by scraping dmesg and converts it to a 
      binary file that's readable by numpy's tofile()
   """
    out_filename = splitext(in_filename)[0] + ".npy"
    dbg_filename = splitext(in_filename)[0] + ".dbg"
    print (out_filename)
    addr = []
    cli = []
    gpc = []
    #with open(in_filename, "r") as fh:
    #    for line in fh:
    #            addr.append(int(line, 16))
    #addr = opts.load_np_array(in_filename)
    addr = []
    
    ogranges = []
    ranges = [0]
    batch = False
    a = []
    with open(in_filename, 'r') as of:
        for line in of:
            #print (line)
            kind, ad = line.strip().split(',')
            if (kind == 'f'):
                a.append(int(ad, 16) >> 12)
            elif kind == 'p':
                # check there's actually a value here...
                val = int(ad, 16)
                pass
                #assert (batch)
            elif kind == 'b':
                #print(kind, ",", ad)
                assert (batch)
                batch = False
                if a != []:
                    addr.append(a)
                    a = []
            elif (kind == 's'):
                #print(kind, ",", ad)
                assert (not batch)
                batch = True
            elif len(kind) > 1:
                ogranges.append(int(ad))
                ranges.append(int(ad) + ranges[-1])
                continue
    if len(a) != 0:
        addr.append(a)
    #addr = [len(ranges)] + ranges + addr
    addr = [np.asarray(a, dtype=np.int32) for a in addr]
    addr = np.asarray(addr)
    #addr = np.asarray(addr) #, dtype=np.int64)
    #print (type(addr))
    #print (type(addr[0]))
    #print (type(addr[0][0]))

    def red(val, threshs, offsets):
        #ov = val
        #print (val)
        for thresh, off in zip(threshs, offsets):
            if val > thresh:
                val -= off
        return val
    
    tranges = [i / 4096 for i in ranges]
    tranges = tranges[1:-1]
    offsets = []
    #addr = addr >> 12
    m = min([a.min() for a in addr])
    for i, a in enumerate(addr):
        a -= m
    #addr = addr.astype(np.int64) # converto to 64 bit
    
    print ("ranges:", ranges)
    #print ("Addr min, max, len:", addr.min(), addr.max(), len(addr))
    fulladdrs = {i for a in addr for i in a}
    for r in tranges:
        r = int(math.ceil(r))
        offsets.append(min(filter(lambda x: x > r-1, fulladdrs), default=r) - r )
        fulladdrs = {red(i, [r], [offsets[-1]]) for i in fulladdrs}
    print ("offsets:", offsets)

    for a in addr:
        for j, v in enumerate(a):
            a[j] = red(v, tranges, offsets) 
    #addr = np.asarray([red(i, tranges, offsets) for i in addr], dtype=np.int64)
    #addr = np.asarray([red(i, tranges, offsets) for i in addr], dtype=np.int64)
    #print ("Addr min, max, len:", addr.min(), addr.max(), len(addr))

    #addr = np.array(addr)
    #ranges = ranges[1:]
    print ("outgoing ranges:", ogranges)
    #out = np.concatenate((np.asarray([len(ogranges)] + ogranges, dtype=np.int64), addr))
    #for i in range(max(addr)):
    #    if i not in addr:
    #        print ("######MISSING:", i)

    temp = [[len(ogranges)] + ogranges] #+ [a for a in addr]
    for a in addr:
        #print (type(a))
        temp.append(a)
    #out = np.vstack(temp)
    out = temp
    print ("saving", out_filename)
    np.save(out_filename, out, allow_pickle=True)

    np.set_printoptions(threshold=sys.maxsize)
    print ("building", dbg_filename)
    #outstr = "\n".join([k o.astype(str) for o in out])
    #outstr = np.array2string(out, separator="\n")
    #outstr = "\n".join(out)
    
    #print ("saving", dbg_filename)
    #with open (dbg_filename, "w+") as of:
    #    of.write(outstr[:-1])
    
    #a = np.load(out_filename)
    #print(a)
    
    text = opts.load_np_array(out_filename)
    newf = splitext(out_filename)[0] + ".npz"
    opts.save_np_array(newf, text)

if __name__ == "__main__":
    p = mp.Pool(24)
    p.map(txt_to_binary, sys.argv[1:])
    #txt_to_binary(sys.argv[1])

    #for f in sys.argv[1:]:
    #    txt_to_binary(f)
