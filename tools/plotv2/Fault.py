#!/usr/bin/python3
import sys
import os
import argparse
import collections
from collections import OrderedDict
from os.path import basename, splitext, dirname
from statistics import stdev
import numpy as np
#from iteration_utilities import duplicates



class Experiment:
    # send in csv file stupidly labeled .txt
    def __init__(self, c):
        ranges = [0]
        faults = []
        batch_id = 0 
        batches = []
        batch_times = []
        prefetches = []
        pfbatches = []

        dfaults = []
        dbatches = []
        csv = None
        with open(c, 'r') as csvf:
            csv = csvf.readlines()
        cbatch = []
        cdbatch = []
        pfbatch = []
        for line in csv:
            cols = line.split(',')
            if cols[0] == 'f':
                faults.append(Fault(*cols[1:], batch_id))
                cbatch.append(faults[-1])
            elif cols[0] == 'p':
                prefetches.append(Fault(*cols[1:]))
                pfbatch.append(prefetches[-1])
            elif cols[0] == 'd':
                dfaults.append(Fault(*cols[1:], batch_id, discarded=True))
                cdbatch.append(faults[-1])
            elif cols[0] == 'b':
                batch_times.append(int(cols[1]))
                if int(cols[2]) != 0:
                    print ("nonzero status for batch detected:", cols[2])
                batches.append(cbatch)
                dbatches.append(cdbatch)
                pfbatches.append(pfbatch)
                cbatch = []
                cdbatch = []
                pfbatch = []
                batch_id += 1
            elif cols[0] == 's' or cols[0] == 'e':
            #elif cols[0] == 'p' or cols[0] == 's' or cols[0] == 'e':
                continue
            # hopefully this is a range
            else:
                #print(line)
                #print (cols[0])
                #FIXME if we move to normalized this needs to be swapped
                #ranges.append(int(cols[0], 16) + ranges[-1])
                ranges.append(int(cols[0], 16))
        del ranges[0]
        #print ("ranges:", ranges)
        #print ("len(ranges):", len(ranges))
        self.ranges = ranges
        self.faults = faults
        self.num_batches = batch_id
        self.batches = batches
        self.dbatches = dbatches
        self.dfaults = dfaults
        self.batch_times = batch_times
        self.prefetches = prefetches
        self.pfbatches = pfbatches

        if "pf" in c:
            self.pf = True
        else:
            self.pf = False

    
    def print_batch_stats(self):
        true_bl = [len(batch) for batch in self.batches]
        print("max batch length:", max(true_bl))
        print("min batch length:", min(true_bl))
        print("avg batch length:", np.mean(true_bl))

    def print_num_batches(self):
        print("# batches:", self.num_batches)
    
    def print_num_faults(self):
        print("# faults:", len(self.faults))

    def print_avg_batch_size(self):
        print("avg batch size:", len(self.faults)/self.num_batches)
    
    def print_stddev_batch_size(self):
        batch_sizes = []
        batch_id = 0
        size = 0
        for fault in self.faults:
            if batch_id != int(fault.batch_id):
                batch_sizes.append(size)
                assert(size != 0)
                size = 0
                batch_id += 1
            size += 1
        batch_sizes.append(size)
        assert(size != 0)
        if len(batch_sizes) == 1:
            print("stddev batch size:", batch_sizes[0])
        else:
            print("stddev batch size:", stdev(batch_sizes))
    
    def print_avg_repeat_faults(self):
        print("avg repeat faults", sum([f.num_instances for f in self.faults])/ len(self.faults))

    def print_unique_access_types(self):
        access_types = {f.access_type for f in self.faults}
        print ("access types:", access_types)
    
    def print_unique_mmu_engine_types(self):
        mmu_engine_types = {f.mmu_engine_type for f in self.faults}
        print ("mmu engine types:", mmu_engine_types)
    
    def print_unique_client_types(self):
        client_types = {f.client_type for f in self.faults}
        print ("client types:", client_types)
    
    def print_mmu_engine_ids(self):
        d = self.count_mmu_engine_ids()
        print("mmu engine ids:", len(d), OrderedDict(sorted(d.items(), key=lambda t:t[0])))
    
    def print_client_ids(self):
        ids = self.count_client_ids()
        print("client ids:", len(ids), OrderedDict(sorted(ids.items(), key=lambda t:t[0])))
 
    def print_utlb_ids(self):
        ids = self.count_utlb_ids()
        print("utlb ids:", len(ids), OrderedDict(sorted(ids.items(), key=lambda t:t[0])))

    def print_gpc_ids(self):
        ids = self.count_gpc_ids()
        print("gpc ids:", len(ids), OrderedDict(sorted(ids.items(), key=lambda t:t[0])))
    
    def print_gpc_utlb_pairs(self):
        ids = self.count_gpc_utlb_pairs()
        print("gpc, utlb ids:", len(ids), OrderedDict(sorted(ids.items(), key=lambda t:t[0])))

    def print_utlb_client_pairs(self):
        ids = self.count_utlb_client_pairs()
        print("utlb, client ids:", len(ids), OrderedDict(sorted(ids.items(), key=lambda t:t[0])))
    
    def print_channel_ids(self):
        ids = self.count_channel_ids()
        print("channel ids:", len(ids), OrderedDict(sorted(ids.items(), key=lambda t:t[0])))

    def print_ve_ids(self):
        ids = self.count_ve_ids()
        print("ve ids:", len(ids), OrderedDict(sorted(ids.items(), key=lambda t:t[0])))
    
    def print_fault_types(self):
        ids = self.count_fault_types()
        print("fault types:", len(ids), OrderedDict(sorted(ids.items(), key=lambda t:t[0])))

    def print_fault_per_sm(self):
        ids = self.count_fault_per_sm()
        print("faults per sm:", ids)

    def print_avg_fault_per_sm(self):
        ids = self.count_fault_per_sm()
        print("avg faults per sm:", ids.most_common())
    
    def print_fault_per_utlb(self):
        ids = self.count_fault_per_utlb()
        print("faults per sm:", ids)

    def print_avg_fault_per_utlb(self):
        ids = self.count_fault_per_utlb()

        vals = [v[1]/self.num_batches for v in ids.most_common()]
        print("avg faults per utlb:", vals)
        print("avg of avg faults per utlb:", np.mean(vals))
        print("std of avg faults per utlb:", np.std(vals))
    
    def print_total_fault_per_utlb(self):
        ids = self.count_fault_per_utlb()
        print("total faults per sm:", ids.most_common())

    def print_avg_fault_per_sm_per_batch(self):
        ids = self.count_fault_per_sm_per_batch()
        avgs = [np.mean([v for v in d.values()]) for d in ids]
        #print("avg faults per sm:", avgs)
        print("avg of avg faults per sm:", np.mean(avgs))
        print("max of avg faults per sm:", max(avgs))
        print("min of avg faults per sm:", min(avgs))
    
    def print_avg_fault_per_utlb_per_batch(self):
        batch_counts = self.count_fault_per_utlb_per_batch()
        avgs = [np.mean([v for v in batch.values()]) for batch in batch_counts]
#        print("avg faults per utlb:", avgs)
        print("avg of avg faults per utlb:", np.mean(avgs))
        print("max of avg faults per utlb:", max(avgs))
        print("min of avg faults per utlb:", min(avgs))

    def print_short_batches(self):
        batches = self.batches
        num_faults = []
        times = self.batch_times

        for time, batch in zip(times, batches):
            if time < 1e4:
                pages = set()
                dups = 0
                for fault in batch:
                    addr = fault.fault_address // 4096
                    if addr in pages:
                        dups += 1
                    pages.add(addr)
                print(f"time, batch, dups: {time}, {len(batch)}, {dups}")


    def print_avg_time_per_batch(self):
        print("runtime / #faults", sum(self.batch_times) / len(self.faults))

    def count_avg_vablocks_per_batch(self):
        vablock_batches = []
        for batch in self.batches:
            vablocks = set()
            for fault in batch:
                vablocks.add(fault.fault_address // 2097152)
            vablock_batches.append(vablocks)
        return np.mean([len(batch) for batch in vablock_batches])

    def print_avg_vablocks_per_batch(self):
        print("avg vablocks per batch:", self.count_avg_vablocks_per_batch)

    def get_duplicate_faults(self, granularity):
        duplicate_batches = []
        for batch in self.batches:
            pages = set()
            dups = 0
            for fault in batch:
                addr = fault.fault_address // granularity
                if addr in pages:
                    dups += 1
                pages.add(addr)
            duplicate_batches.append(dups)
            #if dups > 0:
            #    a = [f.fault_address for f in duplicates(batch, key=lambda f: f.fault_address)]
            #    full_list = []
            #    print ("batch")
            #    for f in batch:
            #        if f.fault_address in a:
            #            full_list.append(f)
            #    for f in full_list:
            #        print (hex(f.fault_address) + ",", str(f.client_id) + ", " + str(f.utlb_id))

        return duplicate_batches

    def get_duplicate_faults_4k(self):
        return self.get_duplicate_faults(4096)

    def get_duplicate_faults_64k(self):
        return self.get_duplicate_faults(65536)

    def print_duplicate_faults_4k(self):
        duplicate_batches = self.get_duplicate_faults_4k()
        print("avg 4k dups per batch:", np.mean(duplicate_batches))
        return duplicate_batches

    
    # returns total
    def print_duplicate_faults_64k(self):
        duplicate_batches = self.get_duplicate_faults_64k()
        print("avg 64k dups per batch:", np.mean(duplicate_batches))
        return duplicate_batches

    #self.gpc_id = gpc_id
    #self.channel_id = channel_id
    #self.ve_id = ve_id
    
    def avg_time_batches(self):
        return np.mean(self.batch_times)

    def count_fault_per_utlb_per_batch(self):
        utlb_ids = {f.utlb_id for f in self.faults}
        per_batch_counts = []
        for batch in self.batches:
            batch_ids = {k:0 for k in utlb_ids}
            for f in batch:
               batch_ids[f.utlb_id] += 1
            per_batch_counts.append(batch_ids)

        return per_batch_counts

    def count_fault_per_sm_per_batch(self):
        sm_ids = {(f.utlb_id, f.client_id) for f in self.faults}
        per_batch_counts = []
        for batch in self.batches:
            batch_ids = {k:0 for k in sm_ids}
            for f in batch:
                batch_ids[(f.utlb_id, f.client_id)] += 1
            per_batch_counts.append(batch_ids)

        return per_batch_counts

    def count_fault_per_sm(self):
        c = collections.Counter([(f.utlb_id, f.client_id) for f in self.faults])
        return c
    
    def count_fault_per_utlb(self):
        c = collections.Counter([f.utlb_id for f in self.faults])
        return c
        
    
    def count_fault_types(self):
        c = collections.Counter([f.fault_type for f in self.faults])
        d = {k:v for k, v in c.most_common() if v > 0}
        return d
    
    def count_ve_ids(self):
        c = collections.Counter([f.ve_id for f in self.faults])
        d = {k:v for k, v in c.most_common() if v > 0}
        return d
    
    def count_channel_ids(self):
        c = collections.Counter([f.channel_id for f in self.faults])
        d = {k:v for k, v in c.most_common() if v > 0}
        return d
    
    def count_gpc_ids(self):
        c = collections.Counter([f.gpc_id for f in self.faults])
        d = {k:v for k, v in c.most_common() if v > 0}
        return d

    def count_utlb_ids(self):
        c = collections.Counter([f.utlb_id for f in self.faults])
        d = {k:v for k, v in c.most_common() if v > 0}
        return d

    def count_utlb_client_pairs(self):
        c = collections.Counter([(f.utlb_id, f.client_id) for f in self.faults])
        d = {k:v for k, v in c.most_common() if v > 0}
        return d
    
    def count_gpc_utlb_pairs(self):
        c = collections.Counter([(f.gpc_id, f.utlb_id) for f in self.faults])
        d = {k:v for k, v in c.most_common() if v > 0}
        return d

    def count_mmu_engine_ids(self):
        c = collections.Counter([f.mmu_engine_id for f in self.faults])
        d = {k:v for k, v in c.most_common() if v > 0}
        return d

    def count_client_ids(self):
        c = collections.Counter([f.client_id for f in self.faults])
        d = {k:v for k, v in c.most_common() if v > 0}
        return d

    def get_duplicate_fault_addrs(self):
        c = collections.Counter([f.fault_address for f in self.faults])
        d = {k:v for k, v in c.most_common() if v > 0}
        return d

    def print_duplicate_fault_addrs(self):
        print(self.get_duplicate_fault_addrs())

    def print_info(self):
        self.print_num_batches()
        self.print_num_faults()
        self.print_avg_batch_size()
        self.print_stddev_batch_size()
        self.print_avg_repeat_faults()
        self.print_unique_access_types()
        self.print_unique_mmu_engine_types()
        self.print_unique_client_types()
        #self.print_mmu_engine_ids()
        #self.print_client_ids()
        #self.print_utlb_ids()
        #self.print_gpc_ids()
        #self.print_channel_ids()
        #self.print_ve_ids()
        #self.print_fault_types()
        #self.print_gpc_utlb_pairs()
        #self.print_utlb_client_pairs()
        #self.print_avg_fault_per_sm()
        self.print_batch_stats()
        self.print_total_fault_per_utlb()
        self.print_avg_fault_per_utlb()
        self.print_avg_fault_per_utlb_per_batch()
        self.print_avg_fault_per_sm_per_batch()

        self.print_avg_time_per_batch()
        self.print_avg_vablocks_per_batch()
        self.print_duplicate_faults_4k()
        self.print_duplicate_faults_64k()
        #self.print_duplicate_fault_addrs()

    def get_raw_data(self):
        return self.ranges, self.faults, self.batches, self.num_batches

    

class Fault:

    def __init__(self,fault_address,timestamp=-1,fault_type=-1,access_type=-1,access_type_mask=-1,num_instances=-1,client_type=-1,mmu_engine_type=-1,client_id=-1,mmu_engine_id=-1,utlb_id=-1,gpc_id=-1,channel_id=-1,ve_id="-1", batch_id=-1, discarded=False):
        self.fault_address = int(fault_address,16)
        self.prefetch = timestamp == -1
        self.timestamp = int(timestamp)
        self.fault_type = self.fault_type_translate(fault_type)
        self.access_type = self.access_type_translate(access_type)
        #self.access_type = access_type
        self.access_type_mask = access_type_mask
        self.num_instances = int(num_instances)
        self.client_type = self.client_type_translate(client_type)
        self.mmu_engine_type = self.mmu_engine_type_translate(mmu_engine_type)
        #self.mmu_engine_type = mmu_engine_type
        self.client_id = int(client_id)
        self.mmu_engine_id = int(mmu_engine_id)
        self.utlb_id = int(utlb_id)
        self.gpc_id = int(gpc_id)
        self.channel_id = int(channel_id)
        self.ve_id = int(ve_id.strip())
        self.batch_id = batch_id
        self.discarded = discarded


    # UVM_FAULT_TYPE_INVALID_PDE = 0,                                                                                     
    # UVM_FAULT_TYPE_INVALID_PTE,                                                                                         
    # UVM_FAULT_TYPE_ATOMIC,                                                                                              
    # // WRITE to READ-ONLY                                                                                               
    # UVM_FAULT_TYPE_WRITE,                                                                                               
    # // READ to WRITE-ONLY (ATS)                                                                                         
    # UVM_FAULT_TYPE_READ,                                                                                                
    # // The next values are considered fatal and are not handled by the UVM driver                                       
    # UVM_FAULT_TYPE_FATAL,                                                                                               
    # // Values required for tools                                                                                        
    # UVM_FAULT_TYPE_PDE_SIZE = UVM_FAULT_TYPE_FATAL,                                                                     
    # UVM_FAULT_TYPE_VA_LIMIT_VIOLATION,                                                                                  
    # UVM_FAULT_TYPE_UNBOUND_INST_BLOCK,                                                                                  
    # UVM_FAULT_TYPE_PRIV_VIOLATION,                                                                                      
    # UVM_FAULT_TYPE_PITCH_MASK_VIOLATION,                                                                                
    # UVM_FAULT_TYPE_WORK_CREATION,                                                                                       
    # UVM_FAULT_TYPE_UNSUPPORTED_APERTURE,                                                                                
    # UVM_FAULT_TYPE_COMPRESSION_FAILURE,                                                                                 
    # UVM_FAULT_TYPE_UNSUPPORTED_KIND,                                                                                    
    # UVM_FAULT_TYPE_REGION_VIOLATION,                                                                                    
    # UVM_FAULT_TYPE_POISONED,                                                                                            
    # UVM_FAULT_TYPE_COUNT

    def fault_type_translate(self, fault_type):
        if fault_type == "0":
            return "pde"
        elif fault_type == "1":
            return "pte"
        elif fault_type == "2":
            return "atom"
        elif fault_type == "3":
            return "write"
        elif fault_type == "4":
            return "read"
        # also PDE size???
        elif fault_type == "5":
            return "fatal"
        elif fault_type == "6":
            return "va_limit"
        elif fault_type == "7":
            return "unbound_inst"
        elif fault_type == "8":
            return "priv"
        elif fault_type == "9":
            return "pitch"
        elif fault_type == "10":
            return "work"
        elif fault_type == "11":
            return "aperture"
        elif fault_type == "12":
            return "compress"
        elif fault_type == "13":
            return "kind"
        elif fault_type == "14":
            return "region"
        elif fault_type == "15":
            return "poisoned"
        elif fault_type == "16":
            return "count"
        elif fault_type == "-1" or fault_type == -1:
            return None
        else:
            print("idk this fault type???")
    

    def client_type_translate(self, client_type):
        if client_type == "0":
            return "g"
        if client_type == "1":
            return "h"
        if client_type == "2":
            return "c"
        if client_type == -1:
            return None
        else:
            print("idk this client type???")


    #TODO: these translate funcs would be static but i'm lazy
    # UVM_FAULT_ACCESS_TYPE_PREFETCH = 0,                                                                                 
    # UVM_FAULT_ACCESS_TYPE_READ,                                                                                         
    # UVM_FAULT_ACCESS_TYPE_WRITE,                                                                                        
    # UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK,                                                                                  
    # UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG,                                                                                
    # UVM_FAULT_ACCESS_TYPE_COUNT   
    def access_type_translate(self, access_type):
        if access_type == "0":
            return "p"
        elif access_type == "1":
            return "r"
        elif access_type == "2":
            return "w"
        elif access_type == "3":
            return "aw"
        elif access_type == "4":
            return "as"
        elif access_type == "5":
            return "c"
        elif access_type == "-1" or access_type == -1:
            return None
        else:
            print("idk this access type???")

    # UVM_MMU_ENGINE_TYPE_GRAPHICS = 0,
    # UVM_MMU_ENGINE_TYPE_HOST,
    # UVM_MMU_ENGINE_TYPE_CE,
    # UVM_MMU_ENGINE_TYPE_COUNT,
    def mmu_engine_type_translate(self, mmu_type):
        if mmu_type == "0":
            return "g"
        elif mmu_type == "1":
            return "h"
        elif mmu_type == "2":
            return "ce"
        elif mmu_type == "3":
            return "c"
        elif mmu_type == -1 or mmu_type == "-1":
            return None
        else:
            print ("Idk this mmu_engine_type???")

    def __eq__(self, other):
        return self.fault_address == other.fault_address

    def __hash__(self):
        return hash(self.fault_address)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Full path to CSV file')
    args = parser.parse_args()
    c = args.csv
    e = Experiment(c)
    e.print_info()
    #e.print_short_batches()
    
    #e.print_duplicate_faults_4k()
    
        
    

if __name__ == "__main__":
        main()
