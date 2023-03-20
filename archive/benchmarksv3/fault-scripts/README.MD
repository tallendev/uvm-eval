# About These Scripts
Refer to ../batchd-scripts readme; everything but data format is the same\
We only include this data for "abc" application in SC paper; however, we are releasing all of this for our prior work as well

### Data Format
Each line is a fault:\
"f,%llx,%llu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n"\
with the format:\
fault_address,timestamp,fault_type,access_type,access_type_mask,num_instances,client_type,mmu_engine_type,client_id,mmu_engine_id,utlb_id,gpc_id,channel_id,ve_id, batch_id, discarded=False
s, if it is the start of a batch\
b, batch_id, status=0  indicates the end of a batch\
a p in the above indicates a prefetch-generated fault\
an e would indicate an eviction

fault_address:\
timestamp: time of fault written by mmu in buffer\
fault_type: fault type matching UVM driver, 0-16, pde/pte/atom/write/read/fatal/va_limit/unbound_inst/priv/pitch/work/aperture/compress/kind/region/poisoned/count\
access_type: UVM_FAULT_ACCESS_TYPE_PREFETCH = 0, UVM_FAULT_ACCESS_TYPE_READ, UVM_FAULT_ACCESS_TYPE_WRITE, UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK, UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG,UVM_FAULT_ACCESS_TYPE_COUNT 
access_type_mask: Mask of access_type vals\
num_instances: repeat faults\
client_type: g/h/c, refer to open gpu doc and/or driver doc; it's what entity created the fault\
mmu_engine_type: graphics, host, CE, count\
client_id: basically SM ID if combined with utlb id\
mmu_engine_id: which mmu\
utlb_id: which tlb\
gpc_id: which gpc\
channel_id: which channel it was grabbed from\
ve_id: We tracked this but weren't sure; it's documented in uvm\
batch_id: batch count\
discarded: if we are tracking discarded faults from the buffer; other info will be missing if this is true
