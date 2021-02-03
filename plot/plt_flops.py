import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import sys

if len(sys.argv) < 2:
    print("Usage ./plot_timing.py [out_tag] [data file].\n")
    sys.exit(1)

COLS=19
label = ["Threads",       "RingD",       "p",           "log2q",       "N", 
         "bs",            "Batch #",     "Exact Flops", "Rel. Err.",   "Rot KeyGen",
         "Sum KeyGen",    "Mult KeyGen", "KeyGen",      "Enc", "Eval",
         "Merge",         "Dec",         "CPU Eval"]

tag=sys.argv[1]
fn=sys.argv[2]
raw = np.loadtxt(fn, delimiter=" ", usecols=np.arange(1, COLS))

## flops...
def plt_flops():

    plt.figure()
    
    nbatch = [1024, 512, 256, 128, 64]
    style = ['-', '--', '-.', ':',':']
    for b,ls in zip(nbatch, style):
	d  = (raw[raw[:,6]==b])
	core       = d[:,0]
	exact_flop = d[:,7]
	hetime     = d[:,14]+d[:,15]
	cputime    = d[:,17]
        plt.plot(core, exact_flop/hetime*1e-6/core,  'r', linestyle=ls, marker='o', label="{} (nb={})".format(tag, b))
        plt.plot(core, exact_flop/cputime*1e-6/core, 'b', linestyle=ls, marker='o')
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=10)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,y: '{:.0f}'.format(x)))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,y: '{}'.format(x)))
    plt.xlabel("Core     (tag={})".format(tag))
    plt.ylabel("MFlops / core")
    plt.grid()
    plt.savefig("flops_{}.png".format(tag), bbox_inches='tight')

def plt_err():

    plt.figure()
    
    core = [32,16,8,4,2]
    for c in core:
        d  = (raw[raw[:,0]==c])
        batch = d[:,6]
        
        plt.plot(batch, np.abs(d[:,8]), marker='o', label="Rel. Err")

    plt.xscale('log', basex=2)
    plt.yscale('log', basey=10)
    plt.xlabel("Batch #  (bs=8192, tag={})".format(tag))
    plt.ylabel("Rel. Err")
    plt.savefig("err_{}.png".format(tag), bbox_inches='tight')

plt_flops()
plt_err()
