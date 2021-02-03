import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import sys

if len(sys.argv) < 3:
    print("Usage ./plot_timing.py [tag] [data file].\n")
    sys.exit(1)

COLS=19
label = ["Threads",       "RingD",       "p",           "log2q",       "N", 
         "bs",            "Batch #",     "Exact Flops", "Rel. Err.",   "Rot KeyGen",
         "Sum KeyGen",    "Mult KeyGen", "KeyGen",      "Enc", "Eval",
         "Merge",         "Dec",         "CPU Eval"]

tag=sys.argv[1]
fn=sys.argv[2]
raw = np.loadtxt(fn, delimiter=" ", usecols=np.arange(1, COLS))
print("File {} loaded...".format(fn))

## time vs batch # w/ fixed core...
def plt_timing(core):
    d  = (raw[raw[:,0]==core])  ## collect raw with thread = xx

    x = d[:,6]

    plt.figure()
    plt.plot(x, d[:,10], marker='o', label=label[10])
    plt.plot(x, d[:,11], marker='o', label=label[11])
    plt.plot(x, d[:,9]/x, marker='o', label='{} / batch'.format(label[9]))
    plt.plot(x, d[:,14]/x, marker='o', label='{} / batch'.format(label[14]))
    plt.plot(x, d[:,15]/x, marker='o', label='{} / batch'.format(label[15]))
    plt.plot(x, d[:,16], marker='o', label=label[16])
    plt.plot(x, d[:,13]/x, marker='o', label='{} / batch'.format(label[13]))
    plt.plot(x, d[:,17]/x, marker='o', label='{} / batch'.format(label[17]))
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=10)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,y: '{:.0f}'.format(x)))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,y: '{}'.format(x)))
    plt.xlabel("Batch #  (bs=8192, core={} tag={})".format(core, tag))
    plt.ylabel("Seconds")
    plt.grid()
    plt.savefig("{}_timing_c{}.png".format(tag, core), bbox_inches='tight')

#plt_timing(252)
#plt_timing(192)
#plt_timing(128)
#plt_timing(64)
plt_timing(32)
plt_timing(16)
plt_timing(8)
plt_timing(4)
plt_timing(2)
plt_timing(1)
