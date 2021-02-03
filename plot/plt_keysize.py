import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import sys

if len(sys.argv) == 1:
    print("Usage ./plot_cmap.py [data file].\n")
    sys.exit(1)

##std::vector<string> lable = {0"Batch", 1:sf, 2:rd, 3:cyclo 4:rou  5:p 6:logq, 
##  7"Context",8"Cipher", 9"PubKey", 10"SecKey", 11"SumKey", 12"RotKey", 13"MultKey"};

def plt_keysize(fname):
    COLS=15
    raw = np.loadtxt(fname, delimiter="\t", usecols=np.arange(1, COLS))
    label = ["Batch",  "SF",     "RingD",   "Cyc",    "RoU", 
             "p",      "log2q",  "Context", "Cipher", "PubKey",
             "SecKey", "SumKey", "RotKey",  "MultKey"]
    d59 = (raw[raw[:,1]==59])
    d  = d59

    plt.figure()
    plt.plot(d[:,0], d[:,12], marker='o', label=label[12])
    plt.plot(d[:,0], d[:,11], marker='o', label=label[11])
    plt.plot(d[:,0], d[:,13], marker='o', label=label[13])
    plt.legend()
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=2)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,y: '{:.0f}'.format(x)))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,y: '{:.0f}'.format(x/1024/1024)))
    plt.xlabel("Batch #  (bs=8192)")
    plt.ylabel("MByte")
    plt.grid()
    plt.savefig("keysize.png")

for fn in sys.argv[1:]:
    print("Plotting {}".format(fn))
    plt_keysize(fn)
