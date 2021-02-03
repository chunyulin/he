import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import sys

COLS=19
label = ["Threads",       "RingD",       "p",           "log2q",       "N", 
         "bs",            "Batch #",     "Exact Flops", "Rel. Err.",   "Rot KeyGen",
         "Sum KeyGen",    "Mult KeyGen", "KeyGen",      "Enc", "Eval",
         "Merge",         "Dec",         "CPU Eval"]

#raw2 = np.loadtxt("timing_bfv.dat",  delimiter=" ", usecols=np.arange(1, COLS))
#raw3 = np.loadtxt("timing_sigmoid.dat",  delimiter=" ", usecols=np.arange(1, COLS))
#raw2 = np.loadtxt("twcc_bfv.dat",  delimiter=" ", usecols=np.arange(1, COLS))
#raw3 = np.loadtxt("twcc_sigmoid.dat",  delimiter=" ", usecols=np.arange(1, COLS))

raw1 = np.loadtxt("timing_ckks.dat", delimiter=" ", usecols=np.arange(1, COLS))
raw2 = np.loadtxt("twcc_ckks.dat", delimiter=" ", usecols=np.arange(1, COLS))

raw1 = np.loadtxt("timing_bfv.dat", delimiter=" ", usecols=np.arange(1, COLS))
raw2 = np.loadtxt("twcc_bfv.dat", delimiter=" ", usecols=np.arange(1, COLS))

raw1 = np.loadtxt("timing_sigmoid.dat", delimiter=" ", usecols=np.arange(1, COLS))
raw2 = np.loadtxt("twcc_sigmoid.dat", delimiter=" ", usecols=np.arange(1, COLS))

def plt_flops():

    plt.figure()

    nbatch = [2048, 1024, 512, 256, 128, 64, 32]
    nbatch = [1024, 512, 256, 128, 64]
    style = ['-', '--', '-.', ':', ':']
    nbatch = [1024]
    style = ['-']
    
    for b,ls in zip(nbatch, style):
	d1  = (raw1[raw1[:,6]==b])
	core1       = d1[:,0]
	exact_flop1 = d1[:,7]
	hetime1     = d1[:,14]+d1[:,15]
	cputime1    = d1[:,17]
        plt.plot(core1, exact_flop1/hetime1*1e-6/core1,  'r', linestyle=ls, marker='o', label="ARM (nb={})".format(b))
        plt.plot(core1, exact_flop1/cputime1*1e-6/core1, 'r-.', marker='x')

	d2  = (raw2[raw2[:,6]==b])
	core2       = d2[:,0]
	exact_flop2 = d2[:,7]
	hetime2    = d2[:,14]+d2[:,15]
	cputime2    = d2[:,17]
        plt.plot(core2, exact_flop2/hetime2*1e-6/core2,  'b', linestyle=ls, marker='o', label="TWCC (nb={})".format(b))
        plt.plot(core2, exact_flop2/cputime2*1e-6/core2, 'b-.', marker='x')

	#d3  = (raw3[raw3[:,6]==b])
	#core3       = d3[:,0]
	#exact_flop3 = d3[:,7]
	#hetime3    = d3[:,14]+d3[:,15]
	#cputime3    = d3[:,17]
        #plt.plot(core3, exact_flop3/hetime3*1e-6/core3,  'g', linestyle=ls, marker='o', label="Sigmoid (nb={})".format(b))
        #plt.plot(core3, exact_flop3/cputime3*1e-6/core3, 'g-.', marker='x')

    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=10)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,y: '{:.0f}'.format(x)))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,y: '{}'.format(x)))
    plt.xlabel("Core")
    plt.ylabel("MFlops / core")
    plt.grid()
    plt.savefig("allflops_sigmoid.png", bbox_inches='tight')

plt_flops()
