import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import sys

#if len(sys.argv) < 1:
#    print("Usage ./plt.py [data file].\n")
#    sys.exit(1)

def getOld():
    COLS=5
    label = ["Mem",       "NBP",       "#Ciphertext",           "PF"]

    fn="data.csv"
    raw = np.loadtxt(fn, delimiter=" ", usecols=np.arange(0, COLS))
    print("File {} loaded...".format(fn))
    d8  = (raw[raw[:,3]==8])
    d4  = (raw[raw[:,3]==4])
    d2  = (raw[raw[:,3]==2])
    d1  = (raw[raw[:,3]==1])
    return d1,d2,d4,d8
    
def getNew():
    COLS=9
    #2931.812500 NBP: 500 NCtxt: 108 PF: 8 RD: 32768 nMult: 12 Time: 0.093666 0.278537 54.236085 0.011299

    fn="data_new.csv"
    raw = np.loadtxt(fn, delimiter=" ", usecols=np.arange(0, COLS))
    print("File {} loaded...".format(fn))
    d8  = (raw[raw[:,3]==8])
    d4  = (raw[raw[:,3]==4])
    d2  = (raw[raw[:,3]==2])
    d1  = (raw[raw[:,3]==1])
    return d1,d2,d4,d8
    
    
d1,d2,d4,d8=getOld()
n1,n2,n4,n8=getNew()



def plt_mem():

    plt.figure()
    plt.plot(d1[:,1], d1[:,0], 'k--', label="nM=17")
    plt.plot(d2[:,1], d2[:,0], 'r--')
    plt.plot(d4[:,1], d4[:,0], 'g--')
    plt.plot(d8[:,1], d8[:,0], 'b--')

    plt.plot(n1[:,1], n1[:,0]/1024, 'k.-', label="PF=1, nM=12")
    plt.plot(n2[:,1], n2[:,0]/1024, 'r.-', label="PF=2")
    plt.plot(n4[:,1], n4[:,0]/1024, 'g.-', label="PF=4")
    plt.plot(n8[:,1], n8[:,0]/1024, 'b.-', label="PF=8")
    plt.legend()#loc='upper left') #, bbox_to_anchor=(1.0, 1.0))
    #plt.xscale('log', basex=2)
    #plt.yscale('log', basey=2)
    plt.xlabel("NBP")
    plt.ylabel("Mem / GB")
    plt.grid()
    plt.savefig("mem_nbp.png", bbox_inches='tight')

def plt_time():
    plt.figure()
    plt.plot(d1[:,1], d1[:,4], 'k--', label="nM=17")
    plt.plot(d2[:,1], d2[:,4], 'r--')
    plt.plot(d4[:,1], d4[:,4], 'g--')
    plt.plot(d8[:,1], d8[:,4], 'b--')
    plt.plot(n1[:,1], n1[:,8], 'k.-', label="PF=1, nM=12")
    plt.plot(n2[:,1], n2[:,8], 'r.-', label="PF=2")
    plt.plot(n4[:,1], n4[:,8], 'g.-', label="PF=4")
    plt.plot(n8[:,1], n8[:,8], 'b.-', label="PF=8")
    plt.legend()#loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #plt.xscale('log', basex=2)
    #plt.yscale('log', basey=2)
    plt.xlabel("NBP")
    plt.ylabel("Time / sec")
    plt.grid()
    plt.savefig("time_nbp.png", bbox_inches='tight')

plt_mem()
plt_time()