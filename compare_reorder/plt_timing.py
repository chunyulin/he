import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import sys

#if len(sys.argv) < 3:
#    print("Usage ./plot_timing.py [tag] [data file].\n")
#    sys.exit(1)

COLS=12
label = ["NB", "KeyG", "RotKeyG", "MultKeyG", 
         "Enc", "Dec", "Compute per batch", "SumMerge", "Wall time", "Raw time", "Err"]

TP = np.loadtxt("tpali.dat", delimiter=" ", usecols=np.arange(1, COLS))
TS = np.loadtxt("tseal.dat", delimiter=" ", usecols=np.arange(1, COLS))
TH = np.loadtxt("theli_159.dat", delimiter=" ", usecols=np.arange(1, COLS))
#print(TP)
#print(TS)


##########################3
plt.figure()
plt.plot(TP[:,0], TP[:,1], 'r' , label="P KeyGen")
plt.plot(TS[:,0], TS[:,1], 'b' , label="S")
plt.plot(TH[:,0], TH[:,1], 'g' , label="H")
plt.plot(TP[:,0], TP[:,2], 'r.', label="P RotKeyG")
plt.plot(TS[:,0], TS[:,2], 'b.', label="S")
plt.plot(TH[:,0], TH[:,2], 'g.' , label="H")
plt.plot(TP[:,0], TP[:,3], 'rx', label="P MultiKeyG")
plt.plot(TS[:,0], TS[:,3], 'bx', label="S")
#plt.plot(TH[:,0], TH[:,3], 'gx' , label="H = 0")
plt.plot(TP[:,0], TP[:,4], 'ro', label="P Enc")
plt.plot(TS[:,0], TS[:,4], 'bo', label="S")
plt.plot(TH[:,0], TH[:,4], 'go' , label="H")
plt.plot(TP[:,0], TP[:,5], 'r--', label="P Dec")
plt.plot(TS[:,0], TS[:,5], 'b--', label="S")
plt.plot(TH[:,0], TH[:,5], 'g--' , label="H")
plt.xlabel("# batch (bs=8192) @ 1-core")
plt.ylabel("Seconds")
plt.xscale('log', basex=2)
plt.yscale('log', basey=10)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,y: '{:.0f}'.format(x)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,y: '{}'.format(x)))
plt.grid()
lg=plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.savefig("keytime.png", bbox_inches='tight', bbox_extra_artists=(lg,))


##########################3
plt.figure()
plt.plot(TP[:,0], TP[:,8]/TP[:,0], 'r', marker='o', label="P Walltime / bench")
plt.plot(TS[:,0], TS[:,8]/TS[:,0], 'b', marker='o', label="S")
plt.plot(TH[:,0], TH[:,8]/TH[:,0], 'g', marker='o', label="H")
plt.plot(TP[:,0], TP[:,6]/TP[:,0], 'r:', label="Compute / bench")
plt.plot(TS[:,0], TS[:,6]/TS[:,0], 'b:')
plt.plot(TH[:,0], TH[:,6]/TH[:,0], 'g:')
plt.plot(TP[:,0], TP[:,7]/TP[:,0], 'r--', marker='x', label="Merging / bench")
plt.plot(TS[:,0], TS[:,7]/TS[:,0], 'b--', marker='x')
plt.plot(TH[:,0], TH[:,7]/TH[:,0], 'g--', marker='x')
plt.plot(TP[:,0], 1000*TP[:,9]/TP[:,0], 'k', label="1000 x Rawtime / bench")

plt.xlabel("# batch (bs=8192) @ 1-core")
plt.ylabel("Seconds")
plt.xscale('log', basex=2)
plt.yscale('log', basey=10)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,y: '{:.0f}'.format(x)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,y: '{}'.format(x)))
plt.grid()
lg=plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.savefig("time.png", bbox_inches='tight', bbox_extra_artists=(lg,))



#############################
plt.figure()
plt.plot(TP[:,0], np.abs(TP[:,10]), color='r', marker='x', label="Palisade")
plt.plot(TS[:,0], np.abs(TS[:,10]), color='b', marker='x', label="Seal")
plt.plot(TH[:,0], np.abs(TH[:,10]), color='g', marker='x', label="Helib")
plt.xlabel("# batch (bs=8192) @ 1-core")
plt.ylabel("Abs. Err (he-raw)")
plt.xscale('log', basex=2)
plt.yscale('log', basey=10)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,y: '{:.0f}'.format(x)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,y: '{}'.format(x)))
plt.grid()
plt.savefig("err.png", bbox_inches='tight')


