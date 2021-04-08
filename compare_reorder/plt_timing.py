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
#print(TP)
#print(TS)


##########################3
plt.figure()
plt.plot(TP[:,0], TP[:,1], 'r' , label="P KeyGen")
plt.plot(TS[:,0], TS[:,1], 'b' , label="S")
plt.plot(TP[:,0], TP[:,2], 'r.', label="P RotKeyG")
plt.plot(TS[:,0], TS[:,2], 'b.', label="S")
plt.plot(TP[:,0], TP[:,3], 'rx', label="P MultiKeyG")
plt.plot(TS[:,0], TS[:,3], 'bx', label="S")
plt.plot(TP[:,0], TP[:,4], 'ro', label="P Enc")
plt.plot(TS[:,0], TS[:,4], 'bo', label="S")
plt.plot(TP[:,0], TP[:,5], 'r--', label="P Dec")
plt.plot(TS[:,0], TS[:,5], 'b--', label="S")
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
plt.plot(TP[:,0], TP[:,6]/TP[:,0], color='r', marker='.', label="P Compute per bench")
plt.plot(TS[:,0], TS[:,6]/TS[:,0], color='b', marker='.', label="S")
plt.plot(TP[:,0], TP[:,7]/TP[:,0], color='r', marker='x', label="Merging / bench")
plt.plot(TS[:,0], TS[:,7]/TS[:,0], color='b', marker='x')
plt.plot(TP[:,0], TP[:,8]/TP[:,0], color='r', marker='o', label="Walltime / bench")
plt.plot(TS[:,0], TS[:,8]/TS[:,0], color='b', marker='o')

plt.xlabel("# batch (bs=8192) @ 1-core")
plt.ylabel("Seconds")
plt.xscale('log', basex=2)
plt.yscale('log', basey=10)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,y: '{:.0f}'.format(x)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,y: '{}'.format(x)))
plt.grid()
lg=plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.savefig("time.png", bbox_inches='tight', bbox_extra_artists=(lg,))

##########################3
plt.figure()
plt.plot(TP[:,0], TP[:,6]/TP[:,0], color='r', marker='.', label="P Compute per bench")
plt.plot(TS[:,0], TS[:,6]/TS[:,0], color='b', marker='.', label="S")
plt.plot(TP[:,0], TP[:,7]/TP[:,0], color='r', marker='x', label="Merging / bench")
plt.plot(TS[:,0], TS[:,7]/TS[:,0], color='b', marker='x')
plt.plot(TP[:,0], TP[:,8]/TP[:,0], color='r', marker='o', label="Walltime / bench")
plt.plot(TS[:,0], TS[:,8]/TS[:,0], color='b', marker='o')

plt.plot(TP[:,0], 1000*TP[:,9]/TS[:,0], 'k--', label="1000 x Rawtime / bench")
plt.plot(TS[:,0], 1000*TS[:,9]/TS[:,0], 'k--')

plt.xlabel("# batch (bs=8192) @ 1-core")
plt.ylabel("Seconds")
plt.xscale('log', basex=2)
plt.yscale('log', basey=10)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,y: '{:.0f}'.format(x)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,y: '{}'.format(x)))
plt.grid()
lg=plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.savefig("time_with_raw.png", bbox_inches='tight', bbox_extra_artists=(lg,))


#############################
plt.figure()
plt.plot(TP[:,0], np.abs(TP[:,10]), color='r', marker='x', label="Palisade")
plt.plot(TS[:,0], np.abs(TS[:,10]), color='b', marker='x', label="Seal")
plt.xlabel("# batch (bs=8192) @ 1-core")
plt.ylabel("Err")
plt.xscale('log', basex=2)
plt.yscale('log', basey=10)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,y: '{:.0f}'.format(x)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,y: '{}'.format(x)))
plt.grid()
plt.savefig("err.png", bbox_inches='tight')


