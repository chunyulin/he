import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import sys


#COLS=5
#label = ["batch size", "sf", "--", "rerr"]
#bs=[512, 1024, 2048, 4096, 8192]
#T60 = np.loadtxt("sf_fb60.log", delimiter=" ", usecols=np.arange(1, COLS))
#T49 = np.loadtxt("sf_fb49.log", delimiter=" ", usecols=np.arange(1, COLS))


COLS=16
label = ["NB", "KeyG", "RotKeyG", "MultKeyG",.
         "Enc", "Dec", "Compute per batch", "SumMerge", "Wall time", "Raw time", "Err",
         "batch size", "sf", "cbits", "rerr"]
         
TP = np.loadtxt("tpali.dat", delimiter=" ", usecols=np.arange(1, COLS))
TS = np.loadtxt("tseal.dat", delimiter=" ", usecols=np.arange(1, COLS))
TH = np.loadtxt("theli.dat", delimiter=" ", usecols=np.arange(1, COLS))
         



from matplotlib.colors import LogNorm

f = plt.figure()
for i in bs:
   data = T60[T60[:,0]==i]
   plt.plot(60-data[:,1], np.abs(data[:,3]),  marker='.', label="fb=60 bs={}".format(i))

plt.gca().set_prop_cycle(None)

for i in bs:
   data = T49[T49[:,0]==i]
   plt.plot(49-data[:,1], np.abs(data[:,3]), linestyle='dashed', marker='.',label="fb=49 bs={}".format(i))

#lt.xscale('log', basex=2)
plt.yscale('log', basey=10)
plt.xlabel("First bit - scale factor         (RingD=16382)")
plt.ylabel("% Relative error. (he-raw)/raw*100")
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,y: '{:.0f}'.format(x)))
#plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,y: '{}'.format(x)))
#plt.grid()

lg=plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.savefig("sf.png", bbox_inches='tight', bbox_extra_artists=(lg,))

