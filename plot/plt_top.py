import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import sys

def tonum(x):
  if x[-1] == 'g':
    return float(x[:-1])*1024*1024
  if x[-1] == 'm':
    return float(x[:-1])*1024
  return float(x)

#  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
#107450 lincy     20   0 1280384 717248   9152 R 130.5  0.4   0:12.75 HE_sigmoid

FNAME="../top.dat"

#raw = np.loadtxt("../top.log", delimiter=" ", usecols=(4,6,8,9))
data = []
with open(FNAME) as f:
  for line in f:
    l=line.strip().split()
    l[4] = tonum(l[4])
    l[5] = tonum(l[5])
    data.append([l[4],l[5],l[6],l[8],l[9] ])

t=np.array(range(len(data)))*0.5
data=np.array(data).astype(np.float)

vm=data[:,0]
res=data[:,1]
shr=data[:,2]
cpu=data[:,3]
mem=data[:,4]
#print(data)

plt.figure(figsize=(7,2))
plt.plot(t, vm*1e-6, label="VIRT")
plt.plot(t, res*1e-6, label="RES")
plt.ylabel("GBytes")
plt.xlabel("Time (sec)")
#plt.grid()
plt.legend()
plt.savefig("top_mem.png", bbox_inches='tight')

plt.figure(figsize=(7,2))
plt.plot(t, cpu, label="%CPU")
plt.ylabel("% CPU")
plt.xlabel("Time (sec)")
#plt.grid()
plt.savefig("top_cpu.png", bbox_inches='tight')

