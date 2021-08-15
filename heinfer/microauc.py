###!sudo pip install sklearn matplotlib

import sys
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

#LABEL = '/home/p00lcy01/he/idash_smaller/post/label.txt'
#SCORE  = '/home/p00lcy01/he/idash_smaller/post/prob2.txt'
LABEL  = sys.argv[1]
SCORE  = sys.argv[2]

y = label_binarize( np.loadtxt(LABEL) , classes=[1, 2, 3, 4]).astype(float)
predict = np.loadtxt(SCORE, comments='#', delimiter=',', usecols=(0,1,2,3))


# MicroAUC
fpr, tpr, _ = roc_curve(y.ravel(), predict.ravel())
roc_auc = auc(fpr, tpr)

print("MicroAUC: ", roc_auc)

