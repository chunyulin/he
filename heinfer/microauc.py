###!sudo pip install sklearn matplotlib

import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

#LABEL = '/home/p00lcy01/he/idash_smaller/post/label.txt'
#SCORE  = '/home/p00lcy01/he/idash_smaller/post/prob2.txt'
LABEL = 'label.txt'
SCORE  = 'prob.txt'

y = label_binarize( np.loadtxt(LABEL) , classes=[1, 2, 3, 4]).astype(float)
predict = np.loadtxt(SCORE, comments='#', delimiter=',',usecols=(0,1,2,3))

# ROC and AUC for each class
fpr,tpr,roc_auc = dict(), dict(), dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y[:,i], predict[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# MicroAUC
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), predict.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

print("MicroAUC: ", roc_auc["micro"])

