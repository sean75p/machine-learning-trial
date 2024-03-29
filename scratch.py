# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import KFold
# data sample
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# prepare cross validation
kfold = KFold(3, True, 1)
# enumerate splits
for train, test in kfold.split(data):
	print('train: %s, test: %s' % (data[train], data[test]))

import numpy
a = numpy.zeros((3, 3))
print(a)
a[1, :] = 3
a[2, :] = 4
a[2,2] = 99
print(a)
b = a[:, 2]
print(b)
print(a[0:3])