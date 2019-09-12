# -*- coding: utf-8 -*-
from __future__ import print_function

import os
def isfinish(path):
	path='../'+path
	finds=['structure.pkl', 'run.py', 'plot.py', 'result-1.txt', 'tps-1.txt', 'logs_1_test', 'Para_1', 'Para_1/Para.ckpt-99.meta']
	if not all([os.path.exists(path+'/'+i) for i in finds]):
		return False
	return True
folders=os.listdir('..')
folders=sorted([i for i in folders if i.startswith('Step') and os.path.isdir('../'+i)])


st=[i for i in folders if isfinish(i)]
print('Has finished:',len(st))
print('Not finished:',len([i for i in folders if i not in st]),'\n',', '.join([i for i in folders if i not in st]))