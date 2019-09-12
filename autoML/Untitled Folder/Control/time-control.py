# -*- coding: utf-8 -*-

import os

folders=sorted([i for i in os.listdir('..') if i.startswith('Step')])
folders=['../'+i for i in folders if os.path.isdir('../'+i)]

for fold in folders:
	if os.path.exists(os.path.join(fold,'log.txt')):
		f=open(os.path.join(fold,'log.txt'),'r')
		a=[i for i in f.read().strip().replace('\r','\n').split('\n') if i.find('Time stamp')!=-1 and i.find('Running fold')!=-1]
		f.close()
		alltime=0
		for c in range(1,6):
			b=[float(i.split('Time stamp:')[1].split(')')[0]) for i in a if i.startswith('Running fold:{}'.format(c))]
			if len(b)!=0:
				alltime+=max(b)-min(b)
		print fold.strip('./'),"{:02.0f}:{:02.0f}:{:05.2f}".format(alltime//3600,alltime%3600//60,alltime%60),alltime