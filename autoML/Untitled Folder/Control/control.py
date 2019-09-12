# -*- coding: utf-8 -*-

import os

import random

import numpy as np

import pickle

import shutil

import autoML

from multiprocessing import Pool,Lock,Array

import time




def isfinish(path):
	path='../'+path
	finds=['structure.pkl', 'run.py', 'plot.py', 'result-1.txt', 'tps-1.txt', 'logs_1_test', 'Para_1', 'Para_1/Para.ckpt-99.meta']
	if not all([os.path.exists(path+'/'+i) for i in finds]):
		return False
	return True

def loadf1(path):
	path='../'+path
	return np.loadtxt(path+'/result-1.txt')[:,2].max()


class RequireGPU(object):
	"""docstring for RequireGPU"""
	__slots__ = ['maxlimit', 'lock','count',]
	def __init__(self, maxlimit=1):
		self.maxlimit = maxlimit
		self.lock=Lock()
		self.count=Array('i',8)
	def getdevice(self):
		self.lock.acquire()
		f=open('CUDAs.txt','r')
		cudas=[int(i) for i in f.read().strip().split('\n') if int(i) in range(8)]
		f.close()
		minuse=min([self.count[i] for i in cudas])
		if minuse>=self.maxlimit:
			choose=None
		else:
			choose=[i for i in cudas if self.count[i]==minuse][0]
			self.count[choose]+=1
		self.lock.release()
		return choose
	def release(self,x):
		self.lock.acquire()
		self.count[x]-=1
		self.lock.release()

allround=700
max_cuda=8
maxruneach=1

f=open('CUDAs.txt','r')
cudas=[str(int(i)) for i in f.read().strip().split('\n') if int(i) in range(8)]
f.close()
if cudas==[]:
	exit()
gpur=RequireGPU(maxlimit=maxruneach)

folders=os.listdir('..')
folders=sorted([i for i in folders if i.startswith('Step') and os.path.isdir('../'+i)])

finished=[isfinish(i) for i in folders]

if folders==[] or all(finished):
	todo=[]
else:
	todo=[int(i.replace('Step','')) for i,j in zip(folders,finished) if j==False]

todo+=list(range((max([int(i[4:]) for i in folders])+1 if folders else 0),allround))



def finishorrunning(x):
	if isfinish('Step{:05}'.format(x)):
		return True
	if not os.path.exists('../Step{:05}/pid'.format(x)):
		return False
	f=open('../Step{:05}/pid'.format(x),'r')
	runpid=f.read()
	f.close()
	if not os.path.exists("/proc/{}/cmdline".format(runpid)):
		return False
	f=open("/proc/{}/cmdline".format(runpid),'r')
	cmdl=f.read()
	f.close()
	if cmdl.find('python')!=-1:
		return True
	return False
def dealwith(stepnum):
	folders=os.listdir('..')
	folders=set(sorted([i for i in folders if i.startswith('Step') and os.path.isdir('../'+i)]))

	while not all(['Step{:05}'.format(i) in folders and finishorrunning(i) for i in range(stepnum)]):
		time.sleep(10)
		folders=os.listdir('..')
		folders=set(sorted([i for i in folders if i.startswith('Step') and os.path.isdir('../'+i)]))

	folders=sorted(list(folders))
	finished=[isfinish(i) for i in folders]

	structures={tuple(autoML.parseStructure('../{}/structure.pkl'.format(i)).structure) for i in folders if i!='Step{:05}'.format(stepnum) and os.path.exists('../{}/structure.pkl'.format(i))}

	if not os.path.exists('../Step{:05}'.format(stepnum)):
		os.mkdir('../Step{:05}'.format(stepnum))
	if os.path.exists('../Step{:05}/structure.pkl'.format(stepnum)):
		a=autoML.parseStructure('../Step{:05}/structure.pkl'.format(stepnum))
	else:
		if stepnum<64:
			a=autoML.WebLayerStructure(stacking_depth=3,
				space_name='SP-our',
				channel_depth=8,
				combination_number=5,)
			while tuple(a.structure) in structures:
				a=autoML.WebLayerStructure(stacking_depth=3,
					space_name='SP-our',
					channel_depth=8,
					combination_number=5,)
		else:
			alives=[]
			for j in range(len(folders)):
				if finished[j] and int(folders[j][4:])<stepnum:
					alives.append(folders[j])
			newalive=alives[-64:]
			random.shuffle(newalive)
			select_population=newalive[:16]
			f1s=[loadf1(j) for j in select_population]
			select=select_population[np.argmax(f1s)]
			olda=autoML.parseStructure('../{}/structure.pkl'.format(select))
			a=olda.mutate()
			while tuple(a.structure) in structures:
				a=olda.mutate()
			f=open('../Step{:05}/parent'.format(stepnum),'w')
			f.write(select)
			f.close()

		autoML.writeStructure('../Step{:05}/structure.pkl'.format(stepnum),a)

	shutil.copy('../template/run.py', '../Step{:05}/run.py'.format(stepnum))
	shutil.copy('../template/plot.py', '../Step{:05}/plot.py'.format(stepnum))

	device=gpur.getdevice()
	while device is None:
		time.sleep(60)
		device=gpur.getdevice()
	os.system('cd ../Step{:05};python run.py {} 2>warning;python plot.py'.format(stepnum,device))
	gpur.release(device)


pool = Pool(max_cuda*maxruneach)
result=pool.map(dealwith,todo,1)
pool.close()
pool.join()

