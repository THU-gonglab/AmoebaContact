# -*- coding: utf-8 -*-

import os,sys

def checkandmake(x):
	if not os.path.exists(x):
		os.mkdir(x)

whichiam,num=os.path.split(os.path.abspath('.'))[1].split('-')
num=int(num)-1


runscript=''
for p in ['CB-6','CB-7','CB-8','CB-9','CB-10','CB-12','CB-14','CB-16','CB-20','CA-6','CA-7','CA-8','CA-9','CA-10','CA-12','CA-14','CA-16','CA-20'][:9]:
	checkandmake("{}".format(p))
	if not os.path.exists('{}/run.py'.format(p)) or 1:
		f=open('{}/run.py'.format(p),'w')
		f.write(r'''# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse

import tensorflow as tf

import os,sys

from numpy import *
import numpy

import time

import random

import autoML

import pickle


def checkandmake(x):
	if not tf.gfile.Exists(x):
		tf.gfile.MkDir(x)



def getdata(file_path, which, cuto):
	data=[]
	for serialized_example in tf.python_io.tf_record_iterator(file_path):
		example = tf.train.Example()
		example.ParseFromString(serialized_example)

		index = example.features.feature['index'].int64_list.value[0]

		name = example.features.feature['name'].bytes_list.value[0]

		length = example.features.feature['length'].int64_list.value[0]

		count = example.features.feature['count'].int64_list.value[0]

		mi = fromstring(example.features.feature['mi'].bytes_list.value[0])
		mi.resize((length,length))

		ccmpred = fromstring(example.features.feature['ccmpred'].bytes_list.value[0])
		ccmpred.resize((length,length))

		deepcnf = fromstring(example.features.feature['deepcnf'].bytes_list.value[0])
		deepcnf.resize((length,3+8))

		seq=fromstring(example.features.feature['seq'].bytes_list.value[0])
		seql=zeros((length,20))
		seql[seq!=20,seq[seq!=20].astype(int)]=1.
		seql[seq==20,:]=1./20

		freq = fromstring(example.features.feature['freq'].bytes_list.value[0])
		freq.resize((length,21))

		gap = fromstring(example.features.feature['gap'].bytes_list.value[0])
		gap.resize((length,length))

		label = fromstring(example.features.feature['label' if which=='CB' else 'label_ca'].bytes_list.value[0])
		label.resize((length,length))
		label=(label<=cuto).astype(float)

		x,y=indices(label.shape)
		x,y=x.flatten(),y.flatten()
		m=abs(x-y)<6
		weight=ones_like(label)
		label[x[m],y[m]]=0

		pssm = fromstring(example.features.feature['pssm'].bytes_list.value[0])
		pssm.resize((length,20))

		spd = fromstring(example.features.feature['spd'].bytes_list.value[0])
		spd.resize((length,10)) # ASA Phi Psi Theta(i-1=>i+1) Tau(i-2=>i+2) HSE_alpha_up HSE_alpha_down P(C) P(H) P(E)

		if not length==ccmpred.shape[0]==ccmpred.shape[1]==deepcnf.shape[0]==seql.shape[0]==freq.shape[0]==gap.shape[0]==gap.shape[1]==label.shape[0]==label.shape[1]==pssm.shape[0]==spd.shape[0]:
			raise ValueError('Wrong shape')

		feature0d=array([length,count]).astype(float32)
		feature1d=concatenate((deepcnf,seql,freq,spd),axis=1).astype(float32)
		pos=abs(arange(length)[newaxis]-arange(length)[:,newaxis])
		feature2d=stack((ccmpred,mi,gap,pos),axis=2).astype(float32)

		data.append([feature0d,feature1d,feature2d,label[newaxis,:,:,newaxis],weight[newaxis,:,:,newaxis]])
	return data

def getlr(epo):
	if epo<10:
		return 1e-3
	if epo<20:
		return 8e-4
	if epo<30:
		return 5e-4
	if epo<40:
		return 2e-4
	return 1e-4

def main(_):
	train_file_path = os.path.join(FLAGS.data_dir, "train.tfrecords")
	logpath = os.path.join(FLAGS.model_dir, "log.txt")

	kflabel=[0,3,3,0,4,0,3,3,4,3,3,1,2,4,3,1,1,1,3,0,0,1,4,4,3,2,1,0,4,1,1,3,0,3,1,4,3,4,1,3,1,2,0,0,3,0,0,3,3,3,2,3,0,2,1,1,1,2,2,0,3,2,3,0,0,4,0,3,4,4,0,0,1,3,4,0,0,0,3,4,4,1,2,1,3,4,3,4,4,0,1,3,3,0,3,0,3,0,2,0,2,1,2,4,0,4,3,0,3,4,1,3,4,4,0,2,0,0,3,1,1,3,4,1,1,3,4,2,0,0,4,2,2,3,3,2,2,0,2,3,3,4,4,4,2,2,3,3,0,1,3,0,4,3,0,3,4,0,2,1,2,0,1,3,1,0,1,0,1,4,3,0,0,3,3,4,4,3,4,2,1,1,3,0,2,3,0,0,3,1,3,2,2,4,3,2,2,3,4,4,3,1,0,2,4,1,1,3,3,4,1,0,2,3,0,4,0,3,0,0,4,0,1,4,4,2,3,2,2,3,2,4,0,3,1,2,1,0,1,1,4,0,0,1,4,4,0,3,4,3,0,1,3,3,2,1,1,2,4,1,4,1,1,2,4,1,0,1,0,3,1,0,4,0,4,4,4,1,3,0,0,3,2,2,4,1,4,2,1,3,1,1,1,4,2,4,0,4,0,0,2,0,4,2,1,2,2,1,0,4,0,2,3,2,0,2,3,1,2,0,4,1,3,0,1,4,1,0,4,1,1,0,3,0,1,0,3,4,0,2,2,3,2,3,4,4,4,0,4,1,1,0,3,1,2,0,1,4,1,3,0,0,3,0,0,2,0,0,2,3,2,1,2,1,2,1,4,2,4,2,4,1,2,2,2,2,0,1,4,2,1,1,4,4,2,4,1,2,2,2,1,2,0,3,0,0,4,1,0,4,4,1,0,2,0,0,0,0,2,2,3,2,3,2,4,4,1,0,3,1,0,0,3,1,2,2,1,2,2,4,0,1,2,2,4,4,1,3,4,3,4,3,1,2,1,2,0,1,0,4,0,0,1,0,2,3,4,2,3,2,3,4,1,3,4,4,1,4,2,4,1,3,0,4,1,1,3,0,3,1,4,4,2,4,4,1,3,0,3,1,0,2,2,0,0,3,1,1,4,0,1,4,1,4,4,1,1,1,4,4,2,3,2,0,4,3,0,0,3,0,2,4,2,2,0,3,0,4,0,3,2,4,2,1,2,3,1,4,3,4,1,0,2,2,2,0,2,0,1,1,4,3,2,0,3,0,1,2,0,2,0,0,3,3,4,0,3,0,3,1,2,2,4,0,1,1,3,3,0,4,2,4,4,1,4,3,4,2,2,4,1,1,1,3,0,2,1,2,4,2,4,3,4,2,4,3,3,1,3,0,0,2,2,0,2,0,2,4,4,2,3,1,2,2,3,0,2,3,2,3,0,0,2,4,1,0,0,0,0,2,1,1,1,4,0,0,2,3,4,3,3,3,0,4,4,2,3,1,2,3,1,4,0,3,1,0,3,2,4,0,4,0,4,4,0,2,4,2,2,3,3,2,3,0,3,4,4,2,3,3,1,2,3,4,4,4,2,4,1,1,3,0,2,4,2,0,4,2,1,3,3,3,4,1,4,4,3,0,0,4,4,0,0,0,2,0,0,3,1,2,4,3,2,2,0,1,3,0,0,3,3,3,1,1,4,2,3,2,2,2,1,2,1,0,0,1,3,2,2,2,1,4,3,0,1,0,2,1,2,3,2,4,4,1,4,0,0,3,1,2,4,2,0,2,3,1,4,2,1,1,0,1,0,2,4,1,0,4,4,3,1,0,1,3,3,1,3,2,0,2,1,2,4,2,1,4,3,4,3,2,1,2,3,0,3,1,1,1,2,3,1,4,3,1,0,3,0,3,3,3,1,3,1,2,2,3,1,3,4,3,4,0,2,4,3,4,2,1,0,1,1,4,4,3,4,2,2,4,3,3,2,2,2,1,2,3,4,2,2,3,3,3,1,0,2,1,4,0,1,4,2,0,4,3,0,2,2,0,3,0,4,4,0,1,3,4,4,1,1,4,1,2,0,2,4,1,2,0,4,1,4,1,1,3,1,3,3,0,3,4,1,2,3,2,3,1,0,4,3,4,3,2,0,4,1,2,1,4,4,4,3,2,1,0,4,3,0,0,4,0,3,4,0,4,0,0,4,3,3,2,2,0,2,2,1,4,3,2,2,3,4,3,1,3,0,1,4,2,3,0,2,3,0,0,2,3,1,4,2,1,1,1,2,1,2,3,3,0,0,1,3,0,2,0,0,0,0,4,3,3,0,2,2,1,2,2,0,0,0,4,2,4,1,3,3,3,3,4,1,1,2,3,4,1,0,4,3,0,4,4,3,0,4,0,4,2,0,3,4,0,4,2,1,4,0,1,3,0,4,1,3,3,0,3,0,2,1,3,1,0,1,2,4,4,4,2,1,1,4,2,3,2,1,0,4,3,2,4,1,0,0,3,0,3,1,0,0,0,4,1,1,3,3,0,4,2,1,3,3,1,4,3,1,1,2,2,3,4,0,0,2,1,2,1,1,2,0,2,4,2,3,3,0,1,0,2,3,2,4,3,3,4,4,0,1,0,0,0,3,1,1,4,4,3,2,3,0,2,0,1,1,1,4,1,0,1,3,4,3,0,0,4,2,3,0,4,4,2,1,3,2,2,2,3,2,2,2,2,3,1,2,0,2,1,4,0,0,4,2,4,0,0,3,2,4,4,2,1,3,4,0,2,4,1,1,2,2,1,3,3,2,4,0,3,4,0,4,3,1,0,1,0,0,3,2,4,1,3,4,1,4,0,0,1,1,2,4,1,2,4,2,1,3,4,0,1,4,3,1,1,0,0,4,1,0,1,2,2,2,0,3,4,4,3,2,2,1,4,4,4,0,0,3,4,4,1,1,2,4,1,0,1,2,4,3,4,0,4,4,3,1,0,2,1,4,4,0,1,3,4,2,2,2,2,3,0,2,0,4,4,2,1,3,1,3,1,1,1,3,2,3,3,1,0,1,4,2,4,3,0,0,2,4,0,1,1,2,4,4,4,4,0,4,2,1,3,4,0,3,1,4,3,0,0,3,4,3,3,2,2,2,3,4,1,3,2,3,4,1,3,4,3,1,2,4,0,3,4,2,4,4,3,3,0,3,4,2,1,1,4,2,2,2,0,2,2,1,2,2,0,2,4,0,4,3,4,3,0,1,3,0,1,2,4,2,1,0,4,3,2,2,2,0,2,0,1,3,0,4,0,4,1,1,3,4,4,4,4,0,0,3,4,1,3,2,0,1,4,4,4,4,2,0,3,1,3,0,1,4,3,1,3,0,1,2,2,2,2,3,3,2,2,0,1,0,3,2,1,0,1,4,0,1,4,4,0,3,3,2,0,3,1,4,0,3,1,0,1,0,3,0,3,4,1,4,4,4,2,3,1,1,0,0,0,3,2,2,1,1,2,3,2,4,0,0,3,1,2,1,4,2,3,0,1,0,1,4,2,1,3,1,1,3,2,1,3,0,3,4,3,3,1,2,3,2,2,2,0,3,3,0,3,4,3,0,4,0,0,2,4,0,1,1,2,1,3,1,3,3,1,3,0,0,1,3,1,2,2,1,3,1,3,1,3,4,3,1,2,0,2,3,2,3,4,2,0,4,3,1,3,4,2,3,0,4,0,0,3,1,1,4,2,2,1,4,0,2,4,0,2,1,3,3,3,0,1,1,2,2,2,3,0,4,3,2,1,0,1,4,3,1,2,3,0,1,0,2,3,1,0,1,0,3,1,4,4,3,4,4,4,4,0,0,4,4,0,1,4,2,4,2,2,2,4,2,1,1,2,0,0,1,0,3,4,4,0,1,0,4,3,2,1,2,0,1,4,1,3,3,2,0,0,3,0,2,3,4,3,2,0,3,2,2,2,2,1,3,0,4,3,0,1,1,2,4,1,2,0,0,1,1,4,4,0,1,1,4,3,3,3,1,4,2,0,3,2,1,0,1,3,2,1,0,1,1,3,2,0,0,4,4,0,2,1,1,3,2,3,2,2,1,4,0,3,0,1,1,4,3,4,3,4,3,1,3,4,4,3,1,4,2,2,4,1,3,1,1,3,1,2,3,0,3,2,4,1,1,4,1,0,2,1,2,2,2,3,1,2,2,2,4,3,1,4,0,3,0,0,0,0,0,3,2,3,2,3,4,4,4,0,1,2,0,2,4,1,4,4,2,0,3,2,3,3,0,2,4,0,1,1,0,3,0,4,3,3,0,1,4,4,4,3,4,1,1,1,0,3,2,1,2,0,2,2,2,1,0,2,0,4,4,3,0,2,1,0,1,3,3,3,1,2,3,1,2,4,1,1,2,1,4,1,0,4,2,1,1,3,0,3,1,2,1,1,0,3,3,2,3,2,2,3,3,2,4,3,1,1,4,4,3,1,0,4,4,3,4,3,3,3,3,1,2,4,0,4,1,3,4,4,0,3,4,0,1,1,3,3,0,0,4,1,2,2,1,3,2,4,1,2,2,0,4,4,0,0,3,3,2,3,4,0,4,2,3,0,0,3,2,0,1,3,0,1,2,0,2,1,1,0,4,0,2,0,4,2,4,4,3,2,1,4,4,0,4,2,1,2,0,0,4,3,2,1,1,3,2,4,2,2,1,1,2,1,0,3,2,0,4,0,1,1,1,3,4,4,1,4,2,4,3,0,0,1,2,2,4,0,0,0,1,3,1,2,4,0,4,3,4,3,2,4,3,4,3,4,2,3,1,0,0,4,1,1,3,2,4,3,4,4,2,2,2,3,2,2,0,0,3,1,0,0,1,2,0,0,3,2,4,2,2,1,2,2,2,0,3,1,0,0,3,0,4,2,1,4,1,2,1,4,2,3,1,4,3,1,2,3,2,2,1,3,0,3,0,2,1,2,1,3,3,3,3,0,3,0,3,3,1,1,2,0,2,0,0,4,3,2,2,4,4,2,3,2,4,3,1,1,4,1,0,3,4,0,0,1,2,0,1,0,2,4,0,4,0,1,0,3,4,1,2,4,4,2,0,2,4,1,3,0,4,0,2,3,0,3,3,1,3,0,1,1,1,4,1,1,4,3,2,4,2,0,0,1,2,2,1,1,0,0,1,2,2,2,3,2,0,1,1,4,4,2,2,4,4,3,1,3,0,1,0,1,4,4,1,3,2,4,0,1,1,1,2,1,3,2,2,4,4,1,0,0,4,1,4,3,0,2,0,3,4,4,4,4,0,2,4,2,2,4,4,4,3,1,2,4,3,1,0,0,2,1,2,1,4,1,4,0,0,4,0,0,3,0,1,1,1,3,1,1,0,3,4,3,0,1,4,3,1,1,4,1,4,4,1,0,1,4,1,1,4,2,4,2,0,1,0,4,2,1,1,1,2,4,0,1,2,2,1,1,4,0,3,0,2,2,4,1,2,0,0,0,4,4,4,0,4,1,2,1,1,0,4,3,2,2,4,4,1,3,0,2,4,3,2,3,4,4,1,1,0,1,3,4,4,3,2,4,0,3,2,0,0,0,3,4,0,2,0,2,0,3,0,3,2,3,2,1,2,4,1,1,2,3,0,4,1,1,2,1,1,1,2,3,4,3,0,3,0,0,1,2,4,1,4,3,0,3,4,1,0,2,2,2,1,4,1,1,2,2,3,1,2,0,0,1,1,0,2,4,0,2,0,4,0,4,1,1,0,3,1,1,3,3,4,0,3,2,3,1,3,0,4,1,0,0,3,0,0,4,2,0,3,4,3,4,1,4,2,2,2,2,1,0,3,2,1,2,2,3,0,1,4,0,1,0,0,0,4,0,3,0,3,0,0,3,2,4,4,0,4,3,3,4,4,2,4,3,3,3,0,1,2,0,2,0,1,4,3,0,4,1,4,3,2,4,1,3,0,4,3,0,2,3,3,1,0,4,4,3,3,3,2,2,4,4,2,1,4,0,0,4,4,3,4,2,2,2,4,1,3,3,3,1,2,3,0,4,0,4,0,0,1,2,4,3,1,3,2,2,3,4,0,2,4,1,4,2,2,3,1,3,3,1,1,0,3,1,2,4,1,4,2,1,3,1,0,1,0,2,2,3,0,4,4,3,4,1,3,0,3,3,3,3,2,3,3,1,2,4,0,0,0,0,2,0,1,3,2,1,0,0,0,2,3,3,0,0,0,1,0,4,4,3,1,0,1,0,1,0,3,1,1,4,3,3,3,2,3,1,4,2,0,3,4,0,0,4,3,1,4,3,2,3,3,1,2,3,1,4,3,4,3,1,2,2,4,4,0,3,3,3,4,0,2,2,1,1,4,0,3,0,2,4,1,4,4,1,4,1,2,0,0,0,2,3,3,1,4,0,0,2,4,3,3,0,2,1,2,4,4,0,1,1,2,0,1,1,3,0,3,3,3,1,0,4,2,2,4,2,3,0,1,4,4,4,0,2,3,4,4,3,2,1,3,2,3,2,0,2,3,3,2,2,1,3,4,1,2,0,2,1,2,4,0,3,3,0,0,4,2,0,4,1,3,4,4,3,4,1,2,1,0,1,2,3,4,2,3,1,3,4,3,2,2,3,0,4,2,0,2,2,4,3,1,1,2,4,3,2,3,1,2,2,4,0,3,0,2,4,4,0,2,2,2,2,0,4,2,0,2,4,1,1,3,3,1,2,1,2,2,4,0,1,2,3,1,1,1,3,4,4,2,4]
	kflabel=array(kflabel)
	n=len(kflabel)

	whichlabel,cuto=os.path.abspath('.').split('/')[-1].split('-')
	cuto=int(cuto)
	data=getdata(train_file_path,whichlabel,cuto)
	inputsize=data[0][0].shape[0]+data[0][1].shape[1]*2+data[0][2].shape[2]

	modelnumber=os.path.abspath('.').split('/')[-2]
	f=open('../../../90_Enlarge/00_Select/{}'.format(modelnumber.split('-')[0]),'r')
	stepnum=f.read().strip().split()[int(modelnumber.split('-')[1])-1]
	f.close()

	# Build network
	already=[int(j.split('.')[-2].split('-')[-1]) for j in os.listdir("Para_1") if j.endswith('.meta')] if os.path.exists('Para_1') else []
	if already==[]:
		structure=autoML.parseStructure('structure.pkl')
		if whichlabel=='CB':
			prepath='../../../90_Enlarge/{}/{}-5-30'.format(modelnumber,stepnum)
		else:
			prepath='../CB-{}'.format(cuto)
		bestepoc=loadtxt('{}/result-1.txt'.format(prepath))[:,2].argmax()
		epocs=sorted([int('.'.join(i.split('.')[:-1]).replace('Para.ckpt-','')) for i in os.listdir('{}/Para_1'.format(prepath)) if i.endswith('.meta')])
		bestepoc=epocs[bestepoc]
		parentpath=os.path.join(os.path.join(prepath,'Para_1','Para.ckpt-{}'.format(bestepoc)))
		weights=autoML.parseWeight(parentpath)
		lr = tf.placeholder("float", shape=(), name='lr')
		x, y, istrain, realpred, train_step, merged_summary_op=autoML.buildNet(inputsize, structure, initweight=weights, lr=1e-4, rncn=True)
		choose=-1
	else:
		choose=max(already)
		prevsaver = tf.train.import_meta_graph('Para_1/Para.ckpt-{}.meta'.format(choose))
		graph = tf.get_default_graph()

	stringlog=''

	crosspart=0
	epicnum = (100 if cuto in [6,8] else (150 if cuto in [7,9] else (200))) if whichlabel=='CB' else 50

	for i in range(5)[:1]:
		crosspart+=1
		train_index=(kflabel!=i).nonzero()[0]
		test_index=(kflabel==i).nonzero()[0]

		print("Cross validataion #{}: (Time stamp:{})".format(crosspart,time.time()))
		stringlog+="Cross validataion #{}: (Time stamp:{})\n".format(crosspart,time.time())

		if tf.gfile.Exists(os.path.join(FLAGS.model_dir,'Para_{}/Para.ckpt-{}.index'.format(crosspart,epicnum-1))):
			continue

		stringresult=''
		stringtp=''
		checkandmake(os.path.join(FLAGS.model_dir,"Para_{}".format(i+1)))

		with tf.Session() as sess:
			if already==[]:
				sess.run(tf.global_variables_initializer())
			else:
				prevsaver.restore(sess, 'Para_1/Para.ckpt-{}'.format(choose))
				x=graph.get_operation_by_name('Input').outputs[0]
				y=graph.get_operation_by_name('Label').outputs[0]
				lr=graph.get_operation_by_name('lr').outputs[0]
				istrain=graph.get_operation_by_name('Placeholder').outputs[0]
				merged_summary_op=graph.get_operation_by_name('Merge/MergeSummary').outputs[0]
				train_step=graph.get_operation_by_name('train_step/control_dependency').outputs[0]
				realpred=graph.get_operation_by_name('Prediction').outputs[0]

			summary_writer_test = tf.summary.FileWriter(os.path.join(FLAGS.model_dir,'logs_{}_test'.format(crosspart)), graph=sess.graph)

			saver = tf.train.Saver(max_to_keep=epicnum+1)

			for epoc in range(choose+1,epicnum):
				print('Running fold:{} epoc:{} (Time stamp:{})'.format(crosspart,epoc,time.time()))
				stringlog+='Running fold:{} epoc:{} (Time stamp:{})\n'.format(crosspart,epoc,time.time())
				trainrank=train_index.copy()
				random.shuffle(trainrank)

				casecount=0
				for i in trainrank:
					casecount+=1
					f0d,f1d,f2d,labelfeed,w=data[i]
					n=f1d.shape[0]
					# print('\r\tTraining {:4d}, {:4d}/{:4d}'.format(i,casecount,len(trainrank)),end='')
					# sys.stdout.flush()
					_=sess.run(train_step,feed_dict={x: concatenate([
																f2d,
																tile(f1d[newaxis],[n,1,1]),
																tile(f1d[:,newaxis],[1,n,1]),
																tile(f0d[newaxis,newaxis],[n,n,1]),
																],axis=2)[newaxis],
													y: labelfeed,
													istrain:True,
													lr:getlr(epoc)})

				if epoc%1==0:
					saver.save(sess, os.path.join(FLAGS.model_dir,"Para_{}/Para.ckpt".format(crosspart)), global_step=epoc)

					testreal=[]
					testpred=[]
					testrank=test_index.copy()
					casecount=0
					for i in testrank:
						casecount+=1
						f0d,f1d,f2d,labelfeed,w=data[i]
						n=f1d.shape[0]
						# print('\r\tTesting  {:4d}, {:4d}/{:4d}'.format(i,casecount,len(test_index)),end='')
						# sys.stdout.flush()
						predresult,summary_str=sess.run([realpred,merged_summary_op],feed_dict={x: concatenate([
																											f2d,
																											tile(f1d[newaxis],[n,1,1]),
																											tile(f1d[:,newaxis],[1,n,1]),
																											tile(f0d[newaxis,newaxis],[n,n,1]),
																											],axis=2)[newaxis],
																								y: labelfeed,
																								# weight:w,
																								istrain:False})
						summary_writer_test.add_summary(summary_str, epoc)

						testreal.append(labelfeed[0,:,:,0])
						testpred.append((predresult[0,:,:,0]+predresult[0,:,:,0].T)/2.)
					pred=[]
					real=[]
					saver1=[]
					ratio=[0.1,0.2,0.5,1.]
					for s,l in zip(testpred,testreal):
						n=s.shape[0]
						x1,y1=indices(l.shape)
						mask=(x1-y1>=6)
						x1,y1=x1[mask],y1[mask]
						pred.append(s[x1,y1])
						real.append(l[x1,y1])

						saver1.append([])
						xx,yy=x1[((x1-y1)>=6)&((x1-y1)<=11)],y1[((x1-y1)>=6)&((x1-y1)<=11)]
						rank=s[xx,yy].argsort()[::-1]
						for j in range(len(ratio)):
							temp=rank[int(ratio[j]*n)-1]
							c=s[xx[temp],yy[temp]]
							temp=(s[xx,yy]>=c)
							saver1[-1].append(l[xx[temp],yy[temp]].sum()*1./l[xx[temp],yy[temp]].size)

						xx,yy=x1[((x1-y1)>=12)&((x1-y1)<=23)],y1[((x1-y1)>=12)&((x1-y1)<=23)]
						rank=s[xx,yy].argsort()[::-1]
						for j in range(len(ratio)):
							temp=rank[int(ratio[j]*n)-1]
							c=s[xx[temp],yy[temp]]
							temp=(s[xx,yy]>=c)
							saver1[-1].append(l[xx[temp],yy[temp]].sum()*1./l[xx[temp],yy[temp]].size)

						xx,yy=x1[((x1-y1)>=24)],y1[((x1-y1)>=24)]
						rank=s[xx,yy].argsort()[::-1]
						for j in range(len(ratio)):
							temp=rank[int(ratio[j]*n)-1]
							c=s[xx[temp],yy[temp]]
							temp=(s[xx,yy]>=c)
							saver1[-1].append(l[xx[temp],yy[temp]].sum()*1./l[xx[temp],yy[temp]].size)
					saver1=array(saver1)
					saver1=saver1.mean(axis=0)

					testreal=hstack(real)
					testpred=hstack(pred)

					prec=[]
					recall=[]
					f1=[]
					findps=[]
					tps=[]
					realps=[]
					realp=testreal.sum()
					for cutoff in arange(0,1,.02):
						bol=testpred>cutoff
						findp=bol.sum()
						findps.append(findp)
						tp=testreal[bol].sum()
						tps.append(tp)
						realps.append(realp)
						prec.append(1.*tp/findp if findp!=0 else 0.)
						recall.append(1.*tp/realp if realp!=0 else 0.)
						f1.append(2.*tp/(realp+findp) if realp+findp!=0 else 0.)
					startcutoff=arange(0,1,.02)[argmax(f1)]
					# print('')
					print("{:.6f} {:.6f} {:.6f}".format(prec[argmax(f1)],recall[argmax(f1)],f1[argmax(f1)]))
					print(' '.join(["{:.4f}".format(i) for i in saver1]))

					flog=tf.gfile.Open(logpath,'ab')
					flog.write("{:.6f} {:.6f} {:.6f} ".format(prec[argmax(f1)],recall[argmax(f1)],f1[argmax(f1)]))
					flog.write(' '.join(["{:.4f}".format(i) for i in saver1])+'\n')
					flog.close()

					fresult=tf.gfile.Open(os.path.join(FLAGS.model_dir,'result-{}.txt'.format(crosspart)),'ab')
					fresult.write("{:.6f} {:.6f} {:.6f} ".format(prec[argmax(f1)],recall[argmax(f1)],f1[argmax(f1)]))
					fresult.write(' '.join(["{:.4f}".format(i) for i in saver1])+'\n')
					fresult.close()

					ftp=tf.gfile.Open(os.path.join(FLAGS.model_dir,'tps-{}.txt'.format(crosspart)),'ab')
					ftp.write(" ".join([str(i) for i in findps+tps+realps])+'\n')
					ftp.close()
	



if __name__ == '__main__':
	modelnumber=os.path.abspath('.').split('/')[-2]
	f=open('../../../90_Enlarge/00_Select/{}'.format(modelnumber.split('-')[0]),'r')
	stepnum=f.read().strip().split()[int(modelnumber.split('-')[1])-1]
	f.close()
	a=autoML.parseStructure('../../../90_Enlarge/{}/{}-5-30/structure.pkl'.format(modelnumber,stepnum))
	autoML.writeStructure('structure.pkl',a)
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='../../../85_New_Train/01_Build_TF/',help='input data path')
	parser.add_argument('--model_dir', type=str, default='./',help='output model path')
	FLAGS, _ = parser.parse_known_args()
	os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
	tf.app.run(main=main)''')
		f.close()
	if not os.path.exists('{}/plot.py'.format(p)):
		f=open('{}/plot.py'.format(p),'w')
		f.write(r'''# -*- coding: utf-8 -*-


from numpy import *

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
plt.rc('axes', prop_cycle=(cycler('color', ['#16a085', '#2980b9','#c0392b','#7f8c8d', '#8e44ad','#2ecc71','#2c3e50','#d35400','#bdc3c7',"#f39c12"])))

import os

f1=loadtxt('result-1.txt',ndmin=2)[:,2]

fig, ax = plt.subplots()
majorLocator = MultipleLocator(.1)
minorLocator = MultipleLocator(.02)
ax.yaxis.set_major_locator(majorLocator)
ax.yaxis.set_minor_locator(minorLocator)
epocs=sorted([int('.'.join(i.split('.')[:-1]).replace('Para.ckpt-','')) for i in os.listdir('Para_1') if i.endswith('.meta')])
plot(epocs,f1,label='Cross',lw=2,c='black',linestyle='-')
print(max(f1))
grid('on',alpha=.5,zorder=0,linestyle='--')
grid('on',which='minor',alpha=.3,zorder=0,linestyle='--')
ylim(floor(min(f1)*100)/100.,ceil(max(f1)*100)/100.)
yticks(arange(floor(min(f1)*100)/100.,ceil(max(f1)*100)/100.+.01,.01))
xlabel('epoc')
ylabel('F1-score')
tight_layout()
legend()
savefig('Cross-F1',dpi=300)''')
		f.close()
	runscript+='''cd {}\npython -u run.py {} 1>out 2>warning\npython plot.py\ncd ..\n\n'''.format(p,sys.argv[1])

f=open('run.sh','w')
f.write(runscript)
f.close()
