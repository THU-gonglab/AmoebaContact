# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse

import tensorflow as tf

import os,sys

from numpy import *
import numpy

import time

import random
sys.path.append('../../')
import autoML
import pickle

import socket

hostname=socket.gethostname()


def checkandmake(x):
	if not tf.gfile.Exists(x):
		tf.gfile.MkDir(x)



def getdata(file_path):
	data=[]
	for serialized_example in tf.python_io.tf_record_iterator(file_path):
		example = tf.train.Example()
		example.ParseFromString(serialized_example)

		index = example.features.feature['index'].int64_list.value[0]

		name = example.features.feature['name'].bytes_list.value[0]

		mi = load('../../04_mi/{}.npy'.format(name))

		length = example.features.feature['length'].int64_list.value[0]

		count = example.features.feature['count'].int64_list.value[0]

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

		label = fromstring(example.features.feature['label'].bytes_list.value[0])
		label.resize((length,length))
		label=(label<=8).astype(float)

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



def main(_):
	train_file_path = os.path.join(FLAGS.data_dir, "train.tfrecords")
	logpath = os.path.join(FLAGS.model_dir, "log.txt")

	kflabel=[0,3,3,0,4,0,3,3,4,3,3,1,2,4,3,1,1,1,3,0,0,1,4,4,3,2,1,0,4,1,1,3,0,3,1,4,3,4,1,3,1,2,0,0,3,0,0,3,3,3,2,3,0,2,1,1,1,2,2,0,3,2,3,0,0,4,0,3,4,4,0,0,1,3,4,0,0,0,3,4,4,1,2,1,3,4,3,4,4,0,1,3,3,0,3,0,3,0,2,0,2,1,2,4,0,4,3,0,3,4,1,3,4,4,0,2,0,0,3,1,1,3,4,1,1,3,4,2,0,0,4,2,2,3,3,2,2,0,2,3,3,4,4,4,2,2,3,3,0,1,3,0,4,3,0,3,4,0,2,1,2,0,1,3,1,0,1,0,1,4,3,0,0,3,3,4,4,3,4,2,1,1,3,0,2,3,0,0,3,1,3,2,2,4,3,2,2,3,4,4,3,1,0,2,4,1,1,3,3,4,1,0,2,3,0,4,0,3,0,0,4,0,1,4,4,2,3,2,2,3,2,4,0,3,1,2,1,0,1,1,4,0,0,1,4,4,0,3,4,3,0,1,3,3,2,1,1,2,4,1,4,1,1,2,4,1,0,1,0,3,1,0,4,0,4,4,4,1,3,0,0,3,2,2,4,1,4,2,1,3,1,1,1,4,2,4,0,4,0,0,2,0,4,2,1,2,2,1,0,4,0,2,3,2,0,2,3,1,2,0,4,1,3,0,1,4,1,0,4,1,1,0,3,0,1,0,3,4,0,2,2,3,2,3,4,4,4,0,4,1,1,0,3,1,2,0,1,4,1,3,0,0,3,0,0,2,0,0,2,3,2,1,2,1,2,1,4,2,4,2,4,1,2,2,2,2,0,1,4,2,1,1,4,4,2,4,1,2,2,2,1,2,0,3,0,0,4,1,0,4,4,1,0,2,0,0,0,0,2,2,3,2,3,2,4,4,1,0,3,1,0,0,3,1,2,2,1,2,2,4,0,1,2,2,4,4,1,3,4,3,4,3,1,2,1,2,0,1,0,4,0,0,1,0,2,3,4,2,3,2,3,4,1,3,4,4,1,4,2,4,1,3,0,4,1,1,3,0,3,1,4,4,2,4,4,1,3,0,3,1,0,2,2,0,0,3,1,1,4,0,1,4,1,4,4,1,1,1,4,4,2,3,2,0,4,3,0,0,3,0,2,4,2,2,0,3,0,4,0,3,2,4,2,1,2,3,1,4,3,4,1,0,2,2,2,0,2,0,1,1,4,3,2,0,3,0,1,2,0,2,0,0,3,3,4,0,3,0,3,1,2,2,4,0,1,1,3,3,0,4,2,4,4,1,4,3,4,2,2,4,1,1,1,3,0,2,1,2,4,2,4,3,4,2,4,3,3,1,3,0,0,2,2,0,2,0,2,4,4,2,3,1,2,2,3,0,2,3,2,3,0,0,2,4,1,0,0,0,0,2,1,1,1,4,0,0,2,3,4,3,3,3,0,4,4,2,3,1,2,3,1,4,0,3,1,0,3,2,4,0,4,0,4,4,0,2,4,2,2,3,3,2,3,0,3,4,4,2,3,3,1,2,3,4,4,4,2,4,1,1,3,0,2,4,2,0,4,2,1,3,3,3,4,1,4,4,3,0,0,4,4,0,0,0,2,0,0,3,1,2,4,3,2,2,0,1,3,0,0,3,3,3,1,1,4,2,3,2,2,2,1,2,1,0,0,1,3,2,2,2,1,4,3,0,1,0,2,1,2,3,2,4,4,1,4,0,0,3,1,2,4,2,0,2,3,1,4,2,1,1,0,1,0,2,4,1,0,4,4,3,1,0,1,3,3,1,3,2,0,2,1,2,4,2,1,4,3,4,3,2,1,2,3,0,3,1,1,1,2,3,1,4,3,1,0,3,0,3,3,3,1,3,1,2,2,3,1,3,4,3,4,0,2,4,3,4,2,1,0,1,1,4,4,3,4,2,2,4,3,3,2,2,2,1,2,3,4,2,2,3,3,3,1,0,2,1,4,0,1,4,2,0,4,3,0,2,2,0,3,0,4,4,0,1,3,4,4,1,1,4,1,2,0,2,4,1,2,0,4,1,4,1,1,3,1,3,3,0,3,4,1,2,3,2,3,1,0,4,3,4,3,2,0,4,1,2,1,4,4,4,3,2,1,0,4,3,0,0,4,0,3,4,0,4,0,0,4,3,3,2,2,0,2,2,1,4,3,2,2,3,4,3,1,3,0,1,4,2,3,0,2,3,0,0,2,3,1,4,2,1,1,1,2,1,2,3,3,0,0,1,3,0,2,0,0,0,0,4,3,3,0,2,2,1,2,2,0,0,0,4,2,4,1,3,3,3,3,4,1,1,2,3,4,1,0,4,3,0,4,4,3,0,4,0,4,2,0,3,4,0,4,2,1,4,0,1,3,0,4,1,3,3,0,3,0,2,1,3,1,0,1,2,4,4,4,2,1,1,4,2,3,2,1,0,4,3,2,4,1,0,0,3,0,3,1,0,0,0,4,1,1,3,3,0,4,2,1,3,3,1,4,3,1,1,2,2,3,4,0,0,2,1,2,1,1,2,0,2,4,2,3,3,0,1,0,2,3,2,4,3,3,4,4,0,1,0,0,0,3,1,1,4,4,3,2,3,0,2,0,1,1,1,4,1,0,1,3,4,3,0,0,4,2,3,0,4,4,2,1,3,2,2,2,3,2,2,2,2,3,1,2,0,2,1,4,0,0,4,2,4,0,0,3,2,4,4,2,1,3,4,0,2,4,1,1,2,2,1,3,3,2,4,0,3,4,0,4,3,1,0,1,0,0,3,2,4,1,3,4,1,4,0,0,1,1,2,4,1,2,4,2,1,3,4,0,1,4,3,1,1,0,0,4,1,0,1,2,2,2,0,3,4,4,3,2,2,1,4,4,4,0,0,3,4,4,1,1,2,4,1,0,1,2,4,3,4,0,4,4,3,1,0,2,1,4,4,0,1,3,4,2,2,2,2,3,0,2,0,4,4,2,1,3,1,3,1,1,1,3,2,3,3,1,0,1,4,2,4,3,0,0,2,4,0,1,1,2,4,4,4,4,0,4,2,1,3,4,0,3,1,4,3,0,0,3,4,3,3,2,2,2,3,4,1,3,2,3,4,1,3,4,3,1,2,4,0,3,4,2,4,4,3,3,0,3,4,2,1,1,4,2,2,2,0,2,2,1,2,2,0,2,4,0,4,3,4,3,0,1,3,0,1,2,4,2,1,0,4,3,2,2,2,0,2,0,1,3,0,4,0,4,1,1,3,4,4,4,4,0,0,3,4,1,3,2,0,1,4,4,4,4,2,0,3,1,3,0,1,4,3,1,3,0,1,2,2,2,2,3,3,2,2,0,1,0,3,2,1,0,1,4,0,1,4,4,0,3,3,2,0,3,1,4,0,3,1,0,1,0,3,0,3,4,1,4,4,4,2,3,1,1,0,0,0,3,2,2,1,1,2,3,2,4,0,0,3,1,2,1,4,2,3,0,1,0,1,4,2,1,3,1,1,3,2,1,3,0,3,4,3,3,1,2,3,2,2,2,0,3,3,0,3,4,3,0,4,0,0,2,4,0,1,1,2,1,3,1,3,3,1,3,0,0,1,3,1,2,2,1,3,1,3,1,3,4,3,1,2,0,2,3,2,3,4,2,0,4,3,1,3,4,2,3,0,4,0,0,3,1,1,4,2,2,1,4,0,2,4,0,2,1,3,3,3,0,1,1,2,2,2,3,0,4,3,2,1,0,1,4,3,1,2,3,0,1,0,2,3,1,0,1,0,3,1,4,4,3,4,4,4,4,0,0,4,4,0,1,4,2,4,2,2,2,4,2,1,1,2,0,0,1,0,3,4,4,0,1,0,4,3,2,1,2,0,1,4,1,3,3,2,0,0,3,0,2,3,4,3,2,0,3,2,2,2,2,1,3,0,4,3,0,1,1,2,4,1,2,0,0,1,1,4,4,0,1,1,4,3,3,3,1,4,2,0,3,2,1,0,1,3,2,1,0,1,1,3,2,0,0,4,4,0,2,1,1,3,2,3,2,2,1,4,0,3,0,1,1,4,3,4,3,4,3,1,3,4,4,3,1,4,2,2,4,1,3,1,1,3,1,2,3,0,3,2,4,1,1,4,1,0,2,1,2,2,2,3,1,2,2,2,4,3,1,4,0,3,0,0,0,0,0,3,2,3,2,3,4,4,4,0,1,2,0,2,4,1,4,4,2,0,3,2,3,3,0,2,4,0,1,1,0,3,0,4,3,3,0,1,4,4,4,3,4,1,1,1,0,3,2,1,2,0,2,2,2,1,0,2,0,4,4,3,0,2,1,0,1,3,3,3,1,2,3,1,2,4,1,1,2,1,4,1,0,4,2,1,1,3,0,3,1,2,1,1,0,3,3,2,3,2,2,3,3,2,4,3,1,1,4,4,3,1,0,4,4,3,4,3,3,3,3,1,2,4,0,4,1,3,4,4,0,3,4,0,1,1,3,3,0,0,4,1,2,2,1,3,2,4,1,2,2,0,4,4,0,0,3,3,2,3,4,0,4,2,3,0,0,3,2,0,1,3,0,1,2,0,2,1,1,0,4,0,2,0,4,2,4,4,3,2,1,4,4,0,4,2,1,2,0,0,4,3,2,1,1,3,2,4,2,2,1,1,2,1,0,3,2,0,4,0,1,1,1,3,4,4,1,4,2,4,3,0,0,1,2,2,4,0,0,0,1,3,1,2,4,0,4,3,4,3,2,4,3,4,3,4,2,3,1,0,0,4,1,1,3,2,4,3,4,4,2,2,2,3,2,2,0,0,3,1,0,0,1,2,0,0,3,2,4,2,2,1,2,2,2,0,3,1,0,0,3,0,4,2,1,4,1,2,1,4,2,3,1,4,3,1,2,3,2,2,1,3,0,3,0,2,1,2,1,3,3,3,3,0,3,0,3,3,1,1,2,0,2,0,0,4,3,2,2,4,4,2,3,2,4,3,1,1,4,1,0,3,4,0,0,1,2,0,1,0,2,4,0,4,0,1,0,3,4,1,2,4,4,2,0,2,4,1,3,0,4,0,2,3,0,3,3,1,3,0,1,1,1,4,1,1,4,3,2,4,2,0,0,1,2,2,1,1,0,0,1,2,2,2,3,2,0,1,1,4,4,2,2,4,4,3,1,3,0,1,0,1,4,4,1,3,2,4,0,1,1,1,2,1,3,2,2,4,4,1,0,0,4,1,4,3,0,2,0,3,4,4,4,4,0,2,4,2,2,4,4,4,3,1,2,4,3,1,0,0,2,1,2,1,4,1,4,0,0,4,0,0,3,0,1,1,1,3,1,1,0,3,4,3,0,1,4,3,1,1,4,1,4,4,1,0,1,4,1,1,4,2,4,2,0,1,0,4,2,1,1,1,2,4,0,1,2,2,1,1,4,0,3,0,2,2,4,1,2,0,0,0,4,4,4,0,4,1,2,1,1,0,4,3,2,2,4,4,1,3,0,2,4,3,2,3,4,4,1,1,0,1,3,4,4,3,2,4,0,3,2,0,0,0,3,4,0,2,0,2,0,3,0,3,2,3,2,1,2,4,1,1,2,3,0,4,1,1,2,1,1,1,2,3,4,3,0,3,0,0,1,2,4,1,4,3,0,3,4,1,0,2,2,2,1,4,1,1,2,2,3,1,2,0,0,1,1,0,2,4,0,2,0,4,0,4,1,1,0,3,1,1,3,3,4,0,3,2,3,1,3,0,4,1,0,0,3,0,0,4,2,0,3,4,3,4,1,4,2,2,2,2,1,0,3,2,1,2,2,3,0,1,4,0,1,0,0,0,4,0,3,0,3,0,0,3,2,4,4,0,4,3,3,4,4,2,4,3,3,3,0,1,2,0,2,0,1,4,3,0,4,1,4,3,2,4,1,3,0,4,3,0,2,3,3,1,0,4,4,3,3,3,2,2,4,4,2,1,4,0,0,4,4,3,4,2,2,2,4,1,3,3,3,1,2,3,0,4,0,4,0,0,1,2,4,3,1,3,2,2,3,4,0,2,4,1,4,2,2,3,1,3,3,1,1,0,3,1,2,4,1,4,2,1,3,1,0,1,0,2,2,3,0,4,4,3,4,1,3,0,3,3,3,3,2,3,3,1,2,4,0,0,0,0,2,0,1,3,2,1,0,0,0,2,3,3,0,0,0,1,0,4,4,3,1,0,1,0,1,0,3,1,1,4,3,3,3,2,3,1,4,2,0,3,4,0,0,4,3,1,4,3,2,3,3,1,2,3,1,4,3,4,3,1,2,2,4,4,0,3,3,3,4,0,2,2,1,1,4,0,3,0,2,4,1,4,4,1,4,1,2,0,0,0,2,3,3,1,4,0,0,2,4,3,3,0,2,1,2,4,4,0,1,1,2,0,1,1,3,0,3,3,3,1,0,4,2,2,4,2,3,0,1,4,4,4,0,2,3,4,4,3,2,1,3,2,3,2,0,2,3,3,2,2,1,3,4,1,2,0,2,1,2,4,0,3,3,0,0,4,2,0,4,1,3,4,4,3,4,1,2,1,0,1,2,3,4,2,3,1,3,4,3,2,2,3,0,4,2,0,2,2,4,3,1,1,2,4,3,2,3,1,2,2,4,0,3,0,2,4,4,0,2,2,2,2,0,4,2,0,2,4,1,1,3,3,1,2,1,2,2,4,0,1,2,3,1,1,1,3,4,4,2,4]
	kflabel=array(kflabel)
	n=len(kflabel)
	data=getdata(train_file_path)

	print(data[0][0].shape, data[0][1].shape, data[0][2].shape)


	# Build network


	# f=open('structure.pkl','rb')
	# structure=pickle.load(f)
	# f.close()
	

	structure=autoML.WebLayerStructure(stacking_depth=FLAGS.stack_num,
				space_name='SP-our',
				channel_depth=FLAGS.op_depth,
				combination_number=FLAGS.combine_num,)
	autoML.writeStructure('structure.pkl',structure)
	os.system('python plot.py')


	if os.path.exists('parent'):
		f=open('parent','r')
		parent=f.read().strip()
		f.close()
		bestepoc=loadtxt(os.path.join('..',parent,'result-1.txt'))[:,2].argmax()
		epocs=sorted([int('.'.join(i.split('.')[:-1]).replace('Para.ckpt-','')) for i in os.listdir(os.path.join('..',parent,'Para_1',)) if i.endswith('.meta')])
		bestepoc=epocs[bestepoc]
		parentpath=os.path.join(os.path.join('..',parent,'Para_1','Para.ckpt-{}'.format(bestepoc)))
		weights=autoML.parseWeight(parentpath)
		parentstructure=autoML.parseStructure(os.path.join('..',parent,'structure.pkl'))
		for i in range(structure.combination_number):
			for j in range(2):
				if structure.structure[i][j]!=parentstructure.structure[i][j]:
					for k in weights.keys()[:]:
						if k.split('/')[1].find('Inter_{}_{}'.format(i,j+1))!=-1:
							weights.pop(k)
		unused=structure.unused
		unused_old=parentstructure.unused
		for k in weights.keys()[:]:
			if k.split('/')[1].find('Final')!=-1:
				if k.split('/')[2].find('bias')!=-1:
					weights.pop(k)
				elif k.split('/')[2].find('kernel')!=-1:
					newweight=[]
					for j in unused:
						if j in unused_old:
							newweight.append(weights[k][:,:,unused_old.index(j)*structure.channel_depth*3:((unused_old.index(j)+1)*structure.channel_depth*3),:])
						else:
							newweight.append(numpy.random.normal(size=(1,1,structure.channel_depth*3,structure.channel_depth)).clip(min=-2,max=2)*0.1)
					weights[k]=concatenate(newweight,axis=2)
		hasparent=True
	else:
		weights={}
		hasparent=False
	x, y, istrain, realpred, train_step, merged_summary_op=autoML.buildNet(data[0][0].shape[0]+data[0][1].shape[1]*2+data[0][2].shape[2], structure, lr=1e-3, initweight=weights, rncn=True)

	stringlog=''
	crosspart=0
	epicnum = 100 if not hasparent else 100
	savediv,savemod=(8,7) if not hasparent else (5,4)

	for i in range(5)[:1]:
		crosspart+=1
		train_index=(kflabel!=i).nonzero()[0]
		test_index=(kflabel==i).nonzero()[0]

		print("Cross validataion #{}: (Time stamp:{})".format(crosspart,time.time()))
		stringlog+="Cross validataion #{}: (Time stamp:{})\n".format(crosspart,time.time())

		if tf.gfile.Exists(os.path.join(FLAGS.model_dir,'Para_{}/Para.ckpt-{}.index'.format(crosspart,epicnum-1))):
			exit()

		stringresult=''
		stringtp=''
		checkandmake(os.path.join(FLAGS.model_dir,"Para_{}".format(i+1)))

		config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1,)
		config.gpu_options.allow_growth=True
		with tf.Session(config=config) as sess:

			summary_writer_test = tf.summary.FileWriter(os.path.join(FLAGS.model_dir,'logs_{}_test'.format(crosspart)), graph=sess.graph)

			saver = tf.train.Saver(max_to_keep=epicnum+1)

			sess.run(tf.global_variables_initializer())

			for epoc in range(epicnum):
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
													istrain:True})

				if epoc%savediv==savemod:
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

					stringlog+="{:.6f} {:.6f} {:.6f} ".format(prec[argmax(f1)],recall[argmax(f1)],f1[argmax(f1)])
					stringlog+=' '.join(["{:.4f}".format(i) for i in saver1])+'\n'

					stringresult+="{:.6f} {:.6f} {:.6f} ".format(prec[argmax(f1)],recall[argmax(f1)],f1[argmax(f1)])
					stringresult+=' '.join(["{:.4f}".format(i) for i in saver1])+'\n'
					stringtp+=" ".join([str(i) for i in findps+tps+realps])+'\n'
				else:
					pass
					# print('')

		fresult=tf.gfile.Open(os.path.join(FLAGS.model_dir,'result-{}.txt'.format(crosspart)),'wb')
		fresult.write(stringresult)
		fresult.close()
		ftp=tf.gfile.Open(os.path.join(FLAGS.model_dir,'tps-{}.txt'.format(crosspart)),'wb')
		ftp.write(stringtp)
		ftp.close()
	flog=tf.gfile.Open(logpath,'wb')
	flog.write(stringlog)
	flog.close()
	if os.path.exists('pid'):
		os.remove('pid')

if __name__ == '__main__':
	f=open('pid','w')
	f.write(str(hostname)+' '+str(os.getpid()))
	f.close()
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='../../01_Build_TF',help='input data path')
	parser.add_argument('--model_dir', type=str, default='./',help='output model path')
	parser.add_argument('--stack_num',type=int)
	parser.add_argument('--combine_num',type=int)
	parser.add_argument('--op_depth',type=int)

	FLAGS, _ = parser.parse_known_args()
	os.environ["CUDA_VISIBLE_DEVICES"]='6'
	tf.app.run(main=main)
