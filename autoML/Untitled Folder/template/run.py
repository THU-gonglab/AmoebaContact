# -*- coding: utf-8 -*-

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



def getdata(file_path):
	data=[]
	for serialized_example in tf.python_io.tf_record_iterator(file_path):
		example = tf.train.Example()
		example.ParseFromString(serialized_example)

		index = example.features.feature['index'].int64_list.value[0]
		length = example.features.feature['length'].int64_list.value[0]
		ccmpred = fromstring(example.features.feature['ccmpred'].bytes_list.value[0])
		ccmpred.resize((length,length))
		deepcnf = fromstring(example.features.feature['deepcnf'].bytes_list.value[0])
		deepcnf.resize((length,3))
		label = fromstring(example.features.feature['label'].bytes_list.value[0])
		label.resize((length,length))
		if not length==ccmpred.shape[0]==ccmpred.shape[1]==deepcnf.shape[0]==label.shape[0]==label.shape[1]:
			raise ValueError('Wrong shape')
		deepcnf2d=concatenate((deepcnf[newaxis,:,:]*ones((length,1,1)),deepcnf[:,newaxis,:]*ones((1,length,1))),axis=2)
		pos=abs(arange(length)[newaxis]-arange(length)[:,newaxis])
		feature2d=concatenate((ccmpred[:,:,newaxis],deepcnf2d,pos[:,:,newaxis]),axis=2)
		data.append([feature2d[newaxis],label[newaxis,:,:,newaxis]])
	return data



def main(_):
	train_file_path = os.path.join(FLAGS.data_dir, "train.tfrecords")
	logpath = os.path.join(FLAGS.model_dir, "log.txt")

	kflabel=[0,3,3,0,4,0,3,3,4,3,3,1,2,4,3,1,1,1,3,0,0,1,4,4,3,2,1,0,4,1,1,3,0,3,1,4,3,4,1,3,1,2,0,0,3,0,0,3,3,3,2,3,0,2,1,1,1,2,2,0,3,2,3,0,0,4,0,3,4,4,0,0,1,3,4,0,0,0,3,4,4,1,2,1,3,4,3,4,4,0,1,3,3,0,3,0,3,0,2,0,2,1,2,4,0,4,3,0,3,4,1,3,4,4,0,2,0,0,3,1,1,3,4,1,1,3,4,2,0,0,4,2,2,3,3,2,2,0,2,3,3,4,4,4,2,2,3,3,0,1,3,0,4,3,0,3,4,0,2,1,2,0,1,3,1,0,1,0,1,4,3,0,0,3,3,4,4,3,4,2,1,1,3,0,2,3,0,0,3,1,3,2,2,4,3,2,2,3,4,4,3,1,0,2,4,1,1,3,3,4,1,0,2,3,0,4,0,3,0,0,4,0,1,4,4,2,3,2,2,3,2,4,0,3,1,2,1,0,1,1,4,0,0,1,4,4,0,3,4,3,0,1,3,3,2,1,1,2,4,1,4,1,1,2,4,1,0,1,0,3,1,0,4,0,4,4,4,1,3,0,0,3,2,2,4,1,4,2,1,3,1,1,1,4,2,4,0,4,0,0,2,0,4,2,1,2,2,1,0,4,0,2,3,2,0,2,3,1,2,0,4,1,3,0,1,4,1,0,4,1,1,0,3,0,1,0,3,4,0,2,2,3,2,3,4,4,4,0,4,1,1,0,3,1,2,0,1,4,1,3,0,0,3,0,0,2,0,0,2,3,2,1,2,1,2,1,4,2,4,2,4,1,2,2,2,2,0,1,4,2,1,1,4,4,2,4,1,2,2,2,1,2,0,3,0,0,4,1,0,4,4,1,0,2,0,0,0,0,2,2,3,2,3,2,4,4,1,0,3,1,0,0,3,1,2,2,1,2,2,4,0,1,2,2,4,4,1,3,4,3,4,3,1,2,1,2,0,1,0,4,0,0,1,0,2,3,4,2,3,2,3,4,1,3,4,4,1,4,2,4,1,3,0,4,1,1,3,0,3,1,4,4,2,4,4,1,3,0,3,1,0,2,2,0,0,3,1,1,4,0,1,4,1,4,4,1,1,1,4,4,2,3,2,0,4,3,0,0,3,0,2,4,2,2,0,3,0,4,0,3,2,4,2,1,2,3,1,4,3,4,1,0,2,2,2,0,2,0,1,1,4,3,2,0,3,0,1,2,0,2,0,0,3,3,4,0,3,0,3,1,2,2,4,0,1,1,3,3,0,4,2,4,4,1,4,3,4,2,2,4,1,1,1,3,0,2,1,2,4,2,4,3,4,2,4,3,3,1,3,0,0,2,2,0,2,0,2,4,4,2,3,1,2,2,3,0,2,3,2,3,0,0,2,4,1,0,0,0,0,2,1,1,1,4,0,0,2,3,4,3,3,3,0,4,4,2,3,1,2,3,1,4,0,3,1,0,3,2,4,0,4,0,4,4,0,2,4,2,2,3,3,2,3,0,3,4,4,2,3,3,1,2,3,4,4,4,2,4,1,1,3,0,2,4,2,0,4,2,1,3,3,3,4,1,4,4,3,0,0,4,4,0,0,0,2,0,0,3,1,2,4,3,2,2,0,1,3,0,0,3,3,3,1,1,4,2,3,2,2,2,1,2,1,0,0,1,3,2,2,2,1,4,3,0,1,0,2,1,2,3,2,4,4,1,4,0,0,3,1,2,4,2,0,2,3,1,4,2,1,1,0,1,0,2,4,1,0,4,4,3,1,0,1,3,3,1,3,2,0,2,1,2,4,2,1,4,3,4,3,2,1,2,3,0,3,1,1,1,2,3,1,4,3,1,0,3,0,3,3,3,1,3,1,2,2,3,1,3,4,3,4,0,2,4,3,4,2,1,0,1,1,4,4,3,4,2,2,4,3,3,2,2,2,1,2,3,4,2,2,3,3,3,1,0,2,1,4,0,1,4,2,0,4,3,0,2,2,0,3,0,4,4,0,1,3,4,4,1,1,4,1,2,0,2,4,1,2,0,4,1,4,1,1,3,1,3,3,0,3,4,1,2,3,2,3,1,0,4,3,4,3,2,0,4,1,2,1,4,4,4,3,2,1,0,4,3,0,0,4,0,3,4,0,4,0,0,4,3,3,2,2,0,2,2,1,4,3,2,2,3,4,3,1,3,0,1,4,2,3,0,2,3,0,0,2,3,1,4,2,1,1,1,2,1,2,3,3,0,0,1,3,0,2,0,0,0,0,4,3,3,0,2,2,1,2,2,0,0,0,4,2,4,1,3,3,3,3,4,1,1,2,3,4,1,0,4,3,0,4,4,3,0,4,0,4,2,0,3,4,0,4,2,1,4,0,1,3,0,4,1,3,3,0,3,0,2,1,3,1,0,1,2,4,4,4,2,1,1,4,2,3,2,1,0,4,3,2,4,1,0,0,3,0,3,1,0,0,0,4,1,1,3,3,0,4,2,1,3,3,1,4,3,1,1,2,2,3,4,0,0,2,1,2,1,1,2,0,2,4,2,3,3,0,1,0,2,3,2,4,3,3,4,4,0,1,0,0,0,3,1,1,4,4,3,2,3,0,2,0,1,1,1,4,1,0,1,3,4,3,0,0,4,2,3,0,4,4,2,1,3,2,2,2,3,2,2,2,2,3,1,2,0,2,1,4,0,0,4,2,4,0,0,3,2,4,4,2,1,3,4,0,2,4,1,1,2,2,1,3,3,2,4,0,3,4,0,4,3,1,0,1,0,0,3,2,4,1,3,4,1,4,0,0,1,1,2,4,1,2,4,2,1,3,4,0,1,4,3,1,1,0,0,4,1,0,1,2,2,2,0,3,4,4,3,2,2,1,4,4,4,0,0,3,4,4,1,1,2,4,1,0,1,2,4,3,4,0,4,4,3,1,0,2,1,4,4,0,1,3,4,2,2,2,2,3,0,2,0,4,4,2,1,3,1,3,1,1,1,3,2,3,3,1,0,1,4,2,4,3,0,0,2,4,0,1,1,2,4,4,4,4,0,4,2,1,3,4,0,3,1,4,3,0,0,3,4,3,3,2,2,2,3,4,1,3,2,3,4,1,3,4,3,1,2,4,0,3,4,2,4,4,3,3,0,3,4,2,1,1,4,2,2,2,0,2,2,1,2,2,0,2,4,0,4,3,4,3,0,1,3,0,1,2,4,2,1,0,4,3,2,2,2,0,2,0,1,3,0,4,0,4,1,1,3,4,4,4,4,0,0,3,4,1,3,2,0,1,4,4,4,4,2,0,3,1,3,0,1,4,3,1,3,0,1,2,2,2,2,3,3,2,2,0,1,0,3,2,1,0,1,4,0,1,4,4,0,3,3,2,0,3,1,4,0,3,1,0,1,0,3,0,3,4,1,4,4,4,2,3,1,1,0,0,0,3,2,2,1,1,2,3,2,4,0,0,3,1,2,1,4,2,3,0,1,0,1,4,2,1,3,1,1,3,2,1,3,0,3,4,3,3,1,2,3,2,2,2,0,3,3,0,3,4,3,0,4,0,0,2,4,0,1,1,2,1,3,1,3,3,1,3,0,0,1,3,1,2,2,1,3,1,3,1,3,4,3,1,2,0,2,3,2,3,4,2,0,4,3,1,3,4,2,3,0,4,0,0,3,1,1,4,2,2,1,4,0,2,4,0,2,1,3,3,3,0,1,1,2,2,2,3,0,4,3,2,1,0,1,4,3,1,2,3,0,1,0,2,3,1,0,1,0,3,1,4,4,3,4,4,4,4,0,0,4,4,0,1,4,2,4,2,2,2,4,2,1,1,2,0,0,1,0,3,4,4,0,1,0,4,3,2,1,2,0,1,4,1,3,3,2,0,0,3,0,2,3,4,3,2,0,3,2,2,2,2,1,3,0,4,3,0,1,1,2,4,1,2,0,0,1,1,4,4,0,1,1,4,3,3,3,1,4,2,0,3,2,1,0,1,3,2,1,0,1,1,3,2,0,0,4,4,0,2,1,1,3,2,3,2,2,1,4,0,3,0,1,1,4,3,4,3,4,3,1,3,4,4,3,1,4,2,2,4,1,3,1,1,3,1,2,3,0,3,2,4,1,1,4,1,0,2,1,2,2,2,3,1,2,2,2,4,3,1,4,0,3,0,0,0,0,0,3,2,3,2,3,4,4,4,0,1,2,0,2,4,1,4,4,2,0,3,2,3,3,0,2,4,0,1,1,0,3,0,4,3,3,0,1,4,4,4,3,4,1,1,1,0,3,2,1,2,0,2,2,2,1,0,2,0,4,4,3,0,2,1,0,1,3,3,3,1,2,3,1,2,4,1,1,2,1,4,1,0,4,2,1,1,3,0,3,1,2,1,1,0,3,3,2,3,2,2,3,3,2,4,3,1,1,4,4,3,1,0,4,4,3,4,3,3,3,3,1,2,4,0,4,1,3,4,4,0,3,4,0,1,1,3,3,0,0,4,1,2,2,1,3,2,4,1,2,2,0,4,4,0,0,3,3,2,3,4,0,4,2,3,0,0,3,2,0,1,3,0,1,2,0,2,1,1,0,4,0,2,0,4,2,4,4,3,2,1,4,4,0,4,2,1,2,0,0,4,3,2,1,1,3,2,4,2,2,1,1,2,1,0,3,2,0,4,0,1,1,1,3,4,4,1,4,2,4,3,0,0,1,2,2,4,0,0,0,1,3,1,2,4,0,4,3,4,3,2,4,3,4,3,4,2,3,1,0,0,4,1,1,3,2,4,3,4,4,2,2,2,3,2,2,0,0,3,1,0,0,1,2,0,0,3,2,4,2,2,1,2,2,2,0,3,1,0,0,3,0,4,2,1,4,1,2,1,4,2,3,1,4,3,1,2,3,2,2,1,3,0,3,0,2,1,2,1,3,3,3,3,0,3,0,3,3,1,1,2,0,2,0,0,4,3,2,2,4,4,2,3,2,4,3,1,1,4,1,0,3,4,0,0,1,2,0,1,0,2,4,0,4,0,1,0,3,4,1,2,4,4,2,0,2,4,1,3,0,4,0,2,3,0,3,3,1,3,0,1,1,1,4,1,1,4,3,2,4,2,0,0,1,2,2,1,1,0,0,1,2,2,2,3,2,0,1,1,4,4,2,2,4,4,3,1,3,0,1,0,1,4,4,1,3,2,4,0,1,1,1,2,1,3,2,2,4,4,1,0,0,4,1,4,3,0,2,0,3,4,4,4,4,0,2,4,2,2,4,4,4,3,1,2,4,3,1,0,0,2,1,2,1,4,1,4,0,0,4,0,0,3,0,1,1,1,3,1,1,0,3,4,3,0,1,4,3,1,1,4,1,4,4,1,0,1,4,1,1,4,2,4,2,0,1,0,4,2,1,1,1,2,4,0,1,2,2,1,1,4,0,3,0,2,2,4,1,2,0,0,0,4,4,4,0,4,1,2,1,1,0,4,3,2,2,4,4,1,3,0,2,4,3,2,3,4,4,1,1,0,1,3,4,4,3,2,4,0,3,2,0,0,0,3,4,0,2,0,2,0,3,0,3,2,3,2,1,2,4,1,1,2,3,0,4,1,1,2,1,1,1,2,3,4,3,0,3,0,0,1,2,4,1,4,3,0,3,4,1,0,2,2,2,1,4,1,1,2,2,3,1,2,0,0,1,1,0,2,4,0,2,0,4,0,4,1,1,0,3,1,1,3,3,4,0,3,2,3,1,3,0,4,1,0,0,3,0,0,4,2,0,3,4,3,4,1,4,2,2,2,2,1,0,3,2,1,2,2,3,0,1,4,0,1,0,0,0,4,0,3,0,3,0,0,3,2,4,4,0,4,3,3,4,4,2,4,3,3,3,0,1,2,0,2,0,1,4,3,0,4,1,4,3,2,4,1,3,0,4,3,0,2,3,3,1,0,4,4,3,3,3,2,2,4,4,2,1,4,0,0,4,4,3,4,2,2,2,4,1,3,3,3,1,2,3,0,4,0,4,0,0,1,2,4,3,1,3,2,2,3,4,0,2,4,1,4,2,2,3,1,3,3,1,1,0,3,1,2,4,1,4,2,1,3,1,0,1,0,2,2,3,0,4,4,3,4,1,3,0,3,3,3,3,2,3,3,1,2,4,0,0,0,0,2,0,1,3,2,1,0,0,0,2,3,3,0,0,0,1,0,4,4,3,1,0,1,0,1,0,3,1,1,4,3,3,3,2,3,1,4,2,0,3,4,0,0,4,3,1,4,3,2,3,3,1,2,3,1,4,3,4,3,1,2,2,4,4,0,3,3,3,4,0,2,2,1,1,4,0,3,0,2,4,1,4,4,1,4,1,2,0,0,0,2,3,3,1,4,0,0,2,4,3,3,0,2,1,2,4,4,0,1,1,2,0,1,1,3,0,3,3,3,1,0,4,2,2,4,2,3,0,1,4,4,4,0,2,3,4,4,3,2,1,3,2,3,2,0,2,3,3,2,2,1,3,4,1,2,0,2,1,2,4,0,3,3,0,0,4,2,0,4,1,3,4,4,3,4,1,2,1,0,1,2,3,4,2,3,1,3,4,3,2,2,3,0,4,2,0,2,2,4,3,1,1,2,4,3,2,3,1,2,2,4,0,3,0,2,4,4,0,2,2,2,2,0,4,2,0,2,4,1,1,3,3,1,2,1,2,2,4,0,1,2,3,1,1,1,3,4,4,2,4]
	kflabel=array(kflabel)
	n=len(kflabel)
	data=getdata(train_file_path)


	# Build network
	f=open('structure.pkl','rb')
	structure=pickle.load(f)
	f.close()
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
	x, y, istrain, realpred, train_step, merged_summary_op=autoML.buildNet(8, structure, lr=1e-3, initweight=weights, rncn=True)

	stringlog=''
	crosspart=0
	epicnum = 100 if not hasparent else 100
	savediv,savemod=(5,4) if not hasparent else (5,4)

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
					featurefeed,labelfeed=data[i]
					# print('\r\tTraining {:4d}, {:4d}/{:4d}'.format(i,casecount,len(trainrank)),end='')
					# sys.stdout.flush()
					_=sess.run(train_step,feed_dict={x: featurefeed, y: labelfeed, istrain:True, })

				if epoc%savediv==savemod:
					saver.save(sess, os.path.join(FLAGS.model_dir,"Para_{}/Para.ckpt".format(crosspart)), global_step=epoc)

					testreal=[]
					testpred=[]
					testrank=test_index.copy()
					casecount=0
					for i in testrank:
						casecount+=1
						featurefeed,labelfeed=data[i]
						# print('\r\tTesting  {:4d}, {:4d}/{:4d}'.format(i,casecount,len(test_index)),end='')
						# sys.stdout.flush()
						predresult,summary_str=sess.run([realpred,merged_summary_op],feed_dict={x: featurefeed, y: labelfeed,istrain:False,})
						summary_writer_test.add_summary(summary_str, epoc)

						testreal.append(labelfeed[0,:,:,0])
						testpred.append((predresult[0,:,:,0]+predresult[0,:,:,0].T)/2.)

					testreal=hstack([i.flatten() for i in testreal])
					testpred=hstack([i.flatten() for i in testpred])

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
					# print('')
					print("{:.6f} {:.6f} {:.6f}".format(prec[argmax(f1)],recall[argmax(f1)],f1[argmax(f1)]))
					stringlog+="{:.6f} {:.6f} {:.6f}\n".format(prec[argmax(f1)],recall[argmax(f1)],f1[argmax(f1)])

					stringresult+="{:.6f} {:.6f} {:.6f}\n".format(prec[argmax(f1)],recall[argmax(f1)],f1[argmax(f1)])
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
	f.write(str(os.getpid()))
	f.close()
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='../../20_Run_Direct/00_Build_TF',help='input data path')
	parser.add_argument('--model_dir', type=str, default='./',help='output model path')
	FLAGS, _ = parser.parse_known_args()
	os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
	tf.app.run(main=main)