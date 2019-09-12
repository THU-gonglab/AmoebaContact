'''
Package needs to be installed: BioPython & prody

what is also needed is blast package, install-command recommended by us is "sudo apt-get install ncbi-blast+"

make sure you have got the tensorflow package of the newest version
'''

from __future__ import print_function

import sys

import os

import numpy as np

import Bio.SeqIO

import prody

import tensorflow as tf
from tensorflow.python.platform import gfile

Amoeba    = './'

"Set these before usage"
## You should config your own address of following packages or databases
HHLIB     = 'XX/XX'
CCMPRED   = 'XX/XX'
DEEPCNF   = 'XX/XX'
SPIDER3   = 'XX/XX'
UNIPROT20 = 'XX/XX'


if __name__ == '__main__':


	if len(sys.argv) == 1:
		print('''Usage: AmoebaContact input.fasta output_dir [options]

Options:
	-d DEVICE_ID  : Calculate on CUDA device number DEVICE_ID (set to -1 to use CPU) [default: 0]
	-t THREAD_NUM : Define the thread used for the calculation. [defalut 1]
''')
		exit()

	if len(sys.argv) < 3:
		print('''Usage: AmoebaContact input.fasta output.mat [options]

Options:
	-d DEVICE_ID  : Calculate on CUDA device number DEVICE_ID (set to -1 to use CPU) [default: 0]
	-t THREAD_NUM : Define the thread used for the calculation. [defalut 1]
''')
		raise ValueError('Wrong number of parameters.')


	infile, output_dir = sys.argv[1:3]

	name = os.path.splitext(os.path.split(infile)[1])[0]
	print("Target Name: {0}".format(name))

	DEVICE = '0'
	if len(sys.argv) > 3:
		if '-d' in sys.argv:
			index = sys.argv.index('-d')+1
			if index < len(sys.argv):
				if sys.argv[index] == '-1':
					DEVICE = ''
				else:
					DEVICE = sys.argv[index]
	print('Use CUDE DEVICE: {0}'.format(DEVICE))

	THREAD = '1'
	if len(sys.argv) > 3:
		if '-t' in sys.argv:
			index = sys.argv.index('-t')+1
			if index < len(sys.argv):
				THREAD = sys.argv[index]
	print('Use THREAD number: {0}'.format(THREAD))

	infile = os.path.abspath(infile)

	output_dir = os.path.abspath(output_dir)
	if not os.path.exists(output_dir):
		print('Making Output Directory...')
		os.mkdir(output_dir)

## This part is for target sequence parsing 
	f = open(infile, 'r')
	content = f.read()
	f.close()
	f = open('{}/{}.fasta'.format(output_dir, name), 'w')
	f.write(content)
	f.close()
	infile = '{}/{}.fasta'.format(output_dir, name)
	if not os.path.exists(infile):
		raise ValueError('Cannot read fasta input.')


## Now, feature preparation and feature extraction begins !
	if not os.path.exists('{0}/{1}.a3m'.format(output_dir, name)):
		if not os.path.exists('{0}/bin/hhblits'.format(HHLIB)):
			print('Cannot find hhblits at: {0}/bin/hhblits'.format(HHLIB))
			raise ValueError('HHLIB path wrong.')
		if not os.path.exists('{0}'.format('/'.join(UNIPROT20.split('/')[:-1]))):
			print('Cannot find UNIPROT20 database at: {0}'.format(
				'/'.join(UNIPROT20.split('/')[:-1])))
			raise ValueError('UNIPROT20 path wrong.')
		print('Runing hhblits to get a3m alignment')
		output = os.popen('cd {2};{0}/bin/hhblits -i {1} -oa3m {3}.a3m -all -maxfilt 100000 -realign_max 100000 -B 100000 -Z 100000 -d {4} -cpu {5} 2>&1'.format(
			HHLIB, infile, output_dir, name, UNIPROT20, THREAD)).read()
		if not os.path.exists('{0}/{1}.a3m'.format(output_dir, name)):
			print(output)
			raise ValueError('Cannot generate a3m alignment')
	else:
		print('a3m alignment is already generated.')

	if not os.path.exists('{0}/{1}.filt.a3m'.format(output_dir, name)):
		if not os.path.exists('{0}/bin/hhfilter'.format(HHLIB)):
			print('Cannot find hhfilter at: {0}/bin/hhfilter'.format(HHLIB))
			raise ValueError('HHLIB path wrong.')
		print('Runing hhfilter to filter a3m alignment')
		output = os.popen(
			'export HHLIB={0};{0}/bin/hhfilter -i {1}/{2}.a3m -o {1}/{2}.filt.a3m -id 90 -neff 15 -qsc -30 2>&1'.format(HHLIB, output_dir, name)).read()
		if not os.path.exists('{0}/{1}.filt.a3m'.format(output_dir, name)):
			print(output)
			raise ValueError('Cannot generate a3m filtered alignment')
	else:
		print('filtered a3m alignment is already generated.')

	if not os.path.exists('{0}/{1}-msa.fasta'.format(output_dir, name)):
		if not os.path.exists('{0}/scripts/reformat.pl'.format(HHLIB)):
			print(
				'Cannot find reformat.pl at: {0}/scripts/reformat.pl'.format(HHLIB))
			raise ValueError('HHLIB path wrong.')
		print('Runing reformat.pl to reformat a3m alignment to fasta')
		output = os.popen(
			'export HHLIB={0};{0}/scripts/reformat.pl {1}/{2}.filt.a3m {1}/{2}-msa.fasta -r 2>&1'.format(HHLIB, output_dir, name)).read()
		if not os.path.exists('{0}/{1}-msa.fasta'.format(output_dir, name)):
			print(output)
			raise ValueError('Cannot generate fasta alignment')
	else:
		print('fasta alignment is already generated.')

	if not os.path.exists('{0}/{1}.aln'.format(output_dir, name)):
		print('Converting fasta alignment to aln')
		f_in = open('{0}/{1}-msa.fasta'.format(output_dir, name), "r")
		f_out = open('{0}/{1}.aln'.format(output_dir, name), "w")
		for record in Bio.SeqIO.parse(f_in, 'fasta'):
			f_out.write(str(record.seq))
			f_out.write("\n")
		f_in.close()
		f_out.close()
		if not os.path.exists('{0}/{1}.aln'.format(output_dir, name)):
			raise ValueError('Cannot generate aln alignment')
	else:
		print('aln alignment is already generated.')

	if not os.path.exists('{0}/{1}-ccmpred.result'.format(output_dir, name)):
		if not os.path.exists('{0}/bin/ccmpred'.format(CCMPRED)):
			print('Cannot find ccmpred at: {0}/bin/ccmpred'.format(CCMPRED))
			raise ValueError('CCMPRED path wrong.')
		print('Runing CCMpred')
		if DEVICE:
			output = os.popen("{0}/bin/ccmpred -d {1} {2}/{3}.aln {2}/{3}-ccmpred.result 2>&1".format(
				CCMPRED, DEVICE, output_dir, name)).read()
		else:
			output = os.popen("{0}/bin/ccmpred -t {1} {2}/{3}.aln {2}/{3}-ccmpred.result 2>&1".format(
				CCMPRED, THREAD, output_dir, name)).read()
		if not os.path.exists('{0}/{1}-ccmpred.result'.format(output_dir, name)):
			print(output)
			raise ValueError('Cannot generate CCMpred')
	else:
		print('CCMpred is already generated.')

	if not os.path.exists('{0}/{1}-mi.result'.format(output_dir, name)):
		print('Runing MI')
		msa = prody.parseMSA('{0}/{1}-msa.fasta'.format(output_dir, name))
		mi = prody.buildMutinfoMatrix(msa)
		np.savetxt('{0}/{1}-mi.result'.format(output_dir, name), mi)
		if not os.path.exists('{0}/{1}-ccmpred.result'.format(output_dir, name)):
			raise ValueError('Cannot generate MI')
	else:
		print('MI is already generated.')

	if not (os.path.exists('{0}/{1}.ss3'.format(output_dir, name)) and os.path.exists('{0}/{1}.ss8'.format(output_dir, name))):
		if not os.path.exists('{0}/DeepCNF_SS.sh'.format(DEEPCNF)):
			print(
				'Cannot find DeepCNF_SS.sh at: {0}/DeepCNF_SS.sh'.format(DEEPCNF))
			raise ValueError('DeepCNF path wrong.')
		print('Runing DeepCNF')
		output = os.popen('cd {0};{1}/DeepCNF_SS.sh -i {2} -c {3} 2>&1'.format(
			output_dir, DEEPCNF, infile, THREAD)).read()
		if os.path.exists('{0}/{1}.ss3'.format(output_dir, name)) and os.path.getsize('{0}/{1}.ss3'.format(output_dir, name)) == 0:
			os.remove('{0}/{1}.ss3'.format(output_dir, name))
		if os.path.exists('{0}/{1}.ss8'.format(output_dir, name)) and os.path.getsize('{0}/{1}.ss8'.format(output_dir, name)) == 0:
			os.remove('{0}/{1}.ss8'.format(output_dir, name))
		if not (os.path.exists('{0}/{1}.ss3'.format(output_dir, name)) and os.path.exists('{0}/{1}.ss8'.format(output_dir, name))):
			print(output)
			raise ValueError('Cannot generate DeepCNF')
	else:
		print('DeepCNF is already generated.')

	if not os.path.exists('{0}/{1}.spd33'.format(output_dir, name)):
		if not os.path.exists('{0}/run_list.sh'.format(SPIDER3)):
			print(
				'Cannot find run_list.sh at: {0}/run_list.sh'.format(SPIDER3))
			raise ValueError('SPIDER3 path wrong.')
		print('Runing SPIDER3')
		output = os.popen(
			'cd {0};{1}/run_list.sh {2} 2>&1'.format(output_dir, SPIDER3, infile)).read()
		if not os.path.exists('{0}/{1}.spd33'.format(output_dir, name)):
			print(output)
			raise ValueError('Cannot generate SPIDER3')
	else:
		print('SPIDER3 is already generated.')


	f = open(infile, 'r')
	seq = ''.join([i.strip() for i in f.read().split('\n')[1:]])
	f.close()

	length = len(seq)

	deepcnf1 = np.loadtxt(
		'{0}/{1}.ss3'.format(output_dir, name), dtype=float, usecols=range(3, 6))
	deepcnf2 = np.loadtxt(
		'{0}/{1}.ss8'.format(output_dir, name), dtype=float, usecols=range(3, 11))
	deepcnf = np.hstack([deepcnf1, deepcnf2])

	l2l1hot = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "E": 5, "Q": 6, "G": 7, "H": 8, "I": 9, "L": 10,
			   "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19, "X": 20}
	seq = np.array([l2l1hot[i] for i in seq], dtype=float)
	seql = np.zeros((length, 20))
	seql[seq != 20, seq[seq != 20].astype(int)] = 1.
	seql[seq == 20, :] = 1./20

	l2l = {	"A": [np.arange(0, 1), 1],
			"R": [np.arange(1, 2), 1],
			"N": [np.arange(2, 3), 1],
			"D": [np.arange(3, 4), 1],
			"C": [np.arange(4, 5), 1],
			"E": [np.arange(5, 6), 1],
			"Q": [np.arange(6, 7), 1],
			"G": [np.arange(7, 8), 1],
			"H": [np.arange(8, 9), 1],
			"I": [np.arange(9, 10), 1],
			"L": [np.arange(10, 11), 1],
			"K": [np.arange(11, 12), 1],
			"M": [np.arange(12, 13), 1],
			"F": [np.arange(13, 14), 1],
			"P": [np.arange(14, 15), 1],
			"S": [np.arange(15, 16), 1],
			"T": [np.arange(16, 17), 1],
			"W": [np.arange(17, 18), 1],
			"Y": [np.arange(18, 19), 1],
			"V": [np.arange(19, 20), 1],
			"U": [np.arange(20, 21), 1],
			"O": [np.arange(20, 21), 1],
			"-": [np.arange(20, 21), 1],
			'X': [np.arange(0, 20), 0.05],
			'B': [np.arange(2, 4), 0.5],
			'Z': [np.arange(5, 7), 0.5],
			'J': [np.arange(9, 11), 0.5]}
	f = open('{0}/{1}.aln'.format(output_dir, name), 'r')
	count = 0
	freq = np.zeros((length, 21))
	gap = np.zeros((length, length))
	for line in f:
		line = line.strip()
		if line == '':
			continue
		for j in range(length):
			place, addv = l2l[line[j]]
			freq[j, place] += addv
		temp = np.array(list(line)) == '-'
		gap[np.ix_(temp, temp)] += 1
		count += 1
	freq /= count
	gap /= count
	f.close()

	spd = np.loadtxt('{0}/{1}.spd33'.format(output_dir, name),
					 dtype=float, usecols=range(3, 13))

	ccmpred = np.loadtxt('{0}/{1}-ccmpred.result'.format(output_dir, name))

	mi = np.loadtxt('{0}/{1}-mi.result'.format(output_dir, name))

	pos = abs(np.arange(length)[np.newaxis]-np.arange(length)[:, np.newaxis])

	f0d = np.array([length, count]).astype(np.float32)
	f1d = np.concatenate((deepcnf, seql, freq, spd), axis=1).astype(np.float32)
	f2d = np.stack((ccmpred, mi, gap, pos), axis=2).astype(np.float32)

	featurefeed2d = np.concatenate([
		f2d,
		np.tile(f1d[np.newaxis], [length, 1, 1]),
		np.tile(f1d[:, np.newaxis], [1, length, 1]),
		np.tile(f0d[np.newaxis, np.newaxis], [length, length, 1]),
	], axis=2)[np.newaxis]

	index = np.arange(length-3)[:, np.newaxis]+np.arange(4)[np.newaxis]
	features = f1d[index].reshape((length-3, f1d.shape[1]*4))
	featurefeed1d = np.concatenate(
		[features, np.tile(f0d[np.newaxis], [length-3, 1])], axis=1)

## Now, the computation of AmoebaContact begins.
	os.environ["CUDA_VISIBLE_DEVICES"]=''
	config = tf.ConfigProto()
	# config.gpu_options.allow_growth=True
	cutoff = [6, 7, 8, 9, 10, 12, 14, 16, 20]
	ensembel_num = [9, 8, 14, 13, 11, 11, 12, 11, 10]
	for i in range(len(cutoff)):
		if os.path.exists('{0}/{1}-{2}.result'.format(output_dir,name,cutoff[i])):
			continue
		scores=None
		for j in range(ensembel_num[i]):
			print('\rPredicting cutoff {0} {1:2}/{2:2}'.format(cutoff[i],j+1,ensembel_num[i]),end='')
			sys.stdout.flush()
			tf.reset_default_graph()
			with tf.Session(config=config) as sess:
				with gfile.GFile('{}/models/Cutoff{}-{}.pb'.format(Amoeba,cutoff[i],j+1), 'rb') as f:
					graph_def = tf.GraphDef()
					graph_def.ParseFromString(f.read())
					sess.graph.as_default()
					tf.import_graph_def(graph_def, name='')
				g = tf.get_default_graph()
				o=g.get_tensor_by_name('Prediction:0')
				x=g.get_tensor_by_name('Input:0')
				pred=sess.run(o,feed_dict={x:featurefeed2d})[0,:,:,0]
				pred=(pred+pred.T)/2.
				if scores is None:
					scores=pred
				else:
					scores+=pred
		scores=scores/ensembel_num[i]
		np.savetxt('{0}/{1}-{2}.result'.format(output_dir,name,cutoff[i]),scores)

	if not os.path.exists('{0}/{1}-local.result'.format(output_dir,name)):
		print('\rPredicting local',end='')
		sys.stdout.flush()
		tf.reset_default_graph()
		with tf.Session(config=config) as sess:
			with gfile.GFile('{}/models/Local.pb'.format(Amoeba), 'rb') as f:
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())
				sess.graph.as_default()
				tf.import_graph_def(graph_def, name='')
			g = tf.get_default_graph()
			o=g.get_tensor_by_name('Prediction:0')
			x=g.get_tensor_by_name('Input:0')
			pred=sess.run(o,feed_dict={x:featurefeed2d})[0,:,:,0]
			pred=(pred+pred.T)/2.
			np.savetxt('{0}/{1}-local.result'.format(output_dir,name),pred)

	if not os.path.exists('{0}/{1}-helix.result'.format(output_dir,name)):
		print('\rPredicting helix',end='')
		sys.stdout.flush()
		tf.reset_default_graph()
		with tf.Session(config=config) as sess:
			with gfile.GFile('{}/models/Helix.pb'.format(Amoeba), 'rb') as f:
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())
				sess.graph.as_default()
				tf.import_graph_def(graph_def, name='')
			g = tf.get_default_graph()
			o=g.get_tensor_by_name('Prediction:0')
			x=g.get_tensor_by_name('Input:0')
			pred=sess.run(o,feed_dict={x:featurefeed1d})[:,0]
			np.savetxt('{0}/{1}-helix.result'.format(output_dir,name),pred)

	print()
## end of the program
