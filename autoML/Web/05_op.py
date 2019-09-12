# -*- coding: utf-8 -*-

import autoML
import os
import numpy
import numpy as np
from numpy import *

import matplotlib
try:
	if not mbio.getMatplotlibDisplay():
		matplotlib.use('Agg')
except:
	matplotlib.use('Agg')
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
plt.rc('axes', prop_cycle=(cycler('color', ['#16a085', '#2980b9','#c0392b','#7f8c8d', '#8e44ad','#2ecc71','#2c3e50','#d35400','#bdc3c7',"#f39c12"])))





path='../../61_AutoML-ourSP-3-8/'
folders=os.listdir(path)
folders=sorted([i for i in folders if i.startswith('Step') and os.path.isdir(path+i)])

folders=[i for i in folders if os.path.exists(path+i+'/structure.pkl')]

ops=['identity', ] + \
   ['conv_1', 'conv_3', ] + \
   ['sep_3', 'sep_5', 'sep_7', ] + \
   ['dil-sep_3', 'dil-sep_4', 'dil-sep_5', 'dil-sep_7', ] + \
   ['2step_1x3_3x1', '2step_1x5_5x1', '2step_1x7_7x1', ] + \
   ['dia_3_2', 'dia_3_3', 'dia_3_4', 'dia_3_6', ]

x=[]
for i in folders:
	loaded=autoML.parseStructure(path+i+'/structure.pkl')
	structure=loaded.structure
	my=[jj[1] for ii in structure for jj in ii]
	x.append([my.count(j) for j in ops])
x=array(x)
segs=[slice(0,64),slice(64,75),]+[slice(i,i+25) for i in range(75,500,25)]
t=array([x[i].mean(axis=0) for i in segs])

for i in range(len(ops)):
	plot(t[:,i],'--' if t[:,i][isfinite(t[:,i])][-1]<.5 else '-',zorder=10)
xticks(range(len(segs)),["{}:{}".format(i.start,i.stop) for i in segs],rotation=30)
xlim(xlim())
ylim(ylim()[0],ylim()[1]+1)
legend(ops,ncol=5,fontsize=7.5,loc='upper center')
ylabel('Average Op number')
xlabel('Generation')
grid(True,	axis='both',linestyle='--',alpha=.5,zorder=0)
tight_layout()
savefig('Proportion',dpi=300)