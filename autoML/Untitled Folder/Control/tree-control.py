# -*- coding: utf-8 -*-
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


folders=sorted([i for i in os.listdir('..') if i.startswith('Step')])
folders=['../'+i for i in folders if os.path.isdir('../'+i)]

parent=[]
f1=[]
for i in folders:
	if os.path.exists('{}/parent'.format(i)):
		f=open('{}/parent'.format(i),'r')
		parent.append(f.read())
		f.close()
	else:
		parent.append(None)
	if os.path.exists('{}/result-1.txt'.format(i)):
		f1.append(loadtxt('{}/result-1.txt'.format(i))[:,2].max())
	else:
		f1.append(None)

folders=[i[3:] for i in folders]

q=[]
for i in range(len(folders)):
	if parent[i] is None:
		q.append(i)
p=0
son={}
while p<len(q):
	son[q[p]]=[j for j in range(len(folders)) if parent[j]==folders[q[p]]]
	q+=son[q[p]]
	p+=1

bases=[i for i in range(len(folders)) if parent[i] is None]
width={}
def find(x):
	if son[x]==[]:
		width[x]=1
		return 1
	width[x]=sum([find(i) for i in son[x]])
	return width[x]
for i in range(len(bases)):
	find(i)
left=0
poses=[None for i in folders]

def setpos(ind,shift,y):
	poses[ind]=[shift+width[ind]/2.,y]
	addshift=0
	for i in son[ind]:
		setpos(i,shift+addshift,y+1)
		addshift+=width[i]

for i in bases:
	setpos(i,left,0)
	left+=width[i]
poses=array(poses)
fig=figure(figsize=(6*(poses[:,0].max()-poses[:,0].min()+2)/(poses[:,1].max()-poses[:,1].min()+2)/5,6))
ax=axes([0,0,1,1])
poses[:,1]*=-1
f1=array(f1,dtype=float)

best3=set(argsort(f1[isfinite(f1)])[-3:])

scatter(poses[isfinite(f1)][:,0],poses[isfinite(f1)][:,1],zorder=10,linewidths=0.5,edgecolors=['k' if i not in best3 else 'red' for i in range(sum(isfinite(f1)))],c=f1[isfinite(f1)],vmin=max(f1[isfinite(f1)])-0.03,vmax=max(f1[isfinite(f1)]),cmap='Wistia',s=60*((f1[isfinite(f1)]-f1[isfinite(f1)].min())/(f1[isfinite(f1)].max()-f1[isfinite(f1)].min()))**4)
scatter(poses[~isfinite(f1),0],poses[~isfinite(f1),1],zorder=10,linewidths=0.5,edgecolors=['k']*len(poses),c='white')
for i in range(len(folders)):
	if parent[i] is not None:
		plot([poses[folders.index(parent[i])][0],poses[i][0]],[poses[folders.index(parent[i])][1],poses[i][1]],c='k',lw=1)
	text(poses[i,0],poses[i,1],str(i),fontsize=2,va='center',ha='center',zorder=200)
	# if len(son[i])>=4:
	# 	text(poses[i,0]-.5,poses[i,1],str(i),fontsize=3,va='center',ha='right')
	# else:
	# 	text(poses[i,0],poses[i,1]-.1,str(i),fontsize=3,va='top',ha='center')
xlim(poses[:,0].min()-1,poses[:,0].max()+1)
ylim(poses[:,1].min()-1,poses[:,1].max()+1)
majorLocator = MultipleLocator(1)
ax.yaxis.set_major_locator(majorLocator)
grid('on',alpha=.5,zorder=0,linestyle='--',axis='y')
savefig('Tree',dpi=300)
