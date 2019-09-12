# -*- coding: utf-8 -*-

import os

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




folders=os.listdir('..')
folders=sorted([i for i in folders if i.startswith('Step') and os.path.isdir('../'+i)])

x=[]
f1=[]
# st=[]
parent=[]
for i in folders:
	if os.path.exists('../{}/result-1.txt'.format(i)):
		f1.append(loadtxt('../{}/result-1.txt'.format(i))[:,2].max())
		x.append(int(i[4:]))
		# f=open('../{}/status'.format(i),'r')
		# st.append(f.read()=='Alive')
		# f.close()
		if os.path.exists('../{}/parent'.format(i)):
			f=open('../{}/parent'.format(i),'r')
			parent.append(int(f.read()[4:]))
			f.close()
		else:
			parent.append(None)
		print x[-1],f1[-1]

# st=array(st)
x=array(x)
f1=array(f1)
fig, ax = plt.subplots()
majorLocator = MultipleLocator(.01)
minorLocator = MultipleLocator(.005)
ax.yaxis.set_major_locator(majorLocator)
ax.yaxis.set_minor_locator(minorLocator)
if len(x)>64:
	scatter(x[-64:],f1[-64:],c='#16a085',zorder=10,s=10)
	scatter(x[:-64],f1[:-64],c='#c0392b',zorder=10,s=10)
else:
	scatter(x,f1,c='#16a085',zorder=10,s=10)
grid('on',alpha=.5,zorder=0,linestyle='--')
grid('on',which='minor',alpha=.3,zorder=0,linestyle='--')
ylim(ylim())
plot([63.5,63.5],ylim(),c='black',zorder=1)
xlabel('Step')
ylabel('F1-score')
tight_layout()
savefig('Trend-Noline',dpi=300)
for i in range(len(x)):
	if parent[i] is not None:
		plot([parent[i],x[i]],[f1[parent[i]],f1[i]],c='#2980b9',zorder=9,alpha=.5)

savefig('Trend',dpi=300)
