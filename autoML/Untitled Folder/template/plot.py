# -*- coding: utf-8 -*-


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
ylim(0)
xlabel('epoc')
ylabel('F1-score')
tight_layout()
legend()
savefig('Cross-F1',dpi=300)


###############################################
import get_structure_fig

get_structure_fig.get_structure_fig('./logs_1_test/')


###############################################


import autoML
autoML.plotStructure('structure.pkl')