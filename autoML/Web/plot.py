# -*- coding: utf-8 -*-

__author__ = 'Wenzhi Mao'

__all__ = ['plotStructure']


def getcolor(name, which):
    if name.startswith('h'):
        return 'white'
    if name == 'mediam':
        return '#dddddd'
    if name.startswith('inter') and name.count('_') == 1:
        return '#16a085'
    if which == 'identity':
        return '#2980b9'
    if which.split('_')[0] == 'conv':
        return '#c0392b'
    if which.split('_')[0] == 'sep':
        return '#7f8c8d'
    if which.split('_')[0] == 'ave':
        return '#8e44ad'
    if which.split('_')[0] == 'min':
        return '#2ecc71'
    if which.split('_')[0] == 'max':
        return '#2c3e50'
    if which.split('_')[0] == 'dil-sep':
        return '#d35400'
    if which.split('_')[0] == '2step':
        return '#bdc3c7'
    if which.split('_')[0] == 'dia':
        return "#f39c12"
    return 'gray'


def easyarrow(patch1, patch2, fig, transf):
    from numpy import array
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig.canvas.draw()
    bb = patch1.get_window_extent(renderer=fig.canvas.renderer)
    bound1 = bb.transformed(transf)
    bb = patch2.get_window_extent(renderer=fig.canvas.renderer)
    bound2 = bb.transformed(transf)
    arrow_kw = dict(
        arrowstyle="Simple,tail_width=0.5,head_width=4,head_length=8", color="#222222", zorder=0)
    pos1 = array(patch1.get_position(), dtype=float)
    pos2 = array(patch2.get_position(), dtype=float)
    if pos1[1] == pos2[1]:
        rad = 0.2
        if pos2[0] > pos1[0]:
            pos1[0] = bound1.x1*(plt.xlim()[1]-plt.xlim()[0])+plt.xlim()[0]
            pos1[1] = (bound1.y0+bound1.y1)/2. * \
                (plt.ylim()[1]-plt.ylim()[0])+plt.ylim()[0]
            pos2[0] = bound2.x0*(plt.xlim()[1]-plt.xlim()[0])+plt.xlim()[0]
            pos2[1] = (bound2.y0+bound2.y1)/2. * \
                (plt.ylim()[1]-plt.ylim()[0])+plt.ylim()[0]
    else:
        rad = 0.3
        if pos2[0] > pos1[0]:
            pos2[0] = bound2.x0*(plt.xlim()[1]-plt.xlim()[0])+plt.xlim()[0]
            pos2[1] = (bound2.y0+bound2.y1)/2. * \
                (plt.ylim()[1]-plt.ylim()[0])+plt.ylim()[0]
        if pos2[1] < pos1[1]:
            pos1[0] = (bound1.x0+bound1.x1)/2. * \
                (plt.xlim()[1]-plt.xlim()[0])+plt.xlim()[0]
            pos1[1] = (bound1.y0)*(plt.ylim()[1]-plt.ylim()[0])+plt.ylim()[0]
        else:
            pos1[0] = (bound1.x0+bound1.x1)/2. * \
                (plt.xlim()[1]-plt.xlim()[0])+plt.xlim()[0]
            pos1[1] = (bound1.y1)*(plt.ylim()[1]-plt.ylim()[0])+plt.ylim()[0]
            rad *= -1
    a = patches.FancyArrowPatch(
        pos1, pos2, connectionstyle="arc3,rad={}".format(rad), **arrow_kw)
    plt.gca().add_patch(a)


def plotStructure(structure):
    from .layer import WebLayerStructure
    from numpy import array
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  

    if not isinstance(structure, WebLayerStructure):
        import pickle
        f = open(structure, 'rb')
        structure = pickle.load(f)
        f.close()

    hiddens = ['h_i-1', 'mediam', 'h_i']
    plotname = [r'$h_{i-1}$', r'...', r'$h_i$']
    poses = [array([0, 0]), array([0.5, 0]), array([1, 0])]
    level = {'h_i-1': 0, 'h_i': 0}
    source = [[], ['h_i-1'], ['mediam']]
    for i in range(len(structure.structure)):
        level['inter_{}'.format(i)] = max(level[structure.structure[i][0][
            0]], level[structure.structure[i][1][0]])+1
    for i in range(len(structure.structure)):
        samelevel = ['inter_{}'.format(j) for j in range(len(structure.structure)) if level[
            'inter_{}'.format(j)] == level['inter_{}'.format(i)]]
        ind = samelevel.index('inter_{}'.format(i))

        poses.append(array([level['inter_{}'.format(i)]+1,
                            (len(samelevel)-1)/2.-ind])+array([-.07, .03]))
        hiddens.append('inter_{}_1'.format(i))
        plotname.append(structure.structure[i][0][1])
        source.append([structure.structure[i][0][0]])

        poses.append(array([level['inter_{}'.format(i)]+1,
                            (len(samelevel)-1)/2.-ind])+array([-.07, -.03]))
        hiddens.append('inter_{}_2'.format(i))
        plotname.append(structure.structure[i][1][1])
        source.append([structure.structure[i][1][0]])

        poses.append(
            array([level['inter_{}'.format(i)]+1, (len(samelevel)-1)/2.-ind]))
        hiddens.append('inter_{}'.format(i))
        plotname.append('+')
        source.append([])

    poses.append(array([max(level.values())+2, 0]))
    hiddens.append('h_i+1')
    plotname.append(r'$h_{i+1}$')
    source.append(['inter_{}'.format(i) for i in range(len(structure.structure)) if len(
        [k for j in structure.structure for k in j if k[0] == 'inter_{}'.format(i)]) == 0])
    boxes = {}

    fig = plt.figure()
    ax = plt.axes([0, 0, 1, 1])
    transf = ax.transAxes.inverted()
    plt.xlim(-1, max(level.values())+3)
    plt.ylim(array(poses)[:, 1].min()-.5, array(poses)[:, 1].max()+.5)
    for i in range(len(hiddens)):
        boxes[hiddens[i]] = ax.text(poses[i][0], poses[i][1], plotname[i], va='center' if not (hiddens[i].startswith('inter') and hiddens[i].count('_') == 2) else ('bottom' if hiddens[i][-1] == '1' else 'top'), ha='center' if not (hiddens[i].startswith(
            'inter') and hiddens[i].count('_') == 2) else 'right', bbox=dict(boxstyle='round' if plotname[i] != '+' else 'circle', facecolor=getcolor(hiddens[i], plotname[i]), alpha=0.7 if (hiddens[i].startswith('inter') and hiddens[i].count('_') == 2) else 1.,))
        for j in source[i]:
            easyarrow(boxes[j], boxes[hiddens[i]], fig, transf)
    fig.savefig('Unit', dpi=300)
    plt.close(fig)
