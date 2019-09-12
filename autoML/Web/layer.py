# -*- coding: utf-8 -*-
"""This module contains the WebLayerStructure class.
"""

__author__ = 'Wenzhi Mao'

__all__ = ['WebLayerStructure']


class WebLayerStructure(object):

    """A header class for mrc file."""

    def __init__(self, **kwargs):

        self._channel_depth = kwargs.get('channel_depth', 8)
        self._stacking_depth = kwargs.get('stacking_depth', 3)
        self._combination_number = kwargs.get('combination_number', 5)
        self._space_name = kwargs.get('space_name', 'SP-I')
        if 'structure' not in kwargs:
            self.generateRandomStructure()
        else:
            self.sortStructure(kwargs['structure'])

    @property
    def channel_depth(self):
        return self._channel_depth

    @channel_depth.setter
    def channel_depth(self, val):
        if int(val) > 0:
            self._channel_depth = int(val)
        else:
            raise ValueError(
                "Cannot assign {} to channel_depth.".format(channel_depth))

    @property
    def stacking_depth(self):
        return self._stacking_depth

    @stacking_depth.setter
    def stacking_depth(self, val):
        if int(val) > 0:
            self._stacking_depth = int(val)
        else:
            raise ValueError(
                "Cannot assign {} to stacking_depth.".format(stacking_depth))

    @property
    def combination_number(self):
        return self._combination_number

    @property
    def space_name(self):
        return self._space_name

    @property
    def structure(self):
        return self._structure

    @property
    def unused(self):
        return self._unused

    def generateRandomStructure(self):
        from .config import getOpsBySpaceName
        from random import choice
        avaiable_ops = getOpsBySpaceName(self.space_name)
        hiddens = ['h_i', 'h_i-1']
        inters = []
        for i in range(self.combination_number):
            trynode = tuple(((choice(hiddens), choice(avaiable_ops))
                             for j in range(2)))
            while trynode[0] == trynode[1] or trynode[0][0] == trynode[1][0] == 'h_i-1':
                trynode = tuple(((choice(hiddens), choice(avaiable_ops))
                                 for j in range(2)))
            inters.append(trynode)
            hiddens.append('inter_{}'.format(i))
        used_inter = set([j[0] for i in inters for j in i])
        unused_inter = set(hiddens)-used_inter-set(['h_i', 'h_i-1'])
        self._structure = inters
        self._unused = sorted(unused_inter)

    def sortStructure(self, structure):
        if len(structure) != self.combination_number:
            raise ValueError(
                "The structure length doesn't fit the combination_number.")
        if set([len(i) for i in structure]) != set([2]):
            raise ValueError("Each structure element must has 2 inputs.")
        if set([len(j) for i in structure for j in i]) != set([2]):
            raise ValueError(
                "Each input element must has 1 source name and 1 op name.")
        from .config import getOpsBySpaceName
        avaiable_ops = getOpsBySpaceName(self.space_name)
        if not set([j[1] for i in structure for j in i]).issubset(set(avaiable_ops)):
            raise ValueError(
                "There are unrecongnize op name. Must in {} or change to other search space.".format(avaiable_ops))
        hiddens = ['h_i', 'h_i-1']
        for i in range(self.combination_number):
            if structure[i][0][0] not in hiddens:
                raise ValueError("Each inter node could only use previous nodes. inter_{} uses {}.".format(
                    i, structure[i][0][0]))
            if structure[i][1][0] not in hiddens:
                raise ValueError("Each inter node could only use previous nodes. inter_{} uses {}.".format(
                    i, structure[i][1][0]))
            hiddens.append('inter_{}'.format(i))
        used_inter = set([j[0] for i in structure for j in i])
        unused_inter = set(hiddens)-used_inter-set(['h_i', 'h_i-1'])
        self._structure = structure
        self._unused = sorted(unused_inter)

    def mutate(self):
        from random import choice
        newstructure = [[[k for k in j] for j in i] for i in self.structure]
        select_inter = choice(range(self.combination_number))
        ind = choice(range(2))
        which_mutate = choice(range(2))
        if which_mutate == 0:
            hiddens = ['h_i', 'h_i-1'] + ['inter_{}'.format(i) for i in range(select_inter)]
            newstructure[select_inter][ind][0] = choice(hiddens)
            while newstructure[select_inter][ind][0] == self.structure[select_inter][ind][0]:
                newstructure[select_inter][ind][0] = choice(hiddens)
        else:
            from .config import getOpsBySpaceName
            avaiable_ops = getOpsBySpaceName(self.space_name)
            newstructure[select_inter][ind][1] = choice(avaiable_ops)
            while newstructure[select_inter][ind][1] == self.structure[select_inter][ind][1]:
                newstructure[select_inter][ind][1] = choice(avaiable_ops)
        newstructure = [tuple(tuple(k for k in j) for j in i) for i in newstructure]
        return WebLayerStructure(channel_depth=self.channel_depth,
                                 stacking_depth=self.stacking_depth,
                                 combination_number=self.combination_number,
                                 space_name=self.space_name,
                                 structure=newstructure)

