# -*- coding: utf-8 -*-

__author__ = 'Wenzhi Mao'

__all__ = ['getOpsBySpaceName']


def getOpsBySpaceName(space_name='SP-I'):
    if space_name == 'SP-I':
        return ['identity', ] + \
               ['sep_3', 'sep_5', 'sep_7', ] + \
               ['ave_3', ] + \
               ['max_3', ] + \
               ['dil-sep_3', ] + \
               ['2step_1x7_7x1', ]
    elif space_name == 'SP-II':
        return ['identity', ] + \
               ['conv_1', 'conv_3', ] + \
               ['sep_3', 'sep_5', 'sep_7', ] + \
               ['ave_2', 'ave_3', ] + \
               ['min_2', ] + \
               ['max_2', 'max_3', ] + \
               ['dil-sep_3', 'dil-sep_5', 'dil-sep_7', ] + \
               ['2step_1x3_3x1', '2step_1x7_7x1', ] + \
               ['dia_3_2', 'dia_3_4', 'dia_3_6', ]
    elif space_name == 'SP-our':
        return ['identity', ] + \
               ['conv_1', 'conv_3', ] + \
               ['sep_3', 'sep_5', 'sep_7', ] + \
               ['dil-sep_3', 'dil-sep_4', 'dil-sep_5', 'dil-sep_7', ] + \
               ['2step_1x3_3x1', '2step_1x5_5x1', '2step_1x7_7x1', ] + \
               ['dia_3_2', 'dia_3_3', 'dia_3_4', 'dia_3_6', ]
    else:
        raise ValueError("Please provide the search space name in ['SP-I', 'SP-II', 'SP-our']")
