# -*- coding: utf-8 -*-

__author__ = 'Wenzhi Mao'

__all__ = ['writeStructure', 'parseStructure', 'parseWeight']


def writeStructure(filepath, layer):
    from .layer import WebLayerStructure
    if not isinstance(layer, WebLayerStructure):
        raise ValueError("Only `WebLayerStructure` class accepted.")
    import pickle
    pickle.dump(layer, open(filepath, 'wb'), 2)


def parseStructure(filepath):
    from os.path import exists, isfile
    if not exists(filepath) or not isfile(filepath):
        raise ValueError("{} is not a valid file path.".format(filepath))
    import pickle
    return pickle.load(open(filepath, 'rb'))

def parseWeight(filepath):
    import tensorflow as tf
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph('{}.meta'.format(filepath))
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        saver.restore(sess, '{}'.format(filepath))
        test=[i for i in graph.get_collection('variables') if not i.name.startswith('train_step')]
        weights={i.name:i.eval() for i in test}
    tf.reset_default_graph()
    return weights