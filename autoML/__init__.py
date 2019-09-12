# -*- coding: utf-8 -*-

__author__ = 'Wenzhi Mao'
__version__ = '0.0.1'

__release__ = [int(x) for x in __version__.split('.')]
del x
__all__ = []

from platform import system
from sys import version_info

_system = system()
del system

_PY3K = version_info[0] > 2
_PY2K = not _PY3K
del version_info


from . import Web
from .Web import *
__all__.extend(Web.__all__)
__all__.append('Web')
