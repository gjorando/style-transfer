"""
Defines a decorator that we can use to automatically handle our
API exports. Based on an original idea from https://stackoverflow.com/
questions/44834/can-someone-explain-all-in-python#answer-35710527.

@author: gjorando
"""

import sys

__all__ = ["export"]


def export(func):
    """
    Add the function/class to the __all__ array of the module.
    """

    func_module = sys.modules[func.__module__]
    if hasattr(func_module, "__all__"):
        func_module.__all__.append(func.__name__)
    else:
        func_module.__all__ = [func.__name__]

    return func
