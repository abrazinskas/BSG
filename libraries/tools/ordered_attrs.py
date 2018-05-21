from collections import OrderedDict


class OrderedAttrs():
    """
    Makes sure that attributes are stored in the order of assignment. 
    
    """
    def __init__(self):
        self.__dict__ = OrderedDict()

    def __setattr__(self, key, value):
        self.__dict__[key] = value