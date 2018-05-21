# A solution to preserve the order of attribute assignment
from collections import OrderedDict


class AttrOrderPreservation:
    def __init__(self, obj):
        self.obj = obj
        self.attr_order = OrderedDict()

    def add_attr(self, name, value):
        self.attr_order[name] = value
        setattr(self.obj,name, value)