# -*- coding: utf-8 -*-
import unicodedata

# removes/replaces strange symbols like é
def deal_with_accents(str):
    return unicodedata.normalize('NFD', str)#.encode('ascii', 'ignore')

