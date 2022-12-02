'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import ModelLoader
import global_utils
from ptflops import get_model_complexity_info
from models import *
def main(argv):
    model = ResNet50()
    flops, params = get_model_complexity_info(model, (3, 32, 32),
                                              as_strings=False,
                                              print_per_layer_stat=True)
    print('Flops:  {:4g}'.format(flops))
    print('Params: {:4g}'.format(params))

if __name__ == "__main__":
  
    main(sys.argv)
