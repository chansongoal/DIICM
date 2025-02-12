#!/usr/bin/env python
# coding: utf-8
from tidecv import TIDE
import tidecv.datasets as datasets
from matplotlib import pyplot as plt
import argparse

def tide_eval(args):
    json_file = args.json_file

    gt = datasets.COCO('/model/gaocs/Detectron2/json/instances_minVal2014_jpg.json')

    tide = TIDE()

    if args.ap_mode == 'bboxdAp':
        bbox_results = datasets.COCOResult(json_file) # These files were downloaded above.
        tide.evaluate_range(gt, bbox_results, mode=TIDE.BOX ) # Several options are available here, see the functions
    if args.ap_mode == 'segmdAp':
        mask_results = datasets.COCOResult(json_file) # Replace them with your own in practice.
        tide.evaluate_range(gt, mask_results, mode=TIDE.MASK) # evaluate and evaluate_range for more details.

    tide.summarize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arguments for configration')
    parser.add_argument('--json_file', '-dn', help='json_file')
    parser.add_argument('--ap_mode', '-c', help='ap_mode')
    args = parser.parse_args()
    tide_eval(args)