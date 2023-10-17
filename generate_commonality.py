import pdb
import sys
from model.lab import Commonality
from figures.maker_copy import PlotResults
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script computing commonality series.')
    parser.add_argument('--L', default=None, help='Time horizon.', type=str)
    parser.add_argument('--transformation', default=None, help='Time horizon.', type=str)
    parser.add_argument('--generate', default=1, type=int, help='Whether to generate table before figures or not.')
    parser.add_argument('--save', default=1, type=int, help='Whether to save figures or not.')
    args = parser.parse_args()
    if bool(args.generate):
        commonality_obj = Commonality(transformation=args.transformation)
        commonality_obj.raw_commonality()
        commonality_obj.commonality()
    else:
        plot_results_obj = PlotResults()
        plot_results_obj.commonality(bool(args.save))
