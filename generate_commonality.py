import pdb
import sys
from model.lab import Commonality
from plots.maker_copy import PlotResults
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script computing commonality series.')
    parser.add_argument('--L', default=None, help='Time horizon.', type=str)
    parser.add_argument('--transformation', default=None, help='Time horizon.', type=str)
    parser.add_argument('--generate', type=int, help='Whether to generate table before plots or not.')
    parser.add_argument('--save', default=1, type=int, help='Whether to save plots or not.')
    args = parser.parse_args()
    if bool(args.generate):
        commonality_obj = Commonality(L=args.L, transformation=args.transformation)
        commonality_obj.commonality()
    else:
        plot_results_obj = PlotResults()
        plot_results_obj.commonality(bool(args.save))
