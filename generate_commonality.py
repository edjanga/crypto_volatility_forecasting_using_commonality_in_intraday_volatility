import pdb
import sys
from model.lab import Commonality
from figures.maker import PlotResults
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script computing commonality series.')
    parser.add_argument('--L', default=None, help='Time horizon.', type=str)
    parser.add_argument('--transformation', default=None, help='Time horizon.', type=str)
    parser.add_argument('--commonality_type', default='adjusted_r2', type=str, help='Type of commonality to produce.')
    args = parser.parse_args()
    commonality_obj = Commonality(transformation=args.transformation, type_commonality=args.commonality_type)
    commonality_obj.commonality()