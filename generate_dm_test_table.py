import os.path
from model.lab import DMTest
from figures.maker import PlotResults
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pair wise DM tests.')
    parser.add_argument('--L', default=False, help='Lookback windows over which models are trained.',
                        type=str)
    args = parser.parse_args()
    #dm_test_object = DMTest()
    plot_results_obj = PlotResults()
    #dm_table = dm_test_object.table(L=args.L)
    plot_results_obj.dm_test(L=args.L)