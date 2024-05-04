import os.path
from model.lab import DMTest
from figures.maker import PlotResults
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pair wise DM tests.')
    parser.add_argument('--test', default=0, type=int, help='Test or not.')
    args = parser.parse_args()
    dm_test_object = DMTest()
    for L in ['1W', '1M', '6M']:
        dm_test_object.L = L
        print(dm_test_object.table(test=args.test))
