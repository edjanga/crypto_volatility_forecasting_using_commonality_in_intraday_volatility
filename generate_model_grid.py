import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from data_centre.data import Reader
import itertools
import subprocess
import pdb
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computing model grid.')
    parser.add_argument('--training_scheme', default=False, help='Training scheme.', type=str)
    args = parser.parse_args()
    training_scheme = [args.training_scheme]
    regression = ['LR', 'LASSO', 'Ridge', 'Elastic']
    models = ['AR', 'RiskMetrics', 'HAR', 'HAR-mkt']
    idx = list(itertools.product(training_scheme, regression, models))
    multi_index = pd.MultiIndex.from_tuples(idx, names=['Training scheme', 'Regression', 'Model'])
    model_grid = pd.DataFrame(index=multi_index, columns=['d', 'w', 'm'], data='NaN').transpose()
    model_grid.replace('NaN', '\\bullet', inplace=True)
    subprocess.run("pbcopy", text=True, input=model_grid.to_latex(multirow=True, multicolumn=False))