from model.lab import DMTest
import argparse
import subprocess
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computing DM test table given one lookback window L.')
    parser.add_argument('--L', default=False, help='Lookback window.', type=str)
    parser.add_argument('--training_scheme', default=False, help='Training scheme.', type=str)
    dm_test_object = DMTest()
    args = parser.parse_args()
    dm_table = dm_test_object.table(L=args.L, training_scheme=args.training_scheme)
    dm_table = dm_table.dropna(how='all').dropna(how='all', axis=1)
    dm_table = dm_table.droplevel(axis=1, level=1).droplevel(axis=0, level=1)
    pdb.set_trace()
    dm_table.fillna('\\cellcolor{black}', inplace=True)
    subprocess.run("pbcopy", text=True, input=dm_table.to_latex(multirow=True, multicolumn=True))