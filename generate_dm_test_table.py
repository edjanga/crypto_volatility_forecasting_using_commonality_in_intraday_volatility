from model.lab import DMTest
import subprocess
import pdb
import numpy as np

if __name__ == '__main__':
    dm_test_object = DMTest()
    dm_table = dm_test_object.table()
    dm_table = dm_table.dropna(how='all').dropna(how='all', axis=1)
    dm_table = dm_table.droplevel(axis=1, level=1).droplevel(axis=0, level=1)
    #percentage = np.sign(dm_table).apply(lambda x: x.value_counts(True), axis=1).mean()
    #print((-1, 1), (percentage.values[0], percentage.values[1]))
    #pdb.set_trace()
    dm_table.fillna('\\cellcolor{black}', inplace=True)
    print(dm_table.to_latex(multirow=True, multicolumn=True))
    #subprocess.run("pbcopy", text=True, input=dm_table.to_latex(multirow=True, multicolumn=True))