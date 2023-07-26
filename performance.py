import pdb

import pandas as pd
import subprocess

if __name__ == '__main__':
    performance = pd.read_csv('performance.csv')
    performance = pd.pivot(data=performance,
                           columns=['L', 'training_scheme'],
                           values='values', index=['metric', 'regression', 'model']).round(4)
    ranking = \
        performance.loc[
        performance.index.get_level_values(0)=='qlike', :].transpose().unstack().unstack().dropna().sort_values()
    top_performers = ranking.iloc[:5]
    bottom_performers = ranking.iloc[-5:]
    performance.replace(top_performers.values.tolist(),
                        [f'\\textcolor{{green}}\{{{qlike}}}' for qlike in top_performers.values.tolist()],
                        inplace=True)
    performance.replace(bottom_performers.values.tolist(),
                        [f'\\textcolor{{red}}\{{{qlike}}}' for qlike in bottom_performers.values.tolist()],
                        inplace=True)
    table = performance.to_latex(multicolumn=True, multirow=True)
    subprocess.run("pbcopy", text=True, input=table.replace('NaN', '\\cellcolor{black}'))