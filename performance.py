import pdb
import pandas as pd
import subprocess
import plotly.io as pio
pio.kaleido.scope.mathjax = None
import plotly.express as px
import os


if __name__ == '__main__':
    n_performers = 5
    performance = pd.read_csv('performance.csv')
    performance = pd.pivot(data=performance,
                           columns=['L', 'training_scheme'],
                           values='values', index=['metric', 'regression', 'model']).round(4)
    qlike = performance.index.get_level_values(0) == 'qlike'
    not_ridge = ~performance.index.get_level_values(1).isin(['ridge'])
    ranking = \
        performance.loc[qlike&not_ridge, :].transpose().unstack().unstack().dropna().sort_values()
    top_performers = ranking.iloc[:n_performers]
    bottom_performers = ranking.iloc[-n_performers:]
    barplot = \
        pd.concat([top_performers, bottom_performers], axis=0).sort_values(ascending=True).droplevel(axis=0, level=0)
    color_discrete_map = barplot.reorder_levels(['training_scheme', 'L', 'regression', 'model'])
    barplot = \
        pd.DataFrame(data=barplot.values,
                     index=[f'({",".join((idx[-2], idx[-1], idx[0], idx[1]))})' for idx in barplot.index])
    colour_ls = px.colors.qualitative.Plotly[0:2]
    color_discrete_map_dd = {model: colour_ls[0] for model in barplot.index[:n_performers]}
    color_discrete_map_dd.update({model: colour_ls[1] for model in barplot.index[n_performers:]})
    fig = px.bar(barplot.iloc[:, :].reset_index(), x='index', y=0, color='index',
                 color_discrete_map=color_discrete_map_dd)
    fig.update_xaxes(title='models')
    fig.update_yaxes(title='QLIKE')
    fig.update_layout(title='QLIKE: Top and bottom 5 performers', showlegend=False)
    fig.write_image(os.path.abspath('./plots/qlike_performance.pdf'))
    performance.loc[performance.index.get_level_values(0) == 'qlike', :] = \
        performance.loc[performance.index.get_level_values(0) == 'qlike', :].replace(top_performers.values.tolist(),
                                                                                    [f'\\textcolor{{green}}\{{{qlike}}}'
                                                                                    for qlike in
                                                                                    top_performers.values.tolist()])
    performance.loc[performance.index.get_level_values(0) == 'qlike', :] = \
        performance.loc[performance.index.get_level_values(0) == 'qlike', :].replace(bottom_performers.values.tolist(),
                                                                                     [f'\\textcolor{{green}}\{{{qlike}}}'
                                                                                      for qlike in
                                                                                      top_performers.values.tolist()])
    table = performance.to_latex(multicolumn=True, multirow=True)
    subprocess.run("pbcopy", text=True, input=table.replace('NaN', '\\cellcolor{black}'))