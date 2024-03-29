import pdb
import pandas as pd
import subprocess
import plotly.io as pio
pio.kaleido.scope.mathjax = None
import plotly.express as px
import os


if __name__ == '__main__':
    n_performers = 5
    performance = pd.read_csv('../results/performance.csv')
    performance = pd.pivot(data=performance,
                           columns=['L', 'training_scheme'],
                           values='values', index=['metric', 'regression', 'model', 'vol_regime']).round(4)
    qlike = performance.index.get_level_values(0) == 'qlike'
    ranking = performance.loc[qlike, :].transpose().unstack().unstack().dropna().sort_values()
    top_performers = ranking.groupby(by=pd.Grouper(level='vol_regime')).apply(lambda x: x.iloc[:n_performers])
    bottom_performers = ranking.groupby(by=pd.Grouper(level='vol_regime')).apply(lambda x: x.iloc[-n_performers:])
    top_performers = pd.DataFrame(top_performers).assign(performer='top')
    bottom_performers = pd.DataFrame(bottom_performers).assign(performer='bottom')
    barplot = \
        pd.concat([top_performers, bottom_performers], axis=0).sort_values(by=0, ascending=True).droplevel(axis=0,
                                                                                                           level=[0, 1])
    barplot = barplot.rename(columns={0: 'metric'})
    barplot = barplot.assign(vol_regime=barplot.index.get_level_values(2))
    color_discrete_map = barplot.reorder_levels(['vol_regime', 'training_scheme', 'L', 'regression', 'model'])
    barplot = \
        pd.DataFrame(data=barplot.values,
                     index=[f'({",".join((idx[-2], idx[-1], idx[0], idx[1], idx[2]))})' for idx in barplot.index],
                     columns=['metric', 'performer', 'vol_regime'])
    barplot = \
        barplot.assign(
            ranking=barplot.groupby(
                by=[
                    pd.Grouper(key='performer'), pd.Grouper(key='vol_regime')
                ]).apply(lambda x: x.rank(method='first'))['metric'].values
        )
    barplot = barplot.assign(model_name=barplot.index)
    colour_ls = px.colors.qualitative.Plotly[0:3]
    color_discrete_map_dd = {model: colour_ls[0] for model in barplot.index[:n_performers]}
    color_discrete_map_dd.update({model: colour_ls[1] for model in barplot.index[-n_performers:]})
    barplot = barplot.rename(columns={'vol_regime': 'market_regime'})
    fig = px.bar(barplot, x='model_name', y='metric', color='market_regime',
                 color_discrete_map=color_discrete_map_dd, text_auto='auto', barmode='group')
    fig.update_xaxes(title='models')
    fig.update_yaxes(title='QLIKE')
    fig.update_layout(title='QLIKE: Top and bottom 5 performers per market regime', showlegend=True)
    fig.write_image(os.path.abspath('../figures/qlike_performance.pdf'))