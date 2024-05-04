import pdb
import pandas as pd
import plotly.express as px
import os
import argparse
from data_centre.data import DBQuery


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating the best performing models.')
    parser.add_argument('--n_performers', default=5, type=int, help='Amount of performers.')
    parser.add_argument('--bottom_performers', default=0, type=int, help='Displaying bottom performs or not.')
    args = parser.parse_args()
    print(args)
    n_performers = args.n_performers
    performance = pd.read_csv('../results/performance.csv')
    vixm_cond1 = performance.model == 'har_eq'
    vixm_cond2 = performance.trading_session.isnull()
    performance.loc[vixm_cond1 & vixm_cond2, 'model'] = 'har_eq_vixm'
    performance['top_book'] = performance['top_book'].replace(1.0, 'tb')
    performance = pd.pivot(data=performance,
                           columns=['L', 'training_scheme'],
                           values='values',
                           index=['metric', 'regression', 'model', 'vol_regime', 'top_book',
                                  'trading_session']).round(4)
    qlike = performance.index.get_level_values(0) == 'qlike'
    ranking = performance.loc[qlike, :].transpose().unstack().unstack().dropna().sort_values()
    performers = ranking.groupby(by=pd.Grouper(level='vol_regime')).apply(lambda x: x.iloc[:n_performers])
    performers = pd.DataFrame(performers).assign(performer='top')
    if args.bottom_performers == 1:
        bottom_performers = ranking.groupby(by=pd.Grouper(level='vol_regime')).apply(lambda x: x.iloc[-n_performers:])
        bottom_performers = pd.DataFrame(bottom_performers).assign(performer='bottom')
        performers = pd.concat([performers, bottom_performers], axis=0)
    barplot = performers.sort_values(by=0, ascending=True).droplevel(axis=0, level=[0, 1])
    barplot = barplot.rename(columns={0: 'metric'})
    barplot = barplot.assign(vol_regime=barplot.index.get_level_values(2))
    barplot = \
        pd.DataFrame(data=barplot.values,
                     index=[f'${{{idx[1].upper()}}}^{{{(idx[-2], idx[0].upper())}}}_{{{idx[-1]}}}$'
                            for idx in barplot.index],
                     columns=['metric', 'performer', 'vol_regime'])
    barplot.index = barplot.index.str.replace('_EQ', '-eq')
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
    if args.bottom_performers == 1:
        color_discrete_map_dd.update({model: colour_ls[1] for model in barplot.index[-n_performers:]})
    barplot = barplot.rename(columns={'vol_regime': 'market_regime'})
    fig = px.bar(barplot, x='metric', y='model_name', color='market_regime', text_auto='auto', barmode='group',
                 orientation='h', category_orders={'market_regime': ['low', 'normal', 'high']})
    fig.update_xaxes(title='QLIKE')
    fig.update_yaxes(title='Models')
    display_number = {True: "", False: f' {n_performers}'}
    plural = {True: "", False: 's'}
    if args.bottom_performers == 1:
        fig.update_layout(title=f'QLIKE: Top and bottom {display_number[n_performers==1]} '
                                f'performer{plural[n_performers==1]} per market regime', showlegend=True)
    else:
        fig.update_layout(title=f'QLIKE: Top{display_number[n_performers==1]} performer{plural[n_performers==1]} '
                                f'per market regime',
                          showlegend=True)
    fig.update_layout(legend=dict(title='Market regime'), width=1_200)
    fig.write_image(os.path.abspath('../figures/qlike_performance_mkt_regime.pdf'))
    print(f'[Figures]: QLIKE performance (market regime) has been saved.')
    db_obj = DBQuery()
    qlike = \
        db_obj.query_data(db_obj.best_model_for_all_windows_query('qlike'), table='qlike').sort_values(by='L',
                                                                                                       ascending=True)
    barplot = qlike.set_index(['training_scheme', 'model', 'regression', 'trading_session',
                               'top_book'])[['values', 'L']]
    barplot['$L_{train}$'] = [f'${L.lower()}$' for L in barplot['L']]
    barplot = barplot.set_index('$L_{train}$', append=True)
    barplot = barplot.drop('L', axis=1)
    barplot = \
        pd.DataFrame(data=barplot.values,
                     index=[f'${{{idx[1].upper()}}}^{{{(idx[0], idx[2].upper())}}}_{{{idx[-1].replace("$","")}}}$'
                            for idx in barplot.index],
                     columns=['QLIKE'])
    fig = px.bar(barplot.reset_index(), x='QLIKE', y='index', text_auto='.4f',
                 orientation='h', title='QLIKE: Best performing model per rolling window')
    fig.update_layout(yaxis=dict(title_text='Models'), legend=dict(yanchor='top', xanchor='right'),
                      showlegend=True, width=1_200)
    fig.write_image(os.path.abspath('../figures/qlike_performance_L_train.pdf'))
    print(f'[Figures]: QLIKE performance (${{L_train}}$) has been saved.')
