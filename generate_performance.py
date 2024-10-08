import pdb
import pandas as pd
from data_centre.data import DBQuery
import numpy as np
import plotly.express as px


TITLE_FONT_SIZE = 40
LABEL_AXIS_FONT_SIZE = 20
LEGEND_FONT_SIZE = 18
WIDTH = 1_500
HEIGHT = 800


if __name__ == '__main__':
    db_obj = DBQuery()
    ####################################################################################################################
    ## VOL REGIMES
    ####################################################################################################################
    # db_obj.best_model_for_all_market_regimes()
    ####################################################################################################################
    ## L_TRAIN
    ####################################################################################################################
    qlike = \
        db_obj.query_data(db_obj.best_model_for_all_windows_query(), table='y').sort_values(by='L', ascending=True)
    qlike.fillna(np.nan, inplace=True)
    qlike.replace(np.nan, '', inplace=True)
    qlike['trading_session'] = qlike['trading_session'].where(qlike['trading_session'] != '1', 'EQ')
    qlike['model'] = qlike['model'].str.replace('_eq', '')
    qlike['model'] = qlike['model'].str.upper()
    qlike['L'] = qlike['L'].str.lower()
    regression_conversion = \
        {'lightgbm': 'LightGBM', 'lasso': 'LASSO', 'elastic': 'ELASTIC',
         'ridge': 'RIDGE', 'linear': 'LR', 'lstm': 'LSTM', 'var': 'VAR',
         'pcr': 'PCR'}
    for regression, target in regression_conversion.items():
        qlike['regression'] = qlike['regression'].str.replace(regression, target)
    for i in range(1, 3):
        tmp = qlike.copy()
        tmp['values'] = 0
        tmp['training_scheme'] = f'ghost{i}'
        qlike = pd.concat([tmp, qlike]) if i < 2 else pd.concat([qlike, tmp])
    qlike['model_values'] = \
        [
            f'{row["model"]}<sub>{row["trading_session"]}</sub><sup>'\
            f'{row["training_scheme"], row["L"], row["regression"], row["top_book"]}</sup>: {row["values"]:.2f}'
            if row["top_book"] != ''
            else f'{row["model"]}<sub>{row["trading_session"]}</sub><sup>' 
                 f'{row["training_scheme"], row["L"], row["regression"]}</sup> : {row["values"]:.2f}'
            for idx, row in qlike[['values', 'training_scheme', 'L', 'model', 'regression', 'trading_session',
                                   'top_book']].iterrows()
        ]
    fig = px.bar(data_frame=qlike[['values', 'model', 'model_values']], x='values', y='model', orientation='h',
                 barmode='group', color='model_values',
                 color_discrete_sequence=px.colors.qualitative.Plotly[3:6] + px.colors.qualitative.Plotly[:3] +
                                         px.colors.qualitative.Plotly[6:10])
    fig.data = [plot.update({'showlegend': False}) if 'ghost' in plot.name else plot for plot in fig.data]
    fig.data = list(map(lambda x: x.update({'y': ['']}), fig.data))
    fig.update_xaxes(title=dict(text='QLIKE'))
    fig.update_yaxes(title=dict(text=''))
    fig.update_layout(width=1_000, height=600, font=dict(size=LABEL_AXIS_FONT_SIZE),
                      title=dict(font=dict(size=TITLE_FONT_SIZE)),
                      xaxis_title_font=dict(size=LABEL_AXIS_FONT_SIZE),
                      legend=dict(y=.05, x=1.1, title=None))
    fig.update_layout(yaxis=dict(domain=[.0, .25]))
    fig.show()
    pdb.set_trace()
