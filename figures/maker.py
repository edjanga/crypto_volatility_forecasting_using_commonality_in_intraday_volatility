import concurrent.futures
import os.path
import pdb
import typing
import pandas as pd
from data_centre.data import Reader
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import sqlite3
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import plotly.io as pio
import torch
pio.kaleido.scope.mathjax = None

TITLE_FONT_SIZE = 40
LABEL_AXIS_FONT_SIZE = 20
LEGEND_FONT_SIZE = 18
HEIGHT = 500
WIDTH = 900


class EDA:
    _figures_dir = os.path.abspath(__file__).replace('/figures/maker.py', '/figures')
    reader_object = Reader()
    rv = reader_object.rv_read()
    returns = reader_object.returns_read()

    def __init__(self, title: bool=False):
        self._title = title

    def optimal_clusters(self) -> None:
        kmeans = KMeans(n_init='auto', random_state=123)
        silhouette = list()
        tmp = EDA.rv.transpose().copy()
        for k in range(2, tmp.shape[0]):
            kmeans.n_clusters = k
            kmeans.fit(tmp)
            silhouette.append(silhouette_score(tmp, kmeans.labels_))
        fig = make_subplots(rows=1, cols=1)
        fig.add_traces(data=go.Bar(x=list(range(2, len(silhouette) + 2)), y=silhouette, showlegend=False))
        fig.add_traces(data=go.Scatter(name='k', x=list(range(2, len(silhouette) + 2)),
                                       mode='lines+markers', marker=dict(size=10), y=silhouette, showlegend=False))
        fig.update_xaxes(title='Number of clusters')
        fig.update_yaxes(title='Silhouette score')
        fig.update_layout(title=dict(text={True:'Optimal number of clusters: Analysis', False: ''}[self._title],
                                     font=dict(size=TITLE_FONT_SIZE)),
                          width=WIDTH, height=HEIGHT, font=dict(size=LABEL_AXIS_FONT_SIZE))
        fig.update_annotations(font=dict(size=TITLE_FONT_SIZE), x=1)
        fig.show()

    def daily_rv(self) -> None:
        daily_rv_df = pd.DataFrame()
        tmp_df = self.rv.resample('1D').sum()
        daily_rv_df = daily_rv_df.assign(cross_average=tmp_df.mean(axis=1),
                                         percentile05=tmp_df.transpose().quantile(.05),
                                         percentile95=tmp_df.transpose().quantile(.95),
                                         percentile25=tmp_df.transpose().quantile(.25),
                                         percentile75=tmp_df.transpose().quantile(.75))
        daily_rv_df = daily_rv_df.ffill().mul(np.sqrt(360))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_rv_df.index, y=daily_rv_df.percentile95, fill=None, mode='lines',
                                 line_color='orange', line={'width': 1, 'dash': 'dash'}, showlegend=False))
        fig.add_trace(go.Scatter(x=daily_rv_df.index, y=daily_rv_df.percentile05, fill='tonexty', mode='lines',
                                 line_color='orange', line={'width': 1, 'dash': 'dash'}, showlegend=False))

        fig.add_trace(go.Scatter(x=daily_rv_df.index, y=daily_rv_df.percentile75, fill=None, mode='lines',
                                 line_color='green', line={'width': 2, 'dash': 'dash'}, showlegend=False))

        fig.add_trace(go.Scatter(x=daily_rv_df.index, y=daily_rv_df.percentile25, fill='tonexty', mode='lines',
                                 line_color='green', line={'width': 2, 'dash': 'dash'}, showlegend=False))

        fig.add_trace(go.Scatter(x=daily_rv_df.index, y=daily_rv_df.cross_average, fill=None, mode='lines',
                                 line_color='blue', line={'width': 3}, showlegend=False))
        fig.update_xaxes(tickangle=45, title_text='Date')
        fig.update_yaxes(title_text='RV')
        fig.update_layout(title=dict(text={True: 'Annualized daily RV', False: ''}[self._title],
                                     font=dict(size=TITLE_FONT_SIZE)),
                          width=WIDTH, height=HEIGHT, font=dict(size=LABEL_AXIS_FONT_SIZE))
        fig.update_annotations(font=dict(size=TITLE_FONT_SIZE), x=1)
        fig.show()

    def intraday_rv(self) -> None:
        rv_df = self.rv.resample('30T').sum().mul(np.sqrt(pd.to_timedelta('360D')//pd.to_timedelta('30T')))
        rv_df = rv_df.ffill()
        mean_group = rv_df.mean(axis=1).groupby(by=[rv_df.index.hour, rv_df.index.minute])
        diurnal_rv_df = pd.DataFrame()
        diurnal_rv_df = \
            diurnal_rv_df.assign(cross_average=mean_group.mean().values,
                                 percentile25=mean_group.quantile(.25).values,
                                 percentile75=mean_group.quantile(.75).values,
                                 percentile05=mean_group.quantile(.05).values,
                                 percentile95=mean_group.quantile(.95).values)
        diurnal_rv_df.index = pd.date_range(start='00:00', end='23:30', freq='30T')
        diurnal_rv_df.index = [idx.strftime('%H:%M') for idx in diurnal_rv_df.index]
        hours_ls = list(diurnal_rv_df.index)
        diurnal_rv_df.ffill(inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=diurnal_rv_df.index, y=diurnal_rv_df.percentile95, fill=None, mode='lines',
                                 line_color='orange', line={'width': 1, 'dash': 'dash'}, showlegend=False))
        fig.add_trace(go.Scatter(x=diurnal_rv_df.index, y=diurnal_rv_df.percentile05, fill='tonexty', mode='lines',
                                 line_color='orange', line={'width': 1, 'dash': 'dash'}, showlegend=False))

        fig.add_trace(go.Scatter(x=diurnal_rv_df.index, y=diurnal_rv_df.percentile75, fill=None, mode='lines',
                                 line_color='green',  line={'width': 2, 'dash': 'dash'}, showlegend=False))

        fig.add_trace(go.Scatter(x=diurnal_rv_df.index, y=diurnal_rv_df.percentile25, fill='tonexty', mode='lines',
                                 line_color='green', line={'width': 2, 'dash': 'dash'}, showlegend=False))

        fig.add_trace(go.Scatter(x=diurnal_rv_df.index, y=diurnal_rv_df.cross_average, fill=None, mode='lines',
                                 line_color='blue', line={'width': 3}, showlegend=False))
        fig.add_vline(x=hours_ls.index('08:30'), line_width=2,
                      line_dash='dash', line_color='black', annotation_text='Europe open',
                      annotation_position='top right')
        fig.add_vline(x=hours_ls.index('16:00'), line_width=2,
                      line_dash='dash', line_color='red', annotation_text='Europe close',
                      annotation_position='bottom right')
        fig.add_vline(x=hours_ls.index('00:00'), line_width=2,
                      line_dash='dash', line_color='black', annotation_text='Asia open',
                      annotation_position='top right')
        fig.add_vline(x=hours_ls.index('06:00'), line_width=2,
                      line_dash='dash', line_color='red', annotation_text='Asia close',
                      annotation_position='bottom left')
        fig.add_vline(x=hours_ls.index('14:30'), line_width=2,
                      line_dash='dash', line_color='black', annotation_text='US open',
                      annotation_position='top right')
        fig.add_vline(x=hours_ls.index('21:00'), line_width=2,
                      line_dash='dash', line_color='red', annotation_text='US close',
                      annotation_position='bottom right')
        fig.update_xaxes(tickangle=45, title_text='Time UTC')
        fig.update_yaxes(title_text='RV')
        fig.update_layout(title=dict(text={True: 'Annualized intraday RV: Statistics', False: ''}[self._title],
                                     font=dict(size=TITLE_FONT_SIZE)),
                          width=WIDTH, height=HEIGHT, font=dict(size=LABEL_AXIS_FONT_SIZE))
        fig.show()

    def daily_correlation_mean_matrix(self) -> None:
        corr = self.rv.copy()
        corr.columns = corr.columns.str.replace('USDT', '')
        symbols = corr.columns.tolist()
        corr = corr.resample('1D').sum().rolling(3).corr().dropna()
        corr = corr.values.reshape(corr.shape[0]//corr.shape[1], corr.shape[1], corr.shape[1])
        corr = torch.tensor(corr, dtype=torch.float32)
        corr = torch.mean(corr, 0)
        corr_df = pd.DataFrame(data=corr.detach().numpy(), index=symbols, columns=symbols)
        fig = go.Figure(data=go.Heatmap(z=corr_df.values, x=corr_df.columns, y=corr_df.columns, colorscale='Blues'))
        fig.update_layout(title=dict(text={True: 'Daily pairwise RV correlation mean', False: ''}[self._title],
                                     font=dict(size=1.5*TITLE_FONT_SIZE)),
                          font=dict(size=2*LABEL_AXIS_FONT_SIZE), height=2.5*HEIGHT, width=2*WIDTH)
        fig.show()

    def daily_pairwise_correlation(self) -> None:
        corr_df = self.rv.resample('1D').sum().mul(np.sqrt(360)).rolling(3).corr().dropna()
        corr_ls = corr_df.values.flatten().tolist()
        corr_ls = [corr for _, corr in enumerate(corr_ls) if corr < 1]
        mean_corr = np.mean(corr_ls)
        fig = px.histogram(x=corr_ls, labels={'x': '', 'count': ''},
                           title='Daily pairwise annualized RV correlation: Distribution',
                           histnorm='probability')
        fig.add_vline(x=mean_corr, line_color='orange')
        fig.update_yaxes(title='')
        fig.update_layout(title=dict(font=dict(size=TITLE_FONT_SIZE)),
                          width=WIDTH, height=HEIGHT, font=dict(size=LABEL_AXIS_FONT_SIZE))
        fig.show()

    def boxplot(self, transformation: str = None):
        if not transformation:
            fig = make_subplots(rows=1, cols=1, column_titles=['RV: Boxplot'])
        else:
            fig = make_subplots(rows=2, cols=1, row_titles=['Raw RV', 'RV'], shared_xaxes=True)
        raw_rv = EDA.reader_object.rv_read(raw=True)
        raw_rv.columns = raw_rv.columns.str.replace('USDT', '')
        raw_rv = pd.melt(raw_rv, var_name='symbol', value_name='values', ignore_index=True)
        if transformation == 'log':
            tmp_rv = np.log(self.rv.copy())
            tmp_rv = pd.melt(tmp_rv.reset_index(), var_name='symbol', value_name='values')
        for symbol in raw_rv.symbol.unique().tolist()[:20]:
            fig.add_trace(go.Box(y=raw_rv.query(f'symbol == \"{symbol}\"')['values'].tolist(),
                                 name=symbol), row=1, col=1)
            if transformation == 'log':
                fig.add_trace(go.Box(y=tmp_rv.query(f'symbol == \"{symbol}\"')['values'].tolist(),
                                     name=symbol), row=2, col=1)
        fig.update_layout(showlegend=False,
                          title_text={True: 'RV: Boxplot',
                                      False: 'Raw and processed RV: Boxplot.'}[transformation is None])
        fig.update_layout(title=dict(font=dict(size=TITLE_FONT_SIZE)),
                          width=WIDTH, height=HEIGHT, font=dict(size=LABEL_AXIS_FONT_SIZE))
        fig.show()