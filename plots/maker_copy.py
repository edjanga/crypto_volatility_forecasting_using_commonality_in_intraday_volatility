import glob
import os.path
import pdb
import typing

import torch
import pandas as pd
from data_centre.helpers import coin_ls
from data_centre.data import Reader
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime
import numpy as np
import sqlite3
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import plotly.io as pio
pio.kaleido.scope.mathjax = None
import torch


class EDA:
    _plots_dir = os.path.abspath(__file__).replace('/plots/maker_copy.py', '/plots')
    reader_object = Reader()
    rv = reader_object.rv_read()
    returns = reader_object.returns_read()

    def __init__(self):
        pass

    @staticmethod
    def optimal_clusters() -> None:
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
        fig.update_layout(title='Optimal number of clusters: Analysis')
        fig.write_image(os.path.abspath(f'{EDA._plots_dir}/n_clusters.pdf'))
        print(f'[Plots]: Selection optimal cluster plot has been generated.')

    def daily_rv(self) -> None:
        daily_rv_df = pd.DataFrame()
        tmp_df = np.log(self.rv.resample('1D').sum())
        daily_rv_df = daily_rv_df.assign(cross_average=tmp_df.mean(axis=1),
                                         percentile05=tmp_df.transpose().quantile(.05),
                                         percentile95=tmp_df.transpose().quantile(.95),
                                         percentile25=tmp_df.transpose().quantile(.25),
                                         percentile75=tmp_df.transpose().quantile(.75))
        daily_rv_df.ffill(inplace=True)
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
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=500, width=800)
        fig.write_image(os.path.abspath(f'{EDA._plots_dir}/daily_rv.pdf'))

    def intraday_rv(self) -> None:
        rv_df = np.log(self.rv.resample('30T').sum())
        rv_df.replace(-np.inf, np.nan, inplace=True)
        rv_df.ffill(inplace=True)
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
                      line_dash='dash', line_color='red', annotation_text='US open',
                      annotation_position='bottom right')
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=500, width=800)
        fig.write_image(os.path.abspath('./plots/intraday_rv.pdf'))

    def daily_mean_correlation_matrix(self) -> None:
        symbols = self.rv.columns.tolist()
        corr = self.rv.resample('1D').sum().rolling(3).corr().dropna()
        corr = corr.values.reshape(corr.shape[0]//corr.shape[1], corr.shape[1], corr.shape[1])
        corr = torch.tensor(corr, dtype=torch.float32)
        corr = torch.mean(corr, 0)
        corr_df = pd.DataFrame(data=corr.detach().numpy(), index=symbols, columns=symbols)
        fig = go.Figure(data=go.Heatmap(z=corr_df.values, x=corr_df.columns, y=corr_df.columns, colorscale='Blues'))
        fig.update_layout(title='Daily pairwise RV correlation mean.')
        fig.write_image(os.path.abspath(f'{EDA._plots_dir}/daily_mean_corr_rv.pdf'))

    def daily_pairwise_correlation(self) -> None:
        corr_df = self.rv.resample('1D').sum().rolling(3).corr().dropna()
        corr_ls = corr_df.values.flatten().tolist()
        corr_ls = [corr for _, corr in enumerate(corr_ls) if corr < 1]
        mean_corr = np.mean(corr_ls)
        fig = px.histogram(x=corr_ls, labels={'x': '', 'count': ''},
                           title='Daily pairwise RV correlation: Distribution',
                           histnorm='probability')
        fig.add_vline(x=mean_corr, line_color='orange')
        fig.update_yaxes(title='')
        fig.write_image(os.path.abspath(f'{EDA._plots_dir}/daily_pairwise_correlation_distribution.pdf'))

    def boxplot(self):

        fig = make_subplots(rows=2, cols=1, row_titles=['Raw RV', 'RV'],
                            column_titles=['Raw and processed RV: Boxplot.'],
                            shared_xaxes=True)
        raw_rv = EDA.reader_object.rv_read(raw=True)
        raw_rv.index.name = 'Time'
        raw_rv = pd.melt(raw_rv.reset_index(), var_name='symbol', value_name='values', id_vars='Time')
        tmp_rv = np.log(self.rv.copy())
        tmp_rv.index.name = 'Time'
        tmp_rv = pd.melt(tmp_rv.reset_index(), var_name='symbol', value_name='values', id_vars='Time')
        for symbol in tmp_rv.symbol.unique().tolist()[:20]:
            fig.add_trace(go.Box(y=raw_rv.query(f'symbol == \"{symbol}\"')['values'].tolist(),
                                 name=symbol), row=1, col=1)
            fig.add_trace(go.Box(y=tmp_rv.query(f'symbol == \"{symbol}\"')['values'].tolist(),
                                 name=symbol), row=2, col=1)
        fig.update_layout(showlegend=False)
        fig.write_image(os.path.abspath(f'{EDA._plots_dir}/boxplot.png'))


class PlotResults:
    _data_centre_dir = os.path.abspath(__file__).replace('/plots/maker_copy.py', '/data_centre')
    db_connect_coefficient = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/coefficients.db'))
    db_connect_mse = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/mse.db'))
    db_connect_qlike = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/qlike.db'))
    db_connect_r2 = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/r2.db'))
    db_connect_y = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/y.db'))
    db_connect_commonality = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/commonality.db'))
    db_connect_rolling_metrics = \
        sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/rolling_metrics.db'))
    #db_connect_correlation = sqlite3.connect(database=os.path.abspath('./data_centre/databases/correlation.db'))
    colors_ls = px.colors.qualitative.Plotly
    models_ls = ['ar', 'risk_metrics', 'har', 'har_eq']

    def __init__(self):
        pass

    @staticmethod
    def coefficient(L: str, cross: bool, save: bool, transformation: str, test: bool = False,
                    models_excl: typing.Union[None, typing.List[str], str] = 'har_csr',
                    regression_type: str = 'linear'):
        if isinstance(models_excl, str):
            models_excl = [models_excl]
        """
        Query data
        """
        query = f'SELECT * FROM coefficient_{L}_{cross}_{transformation}_{regression_type}'
        coefficient = pd.read_sql(con=PlotResults.db_connect_coefficient, sql=query, index_col='index') if not test \
            else pd.read_csv(f'./coefficient_{L}_{cross}_{transformation}_{regression_type}.csv')
        coefficient = coefficient.query(f'model not in {models_excl}') if models_excl else coefficient
        """
        Plot bar plot
        """
        fig_title = f'Coefficient {L} {cross} {transformation} {regression_type}: Bar plot'
        fig = px.bar(coefficient, x='params', y='value', color='model', barmode='group', title=fig_title)
        if save:
            fig.write_image(os.path.abspath(f'./plots/plots/coefficient_{L}_{cross}_{transformation}_'
                                            f'{regression_type}.png'))
        else:
            fig.show()

    @staticmethod
    def rolling_metrics(L: str, training_scheme: str, save: bool, transformation: str,
                        regression_type: str = 'linear'):
        """
        Query data
        """
        con_dd = {'r2': PlotResults.db_connect_r2, 'mse': PlotResults.db_connect_mse,
                  'qlike': PlotResults.db_connect_qlike}
        fig_dd = {'r2': 1, 'mse': 2, 'qlike': 3}
        models_ls = PlotResults.models_ls
        markers_dd = {model: PlotResults.colors_ls[i] for i, model in enumerate(models_ls) if model}
        col_grid = 1
        row_grid = 3
        fig = make_subplots(rows=row_grid, cols=col_grid,
                            row_titles=['average R2', 'average MSE', 'average QLIKE'], shared_xaxes=True)
        fig_title = f'Rolling metrics {L} {training_scheme} {transformation} {regression_type}'
        for table_name, con in con_dd.items():
            query = [f'SELECT \"timestamp\", AVG(\"values\") AS \"values\", \"model\" FROM {table_name}_' \
                     f'{training_scheme}_{L}_{transformation}_{regression_type} ' \
                     f'WHERE model = \"{model}\" GROUP BY "timestamp"' for model in PlotResults.models_ls]
            query = ' UNION '.join(query)
            tmp = pd.read_sql(con=con, sql=query, index_col='timestamp')
            idx_common = tmp.groupby(by='model', group_keys=True).apply(lambda x: x.index)['har']
            for i, model in enumerate(models_ls):
                metric = tmp.query(f'model == "{model}"').loc[idx_common, :]
                fig.add_trace(go.Scatter(x=metric.index, y=metric['values'], marker_color=markers_dd[model],
                                         showlegend=table_name == 'r2', name=model), row=fig_dd[table_name], col=1)
            if table_name == 'r2':
                fig.add_trace(go.Scatter(x=metric.index, y=pd.Series(data=0, index=metric.index),
                                         line=dict(color='black', width=1, dash='dash'), showlegend=False),
                              row=fig_dd[table_name], col=1)
        fig.update_xaxes(tickangle=45, tickformat='%m-%Y')
        fig.update_layout(height=1500, width=1200, title={'text': fig_title})
        if save:
            fig.write_image(os.path.abspath(
                f'./plots/{L}/{training_scheme}/{transformation}/'
                f'rolling_metrics_{training_scheme}_{L}_{transformation}_{regression_type}.png'))
        else:
            fig.show()

    @staticmethod
    def rolling_metrics_barplot(L: str, training_scheme: str, save: bool, transformation: str,
                                regression_type: str = 'linear'):
        """
            Query data
        """
        con_dd = {'r2': PlotResults.db_connect_r2, 'mse': PlotResults.db_connect_mse,
                  'qlike': PlotResults.db_connect_qlike}
        fig_title = f'Rolling metrics {L} {training_scheme} {transformation} {regression_type}: Bar plot'
        stats = ['mean', lambda x: x.quantile(.5), lambda x: x.quantile(.25), lambda x: x.quantile(.75)]
        stats_columns = ['mean', 'median', '25th_percentile', '75th_percentile']
        col_grid = 1
        models_ls = PlotResults.models_ls
        metrics_ls = list(con_dd.keys())
        row_grid = len(metrics_ls)
        colors_ls = px.colors.qualitative.Plotly
        """
            Plot bar plot
        """
        fig = make_subplots(rows=row_grid, cols=col_grid, row_titles=['R2', 'MSE', 'QLIKE'], shared_xaxes=True)
        metrics = pd.DataFrame()
        for table_name, con in con_dd.items():
            """
            Query data
            """
            query = [f'SELECT \"timestamp\", AVG(\"values\") AS \"values\", \"model\" FROM {table_name}_' \
                     f'{training_scheme}_{L}_{transformation}_{regression_type} ' \
                     f'WHERE model = \"{model}\" GROUP BY "timestamp"' for model in PlotResults.models_ls]
            query = ' UNION '.join(query)
            tmp = pd.read_sql(con=con, sql=query, index_col='timestamp')
            tmp = tmp.groupby(by='model')['values'].agg(stats)
            tmp.columns = stats_columns
            tmp['metrics'] = table_name
            metrics = tmp if metrics.empty else pd.concat([metrics, tmp])
        idx_common = r2.groupby(by='model', group_keys=True).apply(lambda x: x.index)['har']
        r2 = r2.loc[r2.index.isin(idx_common)].groupby(by='model').agg(stats)
        for i, metric in enumerate(metrics_ls):
            subfig = go.Figure()
            tmp_df = metrics.query(f'metrics == "{metric}"').drop('metrics', axis=1)
            tmp_df = pd.melt(tmp_df, ignore_index=False, var_name='stats', value_name='value')
            data_bar_plot_ls = \
                [go.Bar(name=stats, marker_color=colors_ls[j], showlegend=i == 0, x=models_ls,
                        y=tmp_df.query(f'stats == "{stats}"').value.values.tolist())
                 for j, stats in enumerate(['25th_percentile', 'mean', 'median', '75th_percentile'])]
            for k, bar in enumerate(data_bar_plot_ls):
                subfig.add_trace(bar)
                fig.add_trace(subfig.data[k], row=i+1, col=1)
        fig.update_layout(height=900, width=1200, title={'text': fig_title}, barmode='group')
        if save:
            fig.write_image(
                os.path.abspath(f'./plots/{L}/{training_scheme}/{transformation}/'
                                f'rolling_metrics_bar_plot_{training_scheme}_{L}_'
                                f'{transformation}_{regression_type}.png'))
        else:
            fig.show()

    @staticmethod
    def scatterplot(L: str, training_scheme: str, transformation: str, regression_type: str = 'linear',
                    save: bool = True):
        """
        Query data
        """
        query = f'SELECT \"y\", \"y_hat\", \"model\", \"timestamp\" ' \
                f'FROM y_{training_scheme}_{L}_{transformation}_{regression_type};'
        y = pd.read_sql(con=PlotResults.db_connect_y, sql=query, index_col='timestamp')
        idx_common = y.groupby(by='model', group_keys=True).apply(lambda x: x.index)['har']
        y = y[y.index.isin(idx_common)]
        y.index = pd.to_datetime(y.index)
        models_ls = PlotResults.models_ls
        col_grid = 1
        row_grid = len(models_ls) + 1
        fig_title = f'Scatter plot - {L} {training_scheme} {transformation} {regression_type}'
        fig = make_subplots(rows=row_grid, cols=col_grid, row_titles=models_ls, shared_xaxes=False)
        for i, model in enumerate(models_ls):
            if model:
                tmp_df = y.query(f'model == "{model}"')
                tmp_df = tmp_df.resample('30T').sum()
                fig.add_trace(go.Scatter(x=tmp_df.y, y=tmp_df.y_hat, showlegend=True, name=model,
                                         mode='markers'), row=i+1, col=1)
        fig.update_yaxes(title='Fitted')
        fig.update_xaxes(title_text='Observed')
        fig.update_layout(height=1500, width=1200, title={'text': fig_title})
        if save:
            fig.write_image(os.path.abspath(f'./plots/{L}/{training_scheme}/{transformation}/'
                                            f'scatter_plot_{training_scheme}'
                                            f'_{L}_{transformation}_{regression_type}.png'))
        else:
            fig.show()

    @staticmethod
    def distribution(L: str, training_scheme: str, transformation: str, regression_type: str = 'linear',
                     save: bool = True):
        colors_ls = px.colors.qualitative.Plotly
        """
        Query data
        """
        query = f'SELECT \"y\", \"y_hat\", \"model\", \"timestamp\" ' \
                f'FROM y_{training_scheme}_{L}_{transformation}_{regression_type};'
        y = pd.read_sql(con=PlotResults.db_connect_y, sql=query, index_col='timestamp')
        idx_common = y.groupby(by='model', group_keys=True).apply(lambda x: x.index)['har']
        y = y[y.index.isin(idx_common)]
        y.index = pd.to_datetime(y.index)
        models_ls = PlotResults.models_ls
        col_grid = 1
        row_grid = len(models_ls) + 1
        fig = make_subplots(rows=row_grid, cols=col_grid, row_titles=models_ls)
        fig_title = f'Distributions - {L} {training_scheme} {transformation} {regression_type}'
        for i, model in enumerate(models_ls):
            tmp_df = y.query(f'model == "{model}"')
            tmp_df = tmp_df.resample('30T').sum()
            fig.add_trace(go.Histogram(x=tmp_df.y, showlegend=True,
                                       name='_'.join((model, 'y')), marker_color=colors_ls[0]), row=i+1, col=1)
            fig.add_trace(go.Histogram(x=tmp_df.y_hat, showlegend=True,
                                       name='_'.join((model, 'y_hat')), marker_color=colors_ls[1]), row=i+1, col=1)
        fig.update_layout(height=1500, width=1200, title={'text': fig_title})
        fig.update_layout(barmode='overlay')
        fig.update_traces(opacity=0.75)
        if save:
            fig.write_image(os.path.abspath(
                f'./plots/{L}/{training_scheme}/{transformation}/distributions_y_vs_y_hat_{L}_{training_scheme}_'
                f'{transformation}_{regression_type}.png'))
        else:
            fig.show()

    @staticmethod
    def commonality(save: bool = True):
        query = 'SELECT * FROM commonality;'
        commonality = pd.read_sql(query, con=PlotResults.db_connect_commonality, index_col='index')
        commonality.index = pd.to_datetime(commonality.index)
        commonality.dropna(inplace=True)
        commonality = commonality.loc[commonality.index.isin(commonality.query('L == "6M"').index), :]
        #commonality_group = commonality.groupby(by='L', group_keys=True)
        #commonality = commonality_group.apply(lambda x: x.resample('6M').mean()).iloc[:-1, :]
        commonality = commonality.reset_index().set_index('index')
        commonality = commonality.rename(columns={'L': 'L_train'})
        commonality.iloc[:, 0] = [f'{L.lower()}' for L in commonality.iloc[:, 0]]
        commonality.index.name = None
        fig_title = r'Commonality for different lookback windows'
        fig = px.line(commonality, y='values', color='L_train', title=fig_title)
        fig.update_layout({'xaxis_title': '', 'yaxis_title': 'Commonality'})
        if save:
            fig.write_image(os.path.abspath(f'./plots/commonality.pdf'))
        else:
            fig.show()


if __name__ == '__main__':
    eda_obj = EDA()
    eda_obj.daily_pairwise_correlation()

    # performance = pd.read_csv('./performance.csv',
    #                           index_col=['training_scheme', 'L', 'regression', 'model'])
    # performance = performance.query('metric == "qlike"')
    # performance = performance[['metric', 'values']]
    # performance = performance.groupby(by=[performance.index.get_level_values(0),
    #                                       performance.index.get_level_values(1),
    #                                       performance.index.get_level_values(2)])
    # top_performers = performance.apply(lambda x: x['values'].idxmin())
    # rolling_metrics = pd.read_csv('./rolling_metrics.csv',
    #                               index_col=['training_scheme', 'L', 'regression', 'model'])
    # rolling_metrics = rolling_metrics.loc[rolling_metrics.index.isin(top_performers.values), :]
    # rolling_metrics = \
    #     rolling_metrics.assign(regression_model=['_'.join((training_scheme, L, regression, model)) for training_scheme,
    #     L, regression, model in zip(rolling_metrics.index.get_level_values(0),
    #                                 rolling_metrics.index.get_level_values(1),
    #                                 rolling_metrics.index.get_level_values(-2),
    #                                 rolling_metrics.index.get_level_values(-1))])
    # rolling_metrics = rolling_metrics.reset_index().set_index('timestamp')
    # for L in ['1D', '1W', '1M']:
    #     fig1 = px.line(data_frame=rolling_metrics.query(f'L == \"{L}\" & metric=="r2"').sort_index(level=0), y='values',
    #                    color='regression_model', title=f'Best performing {L} qlike model - Rolling R2')
    #     fig2 = px.line(data_frame=rolling_metrics.query(f'L == \"{L}\" & metric=="mse"').sort_index(level=0),
    #                    y='values', color='regression_model', title=f'Best performing {L} qlike model - Rolling MSE')
    #     fig3 = px.line(data_frame=rolling_metrics.query(f'L == \"{L}\" & metric=="qlike"').sort_index(level=0),
    #                    y='values', color='regression_model',
    #                    title=f'Best performing {L} qlike model - Rolling QLIKE')
    #     for fig in [fig1, fig2, fig3]:
    #         fig.show()

