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
import itertools


class EDA:
    _figures_dir = os.path.abspath(__file__).replace('/figures/maker.py', '/figures')
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
        fig.write_image(os.path.abspath(f'{EDA._figures_dir}/n_clusters.pdf'))
        print(f'[figures]: Selection optimal cluster plot has been generated.')

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
        fig.update_layout(height=500, width=800, title_text='Annualized daily RV')
        fig.write_image(os.path.abspath(f'{EDA._figures_dir}/daily_rv.pdf'))

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
                      line_dash='dash', line_color='red', annotation_text='US open',
                      annotation_position='bottom right')
        fig.update_xaxes(tickangle=45, title_text='Time')
        fig.update_yaxes(title_text='RV')
        fig.update_layout(height=500, width=800, title_text='Annualized intraday RV: Statistics')
        fig.write_image(os.path.abspath(f'{EDA._figures_dir}/intraday_rv.pdf'))

    def daily_mean_correlation_matrix(self) -> None:
        corr = self.rv.copy()
        corr.columns = corr.columns.str.replace('USDT', '')
        symbols = corr.columns.tolist()
        corr = corr.resample('1D').sum().rolling(3).corr().dropna()
        corr = corr.values.reshape(corr.shape[0]//corr.shape[1], corr.shape[1], corr.shape[1])
        corr = torch.tensor(corr, dtype=torch.float32)
        corr = torch.mean(corr, 0)
        corr_df = pd.DataFrame(data=corr.detach().numpy(), index=symbols, columns=symbols)
        fig = go.Figure(data=go.Heatmap(z=corr_df.values, x=corr_df.columns, y=corr_df.columns, colorscale='Blues'))
        fig.update_layout(title='Daily pairwise RV correlation mean')
        fig.write_image(os.path.abspath(f'{EDA._figures_dir}/daily_mean_corr_rv.pdf'))

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
        fig.write_image(os.path.abspath(f'{EDA._figures_dir}/daily_pairwise_correlation_distribution.pdf'))

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
        fig.write_image(os.path.abspath(f'{EDA._figures_dir}/boxplot.pdf'))


class PlotResults:
    _data_centre_dir = os.path.abspath(__file__).replace('/figures/maker.py', '/data_centre')
    db_connect_mse = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/mse.db'))
    db_connect_qlike = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/qlike.db'))
    db_connect_y = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/y.db'))
    db_connect_commonality = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/commonality.db'))
    db_feature_importance = sqlite3.connect(
        database=os.path.abspath(f'{_data_centre_dir}/databases/feature_importance.db'), check_same_thread=False
    )
    db_pca = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/pca.db'),
                             check_same_thread=False)
    colors_ls = px.colors.qualitative.Plotly
    _models_ls = ['ar', 'har', 'har_eq']
    _training_scheme_ls = ['SAM', 'ClustAM', 'CAM', 'UAM']
    _L = ['1W', '1M', '6M']

    def __init__(self):
        pass

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
                f'./figures/{L}/{training_scheme}/{transformation}/'
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
                os.path.abspath(f'./figures/{L}/{training_scheme}/{transformation}/'
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
            fig.write_image(os.path.abspath(f'./figures/{L}/{training_scheme}/{transformation}/'
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
                f'./figures/{L}/{training_scheme}/{transformation}/distributions_y_vs_y_hat_{L}_{training_scheme}_'
                f'{transformation}_{regression_type}.png'))
        else:
            fig.show()

    @staticmethod
    def commonality(save: bool = True) -> None:
        query = 'SELECT * FROM commonality;'
        commonality = pd.read_sql(query, con=PlotResults.db_connect_commonality, index_col='index',
                                  columns=['index', 'L', 'values'])
        commonality.index = pd.to_datetime(commonality.index, utc=True)
        commonality = commonality.loc[commonality.index.isin(commonality.query('L == "6M"').index), :]
        fig = px.line(commonality, y='values', color='L')
        commonality = commonality.rename(columns={'L': f'$L_train$'})
        fig_title = r'Commonality for different lookback windows'
        fig = px.line(commonality, y='values', color=r'L_train', title=fig_title)
        fig.update_layout({'xaxis_title': '', 'yaxis_title': 'Commonality'})
        if save:
            fig.write_image(os.path.abspath(f'./figures/commonality.pdf'))
        else:
            fig.show()

    @staticmethod
    def first_principal_component(save: bool = True) -> None:
        first_comp_weights = pd.read_sql(con=PlotResults.db_pca,
                                         sql='SELECT * FROM coefficient_1st_principal_component')
        first_comp_weights = first_comp_weights.groupby(by=[pd.Grouper(key='training_scheme'), pd.Grouper(key='L'),
                                                            pd.Grouper(key='variable')])[['values']].mean()
        fig = make_subplots(rows=len(first_comp_weights.index.get_level_values(0).unique()),
                            cols=1, row_titles=first_comp_weights.index.get_level_values(0).unique().tolist(),
                            column_titles=['First principal component - Crypto pair weights'],
                            vertical_spacing=.3)
        for i, training_scheme in enumerate(first_comp_weights.index.get_level_values(0).unique().tolist()):
            tmp = first_comp_weights.loc[first_comp_weights.index.get_level_values(0) == training_scheme, :]
            tmp = tmp.droplevel(0, 0).reset_index(level=1)
            bars = [
                go.Bar(name=f'{L}', x=tmp.query(f"L == \"{L}\"").variable, y=tmp.query(f"L == \"{L}\"")['values'],
                       marker_color=px.colors.qualitative.Plotly[idx], showlegend=i == 0) for idx, L in
                enumerate(['1W', '1M', '6M'])
            ]
            fig.add_traces(data=bars, rows=i + 1, cols=1)
        fig.update_xaxes(tickangle=45)
        fig.update_layout(barmode='group')
        if save:
            fig.write_image(os.path.abspath(f'{EDA._figures_dir}/first_components_weights.pdf'))
        else:
            fig.show()

    @staticmethod
    def feature_importance(save: bool = True) -> None:
        feature_importance = list()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(pd.read_sql, con=PlotResults.db_feature_importance,
                                       sql=f'SELECT * FROM \"{name}\"',
                                       index_col='feature') for name in PlotResults._training_scheme_ls]
            for future in concurrent.futures.as_completed(futures):
                if future.exception() is None:
                    table = future.result()
                    feature_importance.append(table)
        feature_importance = pd.concat(feature_importance)
        feature_importance = \
            feature_importance.groupby(by=[pd.Grouper(level='feature'),
                                           pd.Grouper(key='model_type'),
                                           pd.Grouper(key='training_scheme')]).apply(lambda x: x.importance.mean())
        feature_importance = feature_importance.groupby(by=[pd.Grouper(level='feature'),
                                                            pd.Grouper(level='training_scheme')]).mean()
        feature_importance = \
            feature_importance.groupby(by=pd.Grouper(level='training_scheme')).apply(
                lambda x: x.sort_values(ascending=False)[:10]
            )
        feature_importance = feature_importance.droplevel(axis=0, level=2)
        fig = make_subplots(cols=len(PlotResults._training_scheme_ls),
                            rows=1, column_titles=PlotResults._training_scheme_ls)
        for i, training_scheme in enumerate(PlotResults._training_scheme_ls):
            tmp = feature_importance[feature_importance.index.get_level_values(0) == training_scheme].copy()
            tmp.sort_values(inplace=True, ascending=True)
            fig.add_traces(data=go.Bar(x=tmp.values.tolist(), y=tmp.index.get_level_values(1), showlegend=False,
                                       orientation='h', marker={'color': tmp.values.tolist(), 'colorscale': 'viridis'}),
                           rows=1, cols=i+1)
        fig.update_layout(title='Feature importance: Top 10')
        pdb.set_trace()
        if save:
            fig.write_image(os.path.abspath(f'{EDA._figures_dir}/feature_importance.pdf'))
        else:
            fig.show()
