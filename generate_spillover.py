import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.kaleido.scope.mathjax = None
import plotly.express as px
from data_centre.data import DBQuery
from model.lab import SpilloverEffect
import pdb
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import itertools
import os
import rpy2.robjects as ro


if __name__ == '__main__':
    db_obj = DBQuery()
    spillover_obj = SpilloverEffect()
    data = db_obj.query_data(db_obj.best_model_for_all_windows_query('qlike'),
                             table='qlike').sort_values(by='L', ascending=False)
    suffix_name = \
        {'trading_session': {True: 'eq', False: 'eq_vixm'}, 'top_book': {True: 'top_book', False: 'full_book'}}
    ################################################################################################################
    ### Spillover Analysis
    ################################################################################################################
    spillover_effect = dict()
    degrees = dict()
    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}]],
                        column_titles=[f'${L.lower()}$' for L in ['1W', '1M', '6M']],
                        horizontal_spacing=.1, column_widths=[.33, .33, .33])
    for idx1, row in data.iterrows():
        L = row['L']
        model, regression, training_scheme, trading_session, top_book, h = row['model'], row['regression'], \
            row['training_scheme'], row['trading_session'], row['top_book'], row['h']
        y_hat = db_obj.forecast_query(L=L, model=model, regression=regression,
                                      trading_session=trading_session, top_book=top_book,
                                      training_scheme=training_scheme)[['y_hat', 'symbol']]
        y_hat = pd.pivot(data=y_hat, columns='symbol', values='y_hat')
        y_hat.index = pd.to_datetime(y_hat.index)
        y_hat = y_hat.resample(h).sum()
        dates = np.unique(y_hat.index.date).tolist()
        colour_map_dd = dict()
        for idx2, date in enumerate(dates):
            spillover_obj.forecast = y_hat.loc[y_hat.index.date == date, :]
            try:
                g_spillover = spillover_obj.spillover_matrix()
                spillover_obj.spillover_index_dd[date] = spillover_obj.spillover_index()
                spillover_obj.spillover_network_dd[date] = g_spillover
                spillover_obj.net_transmitter_sender_dd[date] = spillover_obj.spillover_type()
            except ro.rpy2.rinterface_lib.embedded.RRuntimeError:
                spillover_obj.spillover_index_dd[date] = np.nan
                tmp_network = pd.DataFrame(index=y_hat.columns, columns=y_hat.columns, data=np.nan)
                tmp_network.index.name, tmp_network.columns.name = None, None
                spillover_obj.spillover_network_dd[date] = tmp_network
                tmp_net_transmitter_sender = pd.Series(data=np.nan, name='Net')
                spillover_obj.net_transmitter_sender_dd[date] = tmp_net_transmitter_sender
        spillover_effect[f'${L.lower()}$'] = pd.Series(data=spillover_obj.spillover_index_dd,
                                                       name='Spillover Index').sort_index()
        spillover_network = pd.concat(spillover_obj.spillover_network_dd).dropna()
        spillover_network.columns = spillover_network.columns.str.replace('USDT', '')
        net_transmitter_sender = pd.concat(spillover_obj.net_transmitter_sender_dd).unstack(1).mean().dropna()
        net_transmitter_sender.index = net_transmitter_sender.index.str.replace('USDT', '')
        colour_map_dd[idx1] = net_transmitter_sender
        nodes = net_transmitter_sender.index.tolist()
        label_encoder = LabelEncoder()
        label_encoder.fit(spillover_network.columns.tolist())
        spillover_network = spillover_network.values.reshape(spillover_network.shape[1],
                                                             spillover_network.shape[0]//spillover_network.shape[1],
                                                             spillover_network.shape[1])
        spillover_network = np.mean(spillover_network, axis=1)
        threshold = np.percentile(spillover_network.reshape(np.cumprod(spillover_network.shape)[1]), q=95)
        degrees[L] = \
            pd.DataFrame([np.sum(spillover_network >= threshold, 0),
                          np.sum(spillover_network >= threshold, 1)]).transpose()
        degrees[L].index = label_encoder.inverse_transform(degrees[L].index)
        degrees[L].columns = ['in', 'out']
        graph = \
            nx.from_pandas_adjacency(
                df=pd.DataFrame(spillover_network,
                                index=label_encoder.inverse_transform(list(range(0, y_hat.shape[1]))),
                                columns=label_encoder.inverse_transform(list(range(0, y_hat.shape[1])))),
                create_using=nx.DiGraph)
        graph.add_nodes_from(nodes)
        edges = list(graph.edges)
        node_trace = go.Scatterpolar(r=tuple(len(nodes) * [.5]),
                                     theta=[i * (360 // len(nodes)) for i in range(1, len(nodes) + 1)], text=nodes,
                                     mode="markers+text", marker=(dict(size=15, symbol='circle')),
                                     customdata=[net for net in net_transmitter_sender], showlegend=False)
        textposition_ls = pd.cut(node_trace.theta, bins=[0, 89, 91, 179, 181, 269, 271, 360],
                                 labels=['middle right', 'top center', 'middle left', 'middle left', 'middle left',
                                         'bottom center', 'middle right'], ordered=False)
        node_trace.update(dict(textposition=textposition_ls))
        fig.add_trace(row=1, col=1 + idx1, trace=node_trace)
        for edge in edges:
            start_node = node_trace.theta[node_trace.text.index(edge[0])]
            end_node = node_trace.theta[node_trace.text.index(edge[-1])]
            if graph.get_edge_data(u=edge[0], v=edge[1])['weight'] >= threshold:
                direction = net_transmitter_sender[edge[0]] < net_transmitter_sender[edge[-1]]
                fig.add_trace(trace=go.Scatterpolar(r=[.5, .5],
                                                    theta=[start_node, end_node],
                                                    line=dict(width=2, color='black'),
                                                    mode='lines+markers',
                                                    marker=dict(symbol={False: 'triangle-left',
                                                                        True: 'triangle-right'}[direction],
                                                                size=10,
                                                                angleref='previous'),
                                                    showlegend=False), col=1 + idx1, row=1)
    degrees = pd.concat([pd.concat(degrees)[['in']].stack().unstack(1),
                         pd.concat(degrees)[['out']].stack().unstack(1)]).sort_index()
    degrees = degrees.loc[[('1W', 'in'), ('1W', 'out'), ('1M', 'in'), ('1M', 'out'), ('6M', 'in'), ('6M', 'out')], :]
    net_transmission_values = [list(trace.customdata) for trace in fig.data if trace['customdata'] is not None]
    print(degrees.to_latex())
    net_transmission_values = list(itertools.chain.from_iterable(net_transmission_values))
    fig.update_polars(angularaxis=dict(visible=False, showline=False, tickmode='array',
                                       tickvals=list(range(len(nodes))), ticktext=nodes),
                      radialaxis=dict(visible=False, showline=False))
    for trace in fig.data:
        if trace['customdata'] is not None:
            trace.update(marker=dict(cmax=max(net_transmission_values), cmin=min(net_transmission_values),
                                     color=trace.customdata, colorscale="Viridis",
                                     colorbar=dict(title='Net Transmitter/Receiver', len=0.3, x=-.5)))
    fig.update(layout=dict(showlegend=True, title='Connectedness network: Overview'),
               annotation=dict(font_size=20))
    # fig['layout'].update(margin=dict(l=0,r=0,b=0,t=0))
    # fig.update_annotations(font_size=20)
    spillover_effect = pd.concat(spillover_effect).reset_index()
    spillover_effect['level_1'] = pd.to_datetime(spillover_effect['level_1'], utc=True)
    spillover_effect = spillover_effect.set_index(['level_0', 'level_1']).groupby(
        by=[pd.Grouper(level=0), pd.Grouper(level=1, freq='7D')]).mean().reset_index(level=0).rename(
        columns={'level_0': 'models'})
    first_date = spillover_effect.dropna().groupby(by='models').apply(lambda x: x.first_valid_index())[-1]
    spillover_effect = spillover_effect.dropna().loc[spillover_effect.dropna().index >= first_date]
    spillover_effect = spillover_effect.rename(columns={'models': '$L_{train}$'})
    fig2 = px.line(spillover_effect, y='Spillover Index', color='$L_{train}$', title='Total Spillover Index')
    fig2.update_xaxes(title_text='Date', tickangle=45)
    fig.show()
    fig.write_image(os.path.abspath(f'./figures/spillover_network.pdf'))
    fig2.write_image(os.path.abspath(f'./figures/spillover_total_index.pdf'))
    print(f'[Figures]: Spillover network and total index have been saved.')
