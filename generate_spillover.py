import concurrent.futures
import pdb

from data_centre.data import DBQuery
from model.lab import SpilloverEffect
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import rpy2.robjects as ro
import networkx as nx
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import argparse
import plotly.express as px
plt.rcParams['font.family'] = ['Open Sans', 'verdana', 'arial', 'sans-serif']
TITLE_FONT_SIZE = 40
LABEL_AXIS_FONT_SIZE = 20
LEGEND_FONT_SIZE = 18
NODE_FONT_SIZE = 17
WIDTH = 1_000
HEIGHT = 600
L_train_order = {'1W': 0, '1M': 1, '6M': 2}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating spillover analysis.')
    parser.add_argument('--title_figure', default=0, type=int, help='Title figures.')
    parser.add_argument('--best_performing', default=1, type=int)
    parser.add_argument('--spillover_network', default=1, type=int)
    parser.add_argument('--node_degrees_balance', default=1, type=int)
    parser.add_argument('--spillover_tsi', default=1, type=int)
    args = parser.parse_args()
    print(args)
    title_figure = bool(args.title_figure)
    db_obj = DBQuery()
    spillover_obj = SpilloverEffect()
    data = db_obj.query_data(db_obj.best_model_for_all_windows_query(), table='y').sort_values(by='L', ascending=False)
    ################################################################################################################
    ### Spillover Analysis
    ################################################################################################################
    spillover_effect = dict()
    degrees = dict()
    network = dict()
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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda x: x, x=date) for idx2, date in enumerate(dates)]
            for future in concurrent.futures.as_completed(futures):
                date = future.result()
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
        if 'label_encoder' not in vars():
            label_encoder = LabelEncoder()
            label_encoder.fit(net_transmitter_sender.index)
        spillover_network = spillover_network.values.reshape(spillover_network.shape[1],
                                                             spillover_network.shape[0] // spillover_network.shape[1],
                                                             spillover_network.shape[1])
        spillover_network = np.mean(spillover_network, axis=1)
        threshold = np.percentile(spillover_network.reshape(np.cumprod(spillover_network.shape)[1]), q=75)
        network[L] = (spillover_network, threshold)
        degrees[L] = \
            pd.DataFrame([np.sum(spillover_network >= threshold, 0),
                          np.sum(spillover_network >= threshold, 1)]).transpose()
    degrees_copy = degrees.copy()
    degrees = pd.concat(degrees)
    degrees.columns = ['in', 'out']
    degrees.index = pd.MultiIndex.from_product([degrees.index.get_level_values(0).unique().tolist(),
                                                net_transmitter_sender.index.tolist()])
    degrees = degrees.assign(balance=degrees['out'].subtract(degrees['in'])).sort_index(axis=0, level=0)
    L_train = data.drop(data['values'].idxmin()).L.tolist() if args.best_performing == 0 else \
        pd.DataFrame(data.loc[data['values'].idxmin(), :]).transpose().L.tolist()
    if len(L_train) > 1:
        L_train_diff = list(set(list(L_train_order.keys())).difference(set(L_train)))
        L_train_order.pop(L_train_diff[0])
        L_train = list(L_train_order.keys())
    norm = mcolors.Normalize(vmin=min(degrees.balance), vmax=max(degrees.balance))
    if args.spillover_network == 1:
        cmap = cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig = plt.figure(figsize=(35, 8)) if args.best_performing == 1 else plt.figure(figsize=(35, 20))
        gs = gridspec.GridSpec(nrows=2 if args.best_performing == 0 else 1, ncols=2, width_ratios=[1, 0.05],
                               height_ratios=[1, 1] if args.best_performing == 0 else None)
        for idx1, L in enumerate(L_train):
            G = nx.DiGraph()
            tmp_degrees = degrees.loc[degrees.index.get_level_values(0) == L, :]
            for idx2, row in enumerate(list(tmp_degrees.iterrows())):
                G.add_node(row[0][-1], value=row[1]['balance'])
            pos = nx.circular_layout(G)
            tmp_graph = \
                nx.from_pandas_adjacency(df=pd.DataFrame(network[L][0], index=list(G.nodes), columns=list(G.nodes)))
            edges = list(tmp_graph.edges)
            for edge in list(edges):
                u, v = edge[0], edge[-1]
                if tmp_graph.get_edge_data(u, v)['weight'] < network[L][1]:
                    tmp_graph.remove_edge(u, v)
            G.add_edges_from(list(tmp_graph.edges))
            if args.best_performing == 1:
                ax = fig.add_subplot(gs[0, 0])
                ax.set_title(f'${L.lower()}$', fontdict=dict(fontsize=LABEL_AXIS_FONT_SIZE))
                nx.draw_networkx_nodes(G, pos, node_color=tmp_degrees['balance'].values, cmap=cmap, node_size=500,
                                       ax=ax)
                nx.draw_networkx_labels(G, pos, labels={node: str(node) for node in G.nodes()},
                                        font_size=NODE_FONT_SIZE, font_color='black', ax=ax)
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', arrows=True)
            else:
                ax = fig.add_subplot(gs[idx1, 0])
                ax.set_title(f'${L.lower()}$', fontdict=dict(fontsize=LABEL_AXIS_FONT_SIZE))
                nx.draw_networkx_nodes(G, pos, node_color=tmp_degrees['balance'].values, cmap=cmap,
                                       node_size=500, ax=ax)
                nx.draw_networkx_labels(G, pos, labels={node: str(node) for node in G.nodes()},
                                        font_size=NODE_FONT_SIZE, font_color='black', ax=ax)
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', arrows=True)
        cbar_ax = fig.add_subplot(gs[:, 1])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Degree of transmission', fontsize=LABEL_AXIS_FONT_SIZE)
        plt.show()
    if args.spillover_tsi == 1:
        spillover_effect = pd.concat(spillover_effect).reset_index()
        spillover_effect = \
            spillover_effect.loc[spillover_effect.level_0.isin(list(map(lambda x: f'${x.lower()}$', L_train))), :]
        spillover_effect['level_1'] = pd.to_datetime(spillover_effect['level_1'], utc=True)
        spillover_effect = spillover_effect.set_index(['level_0', 'level_1']).groupby(
            by=[pd.Grouper(level=0), pd.Grouper(level=1, freq='7D')]).mean().reset_index(level=0).rename(
            columns={'level_0': 'models'})
        first_date = spillover_effect.dropna().groupby(by='models').apply(lambda x: x.first_valid_index())[-1]
        spillover_effect = spillover_effect.dropna().loc[spillover_effect.dropna().index >= first_date]
        spillover_effect = spillover_effect.rename(columns={'models': r'$L_{train}$'})
        spillover_effect['$L_{train}$'] = \
            spillover_effect['$L_{train}$'].apply(lambda x: f'<i>{x.replace("$","").lower()}</i>')
        fig = px.line(spillover_effect, y='Spillover Index', color=r'$L_{train}$',
                      title={True: 'Total Spillover Index', False: ''}[title_figure],
                      category_orders={'$L_{train}$': [f'<i>{l_train.lower()}</i>' for l_train in L_train]})
        if args.best_performing == 1:
            if (args.best_performing == 1) & title_figure:
                fig.layout.title.text = ' - '.join((fig.layout.title.text, f'<i>{L_train[0].lower()}</i>'))
        fig.update_xaxes(title_text='Date', tickangle=45)
        fig.update_layout(title=dict(font=dict(size=TITLE_FONT_SIZE)),
                          width=WIDTH, height=HEIGHT, font=dict(size=LABEL_AXIS_FONT_SIZE),
                          legend=dict(title=None, orientation='h',
                                      xanchor='right', yanchor='top', y=1.1, x=1, entrywidth=30))
        fig.show()
    if args.node_degrees_balance == 1:
        degrees = pd.concat(degrees_copy)
        degrees = degrees.assign(balance=degrees[1].subtract(degrees[0])).rename(columns={0: 'in', 1: 'out'})
        degrees = \
            pd.concat([degrees[['in']].stack().unstack(1),
                       degrees[['out']].stack().unstack(1),
                       degrees[['balance']].stack().unstack(1)]).sort_index(ascending=[True, False])
        degrees.columns = label_encoder.inverse_transform(degrees.columns)
        degrees.reset_index(inplace=True)
        degrees = degrees.loc[degrees.level_0.isin(L_train), :]
        degrees.level_0 = degrees.level_0.astype('category')
        degrees.level_0 = degrees.level_0.cat.set_categories(L_train)
        degrees.sort_values('level_0', inplace=True)
        degrees['level_0'] = degrees['level_0'].apply(lambda x: f'<i>{x.lower()}</i>')
        degrees.set_index(['level_0', 'level_1'], inplace=True)
        degrees = degrees.stack().reset_index()
        fig = px.bar(data_frame=degrees, x='level_2', barmode='group', y=0,
                     title={True: f'Nodes: degrees and balance', False: ''}[title_figure],
                     text_auto='.0f',
                     category_orders={'level_0': [f'<i>{l_train.lower()}</i>' for l_train in L_train]},
                     facet_row='level_0', color='level_1')
        if (args.best_performing == 1) & title_figure:
            fig.layout.title.text = ' - '.join((fig.layout.title.text, f'<i>{L_train[0].lower()}</i>'))
        fig.update_layout(legend=dict(title=None, orientation='h', xanchor='right', yanchor='bottom', y=1, x=1),
                          title=dict(font=dict(size=TITLE_FONT_SIZE)),
                          font=dict(size=LABEL_AXIS_FONT_SIZE+10))
        fig.update_yaxes(title_text=None)
        fig.update_xaxes(title_text=None, tickangle=45)
        fig.for_each_annotation(lambda x: x.update({'text': x['text'].replace('level_0=', '')}))
        fig.show()
