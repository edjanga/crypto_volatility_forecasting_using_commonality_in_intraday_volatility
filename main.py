import pdb
import pandas as pd
from data_centre.data import Reader
from model.lab import ModelBuilder
from plots.maker import PlotResults
import argparse


if __name__ == '__main__':

    ####################################################################################################################
    #### Fit models
    ####################################################################################################################
    data_obj = Reader(file='./data_centre/tmp/aggregate2022')
    parser = argparse.ArgumentParser(description='Model Lab: Fit and store model results of research project 1.')
    parser.add_argument('--cross', default=False, help='Consider cross effect or not.', type=lambda x: bool(x))
    parser.add_argument('--save', default=True, help='Whether to save plots or not.', type=lambda x: bool(x))
    parser.add_argument('--models', default=['har', 'har_dummy_markets', 'har_cdr', 'har_universal'],
                        help='List of models to fit.')
    parser.add_argument('--F', default=['30T', '1H', '6H', '12H'],
                        help='Set of initial features for har components of the models.')
    parser.add_argument('--L', default='1D', help='Lookback window.')
    parser.add_argument('--var_explained', default=.9, help='Threshold for PCA component selection.')
    parser.add_argument('--regression_type', default='linear', help='For cross whether to allow for ensemble or not.')
    parser.add_argument('--smoother_freq', default='1D', help='Frequency to use to smoothen results.')
    parser.add_argument('--transformation', default=None, help='Transformation to apply on features.')
    parser.add_argument('--test', default=False, help='Test code or not.', type=lambda x: bool(x))
    args = parser.parse_args()
    if isinstance(args.models, str):
        models = [args.models]
    else:
        models = args.models
    if 'covariance_ar' in models:
        returns = data_obj.returns_read(raw=False)
    if 'har_cdr' in models:
        cdr = data_obj.cdr_read()
    if 'har_csr' in models:
        csr = data_obj.csr_read()
    if_exists_dd = {True: 'append', False: 'replace'}
    save = args.save
    var_explained = float(args.var_explained) if args.cross else None
    rv = data_obj.rv_read()
    F = args.F
    regression_type = args.regression_type
    model_builder_obj = ModelBuilder(F=F, h='30T', L=args.L, Q='1D')
    model_builder_obj.models = models
    agg = args.smoother_freq
    cross_name_dd = {False: 'not_crossed', True: 'cross'}
    transformation_dd = {None: 'level', 'log': 'log'}
    test = args.test
    lookback_ls = ['1D', '1W', '1M']
    lookback_ls = lookback_ls[lookback_ls.index(args.L):1] if lookback_ls.index(args.L) == 0 else \
        lookback_ls[0:lookback_ls.index(args.L)+1]
    F = F + lookback_ls
    model_builder_obj.L = lookback_ls[-1]
    model_builder_obj.F = F
    for _, model_type in enumerate(model_builder_obj.models):
        model_builder_obj.reinitialise_models_forecast_dd()
        print_tag = (model_builder_obj.L, model_builder_obj.F, cross_name_dd[args.cross],
                     transformation_dd[args.transformation])
        print(f'[Computation]: Compute all tables for {print_tag}...')
        """
        Generate all tables for model L, F cross
        """
        model_builder_obj.model_type = model_type
        print(model_builder_obj.model_type)
        if model_type in ['har', 'har_dummy_markets', 'har_universal']:
            if model_builder_obj.model_type != 'har_universal':
                model_builder_obj.add_metrics(df=rv, cross=args.cross, agg=agg,
                                              transformation=args.transformation,
                                              variance_explained=var_explained,
                                              regression_type=regression_type)
            else:
                model_builder_obj.add_metrics(df=rv, cross=args.cross, agg=agg,
                                              transformation=args.transformation,
                                              regression_type=regression_type)
        elif model_builder_obj.model_type == 'har_cdr':
            model_builder_obj.add_metrics(df=rv, df2=cdr, cross=args.cross, agg=agg,
                                          transformation=args.transformation,
                                          variance_explained=var_explained,
                                          regression_type=regression_type)
        elif model_builder_obj.model_type == 'har_csr':
            model_builder_obj.add_metrics(df=rv, df2=csr, cross=args.cross, agg=agg,
                                          transformation=args.transformation,
                                          variance_explained=var_explained,
                                          regression_type=regression_type)
        elif model_builder_obj.model_type in ['covariance_ar', 'risk_metrics']:
            model_builder_obj.add_metrics(df=returns, agg=agg,
                                          transformation=args.transformation,
                                          regression_type=regression_type)
        mse = [model_builder_obj.models_rolling_metrics_dd[model]['mse'] for model in
               model_builder_obj.models_rolling_metrics_dd.keys() if model
               and isinstance(model_builder_obj.models_rolling_metrics_dd[model]['mse'], pd.DataFrame)]
        mse = pd.concat(mse)
        qlike = [model_builder_obj.models_rolling_metrics_dd[model]['qlike'] for model in
                 model_builder_obj.models_rolling_metrics_dd.keys() if model
                 and isinstance(model_builder_obj.models_rolling_metrics_dd[model]['qlike'], pd.DataFrame)]
        qlike = pd.concat(qlike)
        r2 = [model_builder_obj.models_rolling_metrics_dd[model]['r2'] for model in
              model_builder_obj.models_rolling_metrics_dd.keys() if model
              and isinstance(model_builder_obj.models_rolling_metrics_dd[model]['r2'], pd.DataFrame)]
        r2 = pd.concat(r2)
        model_axis_dd = {model: False if model == 'har_universal' else True
                         for _, model in enumerate(model_builder_obj.models)}
        coefficient = \
            [pd.DataFrame(model_builder_obj.models_rolling_metrics_dd[model]['coefficient'].mean(
                axis=model_axis_dd[model]), columns=[model]) for model in
                model_builder_obj.models_rolling_metrics_dd.keys() if model and
            isinstance(model_builder_obj.models_rolling_metrics_dd[model]['coefficient'], pd.DataFrame)]
        if args.cross & (model_builder_obj._model_type != 'har_universal'):
            pass
        else:
            coefficient = pd.concat(coefficient, axis=1)
            model_specific_features = list(set(coefficient.index).difference((set(['const']+F))))
            model_specific_features.sort()
            if not args.cross:
                coefficient = coefficient.T[['const']+model_builder_obj.F+model_specific_features].T
                #if \not cross else coefficient.T[['const']+model_specific_features].T
            else:
                coefficient = coefficient.T[['const'] + model_specific_features].T
            coefficient.index.name = 'params'
            coefficient = pd.melt(coefficient.reset_index(), value_name='value', var_name='model',
                                  id_vars='params')
            coefficient.dropna(inplace=True)
        ModelBuilder.remove_redundant_key(model_builder_obj.models_forecast_dd)
        y = pd.concat(model_builder_obj.models_forecast_dd)
    """
    Table insertion
    """
    r2.index.name = 'timestamp'
    mse.index.name = 'timestamp'
    qlike.index.name = 'timestamp'
    y.index.name = 'timestamp'
    table_name_suffix_ls = [model_builder_obj.L, cross_name_dd[args.cross], transformation_dd[args.transformation]]
    if test:
        r2.to_csv(f'{"_".join(["r2"]+table_name_suffix_ls)}.csv')
        mse.to_csv(f'{"_".join(["mse"]+table_name_suffix_ls)}.csv')
        qlike.to_csv(f'{"_".join(["qlike"]+table_name_suffix_ls)}.csv')
        if (args.cross) & (model_builder_obj.models != 'har_universal'):
            pass
        else:
            coefficient.to_csv(f'{"_".join(["coefficient"]+table_name_suffix_ls)}.csv')
        y.to_csv(f'{"_".join(["y"]+table_name_suffix_ls)}.csv')
        print(f'[Creation]: All tables for {print_tag} have been generated.')
    else:
        r2.to_sql(con=model_builder_obj.db_connect_r2, name=f'{"_".join(["r2"]+table_name_suffix_ls)}',
                  if_exists=if_exists_dd[args.cross])
        mse.to_sql(con=model_builder_obj.db_connect_mse, name=f'{"_".join(["mse"]+table_name_suffix_ls)}',
                   if_exists=if_exists_dd[args.cross])
        qlike.to_sql(con=model_builder_obj.db_connect_qlike, name=f'{"_".join(["qlike"]+table_name_suffix_ls)}',
                     if_exists=if_exists_dd[args.cross])
        coefficient.to_sql(con=model_builder_obj.db_connect_coefficient,
                           name=f'{"_".join(["coefficient"]+table_name_suffix_ls)}', if_exists=if_exists_dd[args.cross])
        y.to_sql(con=model_builder_obj.db_connect_y,
                 name=f'{"_".join(["y"]+table_name_suffix_ls)}', if_exists='replace')
        print(f'[Insertion]: All tables for {print_tag} have been inserted into the database.')
    """
        Stats about number of PCAs per model and symbol
    """
    for L, ls in model_builder_obj.outliers_dd.items():
        model_builder_obj.outliers_dd[L] = pd.Series(ls, name=L)
    outliers = pd.DataFrame(model_builder_obj.outliers_dd)
    outliers = pd.melt(outliers, var_name='L', value_name='values')
    outliers.to_csv('outliers.csv') if test else \
        outliers.to_sql(con=model_builder_obj.db_connect_outliers, name='outliers', if_exists=if_exists_dd[args.cross])
    if not test:
        """
        Close databases
        """
        model_builder_obj.db_connect_r2.close()
        model_builder_obj.db_connect_mse.close()
        model_builder_obj.db_connect_qlike.close()
        model_builder_obj.db_connect_coefficient.close()
        model_builder_obj.db_connect_y.close()
    ####################################################################################################################
    ### Save plots
    ####################################################################################################################
    plot_maker_obj = PlotResults
    plot_maker_obj.scatterplot(save=save, L=args.L, cross=cross_name_dd[args.cross],
                               transformation=transformation_dd[args.transformation],
                               test=test, regression_type=regression_type)
    plot_maker_obj.rolling_metrics(save=save, L=args.L, cross=cross_name_dd[args.cross],
                                   transformation=transformation_dd[args.transformation],
                                   test=test, regression_type=regression_type)
    plot_maker_obj.distribution(save=save, L=args.L, cross=cross_name_dd[args.cross],
                                transformation=transformation_dd[args.transformation],
                                test=test, regression_type=regression_type)
    plot_maker_obj.rolling_metrics_barplot(save=save, L=args.L, cross=cross_name_dd[args.cross],
                                           transformation=transformation_dd[args.transformation], test=test,
                                           regression_type=regression_type)

