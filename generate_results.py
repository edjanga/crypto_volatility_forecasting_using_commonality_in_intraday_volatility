import os.path
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
    parser.add_argument('--training_scheme', default=False, help='Training scheme under which models are trained.',
                        type=str)
    parser.add_argument('--save', default=True, help='Whether to save plots or not.', type=bool)
    parser.add_argument('--models', help='List of models to fit.')
    parser.add_argument('--F', default=['30T', '1H', '6H', '12H'],
                        help='Set of initial features for har components of the models.')
    parser.add_argument('--L', default='1D', help='Lookback window.')
    parser.add_argument('--var_explained', default=.9, help='Threshold for PCA component selection.')
    parser.add_argument('--regression_type', default='linear', help='For cross whether to allow for ensemble or not.')
    parser.add_argument('--smoother_freq', default='1W', help='Frequency to use to smoothen results.')
    parser.add_argument('--transformation', default=None, help='Transformation to apply on features.')
    parser.add_argument('--test', default=False, help='Test code or not.', type=bool)
    args = parser.parse_args()
    if isinstance(args.models, str):
        models = [args.models]
    else:
        models = args.models
    if models == 'risk_metrics' > 0:
        returns = data_obj.returns_read(raw=False)
    if 'har_cdr' in models:
        cdr = data_obj.cdr_read()
    if 'har_csr' in models:
        csr = data_obj.csr_read()
    save = args.save
    rv = data_obj.rv_read(variance=True)
    F = args.F
    regression_type = args.regression_type
    model_builder_obj = ModelBuilder(F=F, h='30T', L=args.L, Q='1D', model_type=models[0])
    agg = args.smoother_freq
    transformation_dd = {None: 'level', 'log': 'log'}
    test = args.test
    lookback_ls = ['1D', '1W', '1M']
    lookback_ls = lookback_ls[lookback_ls.index(args.L):1] if lookback_ls.index(args.L) == 0 else \
        lookback_ls[0:lookback_ls.index(args.L)+1]
    F = F + lookback_ls
    model_builder_obj.L = lookback_ls[-1]
    model_builder_obj.F = [F[0]] if len(set(models).intersection(set(['ar', 'risk_metrics']))) > 0 else F
    model_builder_obj.reinitialise_models_forecast_dd()
    print_tag = (model_builder_obj.model_type,
                 model_builder_obj.L, model_builder_obj.F[0], args.training_scheme,
                 transformation_dd[args.transformation], regression_type) if model_builder_obj.model_type in \
                ['ar', 'risk_metrics'] else (model_builder_obj.model_type,
                                                        model_builder_obj.L, model_builder_obj.F,
                args.training_scheme, transformation_dd[args.transformation], regression_type)
    print(f'[Computation]: Compute all tables for {print_tag}...')
    """
    Generate all tables for model L, F cross
    """
    if model_builder_obj.model_type in ['har', 'har_mkt', 'har_universal', 'ar', 'risk_metrics']:
        model_builder_obj.add_metrics(df=rv, agg=agg, transformation=args.transformation,
                                      regression_type=regression_type, training_scheme=args.training_scheme)
    elif model_builder_obj.model_type == 'har_cdr':
        model_builder_obj.add_metrics(df=rv, df2=cdr, agg=agg, transformation=args.transformation,
                                      regression_type=regression_type, training_scheme=args.training_scheme)
    elif model_builder_obj.model_type == 'har_csr':
        model_builder_obj.add_metrics(df=rv, df2=csr, agg=agg, transformation=args.transformation,
                                      regression_type=regression_type, training_scheme=args.training_scheme)