import os.path
import pdb
from data_centre.data import Reader
from model.training_schemes import SAM, ClustAM, CAM, UAM
import argparse

training_schemes = [SAM, ClustAM, CAM, UAM]
training_scheme_factory_dd = {training_scheme.__name__: training_scheme for training_scheme in training_schemes}

if __name__ == '__main__':

    ####################################################################################################################
    #### Fit models
    ####################################################################################################################
    data_obj = Reader()
    parser = argparse.ArgumentParser(description='Model Lab: Fit and store model results of research project 1.')
    parser.add_argument('--training_scheme', default=False, help='Training scheme under which models are trained.',
                        type=str)
    parser.add_argument('--model', help='Model to fit.')
    parser.add_argument('--F', default=['30T', '1H', '6H', '12H'],
                        help='Set of initial features for components of the models.')
    parser.add_argument('--freq', default='1D', type=str, help='Frequency of model update.')
    parser.add_argument('--L', default='1W', help='Lookback window.')
    parser.add_argument('--regression_type', default='linear', help='For cross whether to allow for ensemble or not.')
    parser.add_argument('--transformation', default=None, type=str, help='Transformation to apply on features.')
    parser.add_argument('--trading_session', default=None, type=int, help='Trading session or VIXM for HAR_EQ.')
    parser.add_argument('--top_book', default=None, type=int, help='In case VIXM for previous param, top book or not.')
    args = parser.parse_args()
    trading_session = args.trading_session
    top_book = args.top_book
    rv = data_obj.rv_read(variance=True, cutoff_low=.05, cutoff_high=.05)
    agg = args.smoother_freq
    regression_type = args.regression_type
    training_scheme_obj = training_scheme_factory_dd[args.training_scheme]
    model_builder_obj = \
        training_scheme_obj(h='30T', L=args.L, Q='1D', model_type=args.model, universe=rv.columns.tolist())
    if args.regression_type != 'var':
        F = args.F
        lookback_ls = ['1D', '1W', '1M', '6M']
        lookback_ls = lookback_ls[lookback_ls.index(args.L):1] if lookback_ls.index(args.L) == 0 else \
            lookback_ls[0:lookback_ls.index(args.L)+1]
        F = F + lookback_ls
        model_builder_obj.F = [F[0]] if model_builder_obj.model_type in ['ar', 'risk_metrics', 'heavy'] else F
    print_tag = (model_builder_obj.model_type, model_builder_obj.L, model_builder_obj.F, args.training_scheme,
                 args.transformation, regression_type, args.trading_session)
    print(f'[Computation]: Compute all tables for {print_tag}...')
    """
    Generate all tables for model L, F cross
    """
    extra_args = {True: {'trading_session': trading_session, 'top_book': top_book}, False: {}}
    if trading_session == 0:
        vixm = Reader().rv_read(data='vixm')
        extra_args[True].update({'vixm': vixm.iloc[:, :top_book]}) if top_book == 1 else \
            extra_args[True].update({'vixm': vixm})
    model_builder_obj.add_metrics(regression_type=args.regression_type, agg='1D', df=rv,
                                  transformation=args.transformation, **extra_args[args.model == 'har_eq'],
                                  freq=args.freq)