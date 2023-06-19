import pdb
import typing
from plots.maker import PlotResults
import argparse


if __name__ == '__main__':
    ####################################################################################################################
    ### Save plots
    ####################################################################################################################
    parser = argparse.ArgumentParser(description='Model Lab: Fit and store model results of research project 1.')
    parser.add_argument('--cross', default=False, help='Consider cross effect or not.', type=bool)
    parser.add_argument('--save', default=True, help='Whether to save plots or not.', type=bool)
    parser.add_argument('--models', default=['har', 'har_dummy_markets', 'har_cdr', 'har_csr', 'har_universal'],
                        help='List of models to fit.')
    parser.add_argument('--F', default=['30T', '1H', '6H', '12H'],
                        help='Set of initial features for har components of the models.')
    parser.add_argument('--L', default='1D', help='Lookback window.')
    parser.add_argument('--models_excl', default=.9, help='Models to be excluded from plots.',
                        type=typing.Union[str, list])
    parser.add_argument('--regression_type', default='linear', help='For cross whether to allow for ensemble or not.')
    parser.add_argument('--smoother_freq', default='1W', help='Frequency to use to smoothen results.')
    parser.add_argument('--transformation', default=None, help='Transformation to apply on features.')
    parser.add_argument('--test', default=False, help='Test code or not.', type=bool)
    args = parser.parse_args()
    if isinstance(args.models, str):
        models = [args.models]
    else:
        models = args.models
    if isinstance(args.models_excl, str):
        models_excl = [args.models_excl]
    else:
        models_excl = args.models_excl
    save = args.save
    regression_type = args.regression_type
    cross_name_dd = {False: 'not_crossed', True: 'cross'}
    transformation_dd = {None: 'level', 'log': 'log'}
    test = args.test
    plot_maker_obj = PlotResults
    plot_maker_obj.scatterplot(save=save, L=args.L, cross=cross_name_dd[args.cross],
                               transformation=transformation_dd[args.transformation],
                               test=test, regression_type=regression_type, models_excl=None)
    plot_maker_obj.rolling_metrics(save=save, L=args.L, cross=cross_name_dd[args.cross],
                                   transformation=transformation_dd[args.transformation],
                                   test=test, regression_type=regression_type, models_excl=None)
    plot_maker_obj.distribution(save=save, L=args.L, cross=cross_name_dd[args.cross],
                                transformation=transformation_dd[args.transformation],
                                test=test, regression_type=regression_type, models_excl=None)
    plot_maker_obj.rolling_metrics_barplot(save=save, L=args.L, cross=cross_name_dd[args.cross],
                                           transformation=transformation_dd[args.transformation],
                                           test=test, regression_type=regression_type, models_excl=None)
    if args.cross & (len(set(models).intersection(set('har_universal'))) > 0):
        pass
    else:
        plot_maker_obj.coefficient(save=save, L=args.L, cross=cross_name_dd[args.cross],
                                   transformation=transformation_dd[args.transformation],
                                   test=test, regression_type=regression_type, models_excl=None)
    print(f'[Plots]: All plots with {args.L} lookback window have been generated and saved.')