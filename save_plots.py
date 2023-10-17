import os.path
import pdb
import typing
from figures.maker_copy import PlotResults
import argparse
import pandas as pd

if __name__ == '__main__':
    ####################################################################################################################
    ### Save figures
    ####################################################################################################################
    parser = argparse.ArgumentParser(description='Model Lab: Fit and store model results of research project 1.')
    parser.add_argument('--training_scheme', default=False, help='Training scheme under which models are trained.',
                        type=str)
    parser.add_argument('--save', default=True, help='Whether to save figures or not.', type=bool)
    parser.add_argument('--models', default=['har', 'har_eq', 'har_cdr', 'har_csr', 'har_universal'],
                        help='List of models to fit.')
    parser.add_argument('--L', default='1D', help='Lookback window.')
    parser.add_argument('--regression_type', default='linear', help='For cross whether to allow for ensemble or not.')
    parser.add_argument('--transformation', default=None, help='Transformation to apply on features.')
    parser.add_argument('--test', default=False, help='Test code or not.', type=bool)
    args = parser.parse_args()
    if isinstance(args.models, str):
        models = [args.models]
    else:
        models = args.models
    save = args.save
    regression_type = args.regression_type
    transformation_dd = {None: 'level', 'log': 'log'}
    plot_maker_obj = PlotResults
    if not os.path.exists(f'./figures/{args.L}/'):
        os.mkdir(f'./figures/{args.L}/')
    if not os.path.exists(f'./figures/{args.L}/{args.training_scheme}/'):
        os.mkdir(f'./figures/{args.L}/{args.training_scheme}/')
    if not os.path.exists(f'./figures/{args.L}/{args.training_scheme}/{args.transformation}/'):
        os.mkdir(f'./figures/{args.L}/{args.training_scheme}/{args.transformation}/')

    plot_maker_obj.commonality(transformation=transformation_dd[args.transformation], save=args.save)
    # plot_maker_obj.distribution(L=args.L, training_scheme=args.training_scheme,
    #                             transformation=transformation_dd[args.transformation],
    #                             regression_type=regression_type, save=args.save)
    #
    # plot_maker_obj.rolling_metrics(L=args.L, training_scheme=args.training_scheme,
    #                                transformation=transformation_dd[args.transformation],
    #                                regression_type=regression_type, save=args.save)
    #
    # plot_maker_obj.rolling_metrics_barplot(L=args.L, training_scheme=args.training_scheme,
    #                                        transformation=transformation_dd[args.transformation],
    #                                        regression_type=regression_type, save=args.save)
    # plot_maker_obj.rolling_metrics(save=save, L=args.L, cross=cross_name_dd[args.cross],
    #                                transformation=args.transformation,
    #                                test=test, regression_type=regression_type, models_excl=None)
    # plot_maker_obj.distribution(save=save, L=args.L, cross=cross_name_dd[args.cross],
    #                             transformation=args.transformation,
    #                             test=test, regression_type=regression_type, models_excl=None)
    # plot_maker_obj.rolling_metrics_barplot(save=save, L=args.L, cross=cross_name_dd[args.cross],
    #                                        transformation=args.transformation,
    #                                        test=test, regression_type=regression_type, models_excl=None)
    # try:
    #     plot_maker_obj.coefficient(save=save, L=args.L, cross=cross_name_dd[args.cross],
    #                                transformation=args.transformation,
    #                                test=test, regression_type=regression_type, models_excl=None)
    # except pd.errors.DatabaseError:
    #     pass
    if save:
        print(f'[figures]: All figures with {args.L} lookback window have been generated and saved.')