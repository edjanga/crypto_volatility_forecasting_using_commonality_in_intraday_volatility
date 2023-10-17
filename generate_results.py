import os.path
import pdb
from data_centre.data import Reader
from model.training_schemes import SAM, ClustAM, CAM
import argparse

training_schemes = [SAM, ClustAM, CAM]
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
    parser.add_argument('--L', default='1D', help='Lookback window.')
    parser.add_argument('--regression_type', default='linear', help='For cross whether to allow for ensemble or not.')
    parser.add_argument('--smoother_freq', default='1W', help='Frequency to use to smoothen results.')
    parser.add_argument('--transformation', default='log', type=str, help='Transformation to apply on features.')
    args = parser.parse_args()
    rv = data_obj.rv_read(variance=True)
    F = args.F
    regression_type = args.regression_type
    training_scheme_obj = training_scheme_factory_dd[args.training_scheme]
    model_builder_obj = training_scheme_obj(F=F, h='30T', L=args.L, Q='1D', model_type=args.model,
                                            universe=rv.columns.tolist())
    agg = args.smoother_freq
    lookback_ls = ['1D', '1W', '1M', '6M']
    lookback_ls = lookback_ls[lookback_ls.index(args.L):1] if lookback_ls.index(args.L) == 0 else \
        lookback_ls[0:lookback_ls.index(args.L)+1]
    F = F + lookback_ls
    model_builder_obj.L = lookback_ls[-1]
    model_builder_obj.F = [F[0]] if model_builder_obj.model_type in ['ar', 'risk_metrics'] else F
    print_tag = (model_builder_obj.model_type, model_builder_obj.L, model_builder_obj.F, args.training_scheme,
                 args.transformation, regression_type)
    print(f'[Computation]: Compute all tables for {print_tag}...')
    """
    Generate all tables for model L, F cross
    """
    model_builder_obj.add_metrics(regression_type=args.regression_type, transformation=args.transformation, agg='1W',
                                  df=rv)
