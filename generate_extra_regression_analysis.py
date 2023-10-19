import concurrent.futures
import itertools
import pdb
from model.extra_regression_analysis import objective, callback, TrainingSchemeAnalysis
from figures.maker_copy import PlotResults
import argparse

if __name__ == '__main__':

    ####################################################################################################################
    # Additional analysis for decision trees (Feature importance) and PCR (1st component analysis)
    ####################################################################################################################
    maker_obj = PlotResults()
    parser = argparse.ArgumentParser(description='Model Lab: Additional analysis for regressions.')
    parser.add_argument('--analysis', help='Type of analysis to compute.', type=str)
    args = parser.parse_args()
    F = ['30T', '1H', '6H', '12H']
    lookback_ls = ['1D', '1W', '1M', '6M']
    components_analysis_obj = TrainingSchemeAnalysis()
    for training_scheme in ['CAM']:
        components_analysis_obj.training_scheme = training_scheme
        for L in ['1W']:#'6M', '1M',
            components_analysis_obj.L = L
            if args.analysis == 'feature_importance':
                # for model_type in ['har_eq', 'ar', 'har', 'risk_metrics']:
                #     components_analysis_obj.feature_imp(h='30T', F=F, model_type=model_type,
                #                                         universe=components_analysis_obj.rv.columns.tolist(),
                #                                         transformation='log')
                maker_obj.feature_importance(save=True)
            elif args.analysis == 'coefficient_analysis':
                components_analysis_obj.coefficient_analysis()
                maker_obj.first_principal_component(save=True)