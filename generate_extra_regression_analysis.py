import concurrent.futures
import pdb

from model.extra_regression_analysis import objective, callback, TrainingSchemeAnalysis
import argparse

if __name__ == '__main__':

    ####################################################################################################################
    # Additional analysis for decision trees (Feature importance) and PCR (1st component analysis)
    ####################################################################################################################
    parser = argparse.ArgumentParser(description='Model Lab: Additional analysis for regressions.')
    parser.add_argument('--analysis', help='Type of analysis to compute.', type=str)
    args = parser.parse_args()
    F = ['30T', '1H', '6H', '12H']
    lookback_ls = ['1D', '1W', '1M', '6M']
    components_analysis_obj = TrainingSchemeAnalysis()
    for training_scheme in ['ClustAM', 'CAM']:
        components_analysis_obj.training_scheme = training_scheme
        for L in ['1W', '1M', '6M']:
            components_analysis_obj.L = L
            if args.analysis == 'feature_importance':
                # for model_type in ['ar', 'har', 'har_eq']:
                lookback_ls = lookback_ls[lookback_ls.index(L):1] if lookback_ls.index(L) == 0 else \
                   lookback_ls[0:lookback_ls.index(L) + 1]
                F = F + lookback_ls# if model_type not in ['ar', 'risk_metrics'] else [F[0]]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            components_analysis_obj.feature_imp, h='30T', F=F, model_type=model_type,
                            universe=components_analysis_obj.rv.columns.tolist(), transformation='log'
                        ) for model_type in ['ar', 'har', 'har_eq']
                    ]
                    # components_analysis_obj.feature_imp(h='30T', F=F, model_type=model_type,
                    #                                     universe=components_analysis_obj.rv.columns.tolist(),
                    #                                     transformation='log')
            elif args.analysis == 'coefficient_analysis':
                components_analysis_obj.coefficient_analysis()