from figures.maker import EDA
import plotly.io as pio
pio.kaleido.scope.mathjax = None
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Lab: Fit and store model results of research project 1.')
    parser.add_argument('--save', default=False, help='Save figures or not.', type=int)
    args = parser.parse_args()
    eda_obj = EDA()
    eda_obj.boxplot(save=args.save)
    eda_obj.daily_rv(save=args.save)
    eda_obj.intraday_rv(save=args.save)
    eda_obj.daily_mean_correlation_matrix(save=args.save)
    eda_obj.daily_pairwise_correlation(save=args.save)
    eda_obj.optimal_clusters(save=args.save)
    if args.save == 1:
        print('[EDA Figures]: All EDA Figures have just been generated.')