from figures.maker import EDA
import plotly.io as pio
pio.kaleido.scope.mathjax = None

if __name__ == '__main__':
    eda_obj = EDA()
    # eda_obj.boxplot()
    # eda_obj.daily_rv()
    # eda_obj.intraday_rv()
    eda_obj.daily_mean_correlation_matrix()
    # eda_obj.daily_pairwise_correlation()
    # eda_obj.optimal_clusters()
    print('[EDA Figures]: All EDA Figures have just been generated.')