from figures.maker import EDA

if __name__ == '__main__':
    eda_obj = EDA(title=False)
    #eda_obj.boxplot()
    #eda_obj.daily_rv()
    eda_obj.intraday_rv()
    eda_obj.daily_correlation_mean_matrix()
    #eda_obj.daily_pairwise_correlation()
    eda_obj.optimal_clusters()
