
def custom_norm(df,count_column):
    from sklearn.preprocessing import PowerTransformer, QuantileTransformer, MinMaxScaler, Normalizer
    from scipy.stats import boxcox
    import numpy as np



    df["wp_pop_density_log"] = np.log(1+df["wp_pop_density"])

    minmax = MinMaxScaler(feature_range=(0, 1))
    df["wp_pop_density_log_minmax"] = minmax.fit_transform(df[[f"wp_pop_density_log"]])

    df[f"{count_column}_multiply_pop_density"] = df[count_column] * df["wp_pop_density_log_minmax"]

    df = df.loc[df[f"{count_column}_multiply_pop_density"] > 0]


    #df[f"{count_column}_multiply_pop_density_log"] = np.log(
    #    1 + df[f"{count_column}_multiply_pop_density"])

    #df = df.loc[df[f"{count_column}_multiply_pop_density_log"] > 0.5]

    transformed_data, best_lambda = boxcox(df[f"{count_column}_multiply_pop_density"])
    df["boxcoxb"]=transformed_data

    minmax = MinMaxScaler(feature_range=(0, 1))
    df[f"boxcoxb_log"] = minmax.fit_transform(
        df[[f"boxcoxb"]])



    #qt = QuantileTransformer(n_quantiles=5000, output_distribution='normal')
    #df[f"{count_column}_multiply_pop_density_log_q"] = qt.fit_transform(df[[f"{count_column}_multiply_pop_density_log"]])
    #minmax = MinMaxScaler(feature_range=(0, 1))
    #df[f"{count_column}_multiply_pop_density_log_minmax"] = minmax.fit_transform(
    #    df[[f"{count_column}_multiply_pop_density_log"]])



    minmax = MinMaxScaler(feature_range=(0, 1))
    df[f"boxcoxb_log_minmax"] = minmax.fit_transform(
        df[[f"boxcoxb_log"]])


    return df
