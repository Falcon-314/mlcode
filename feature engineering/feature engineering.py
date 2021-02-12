#feature engineering - import
import pandas as pd

def aggregation(df, category, target, mode):
    if mode == 'mean':
        df_agg = pd.merge(df,df[[category, target]].groupby(category).mean().rename(columns = {target :target + '_' + category + '_mean'}), on= category, how='left')
        
    if mode == 'max':
        df_agg = pd.merge(df,df[[category, target]].groupby(category).max().rename(columns = {target :target + '_' + category + '_max'}), on= category, how='left')
    
    if mode == 'min':
        df_agg = pd.merge(df,df[[category, target]].groupby(category).min().rename(columns = {target :target + '_' + category + '_min'}), on= category, how='left')
    
    if mode == 'std':
        df_agg = pd.merge(df,df[[category, target]].groupby(category).std().rename(columns = {target :target + '_' + category + '_std'}), on= category, how='left')

    return df_agg

def combination(df,col1,col2):
    df[col1 + 'and' + col2] = df[col1].astype('str') + df[col2].astype('str')
    return df

def binning(df,col, bin_edges):
    df[col + '_bin'] = pd.cut(df[col], bin_edges, labels = False)
    return df

