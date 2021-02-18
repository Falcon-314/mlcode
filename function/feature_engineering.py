#モジュールのインポート
import pandas as pd
import numpy as np
from tqdm import tqdm

#BaseBlockを継承して特徴量を作成する
class BaseBlock(object):
    def fit(self,input_df,y=None):
        return self.transform(input_df)
    
    def transform(self,input_df):
        raise NotImplementedError()
        
#WrapperBlock
class WrapperBlock(BaseBlock):
    def __init__(self,function):
        self.function=function
    
    def transform(self,input_df):
        return self.function(input_df)

#fitとtransformの分岐処理
def get_function(block,is_train):
    s = mapping ={
        True:'fit',
        False:'transform'
    }.get(is_train)
    return getattr(block,s)

#feature_enginneringの実行
def to_feature(input_df,remain_df,blocks,is_train=False):
    out_df = remain_df
    
    for block in tqdm(blocks,total=len(blocks)):
        func = get_function(block,is_train)
        _df = func(input_df)
        
        #errorチェック
        assert len(_df) == len(input_df),func._name_
        
        out_df = pd.concat([out_df,_df],axis=1)
    return out_df    
    
    
 #----基本的なfeature engineering---#

#Aggregation_mean
class MeanBlock(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df = input_df[self.cols].groupby(input_df[self.key]).mean()
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Mean_')
        return out_df    
    
#Aggregation_std
class StdBlock(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df = input_df[self.cols].groupby(input_df[self.key]).std()
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Std_')
        return out_df
    
#Aggregation_sum
class SumBlock(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df = input_df[self.cols].groupby(input_df[self.key]).sum()
        _df = (_df.T / _df.sum(axis=1)).T #単純に合計すると値が大きくなりすぎるので規格化する
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Sum_')
        return out_df    

#Aggregation_max
class MaxBlock(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df = input_df[self.cols].groupby(input_df[self.key]).max()
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Max_')
        return out_df

#Aggregation_min
class MinBlock(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df = input_df[self.cols].groupby(input_df[self.key]).min()
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Max_')
        return out_df    
    
#Aggregation_count
class CountBlock(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df = input_df[self.cols].groupby(input_df[self.key]).count()
        _df = (_df.T / _df.sum(axis=1)).T #単純に合計すると値が大きくなりすぎるので規格化する
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Count_')
        return out_df    
    
#カテゴリ変数の組み合わせ
class CombinationBlock(BaseBlock):
    
    def __init__(self,col1,col2):
        self.lag = lag
        self.col1 = col1
        self.col2 = col2

    def transform(self,input_df):
        remain_df = input_df.copy()
        remain_df[self.col1 + 'and' + self.col2] = input_df[self.col1].astype('str') + input_df[self.col2].astype('str')
        return remain_df[self.col1 + 'and' + self.col2]
       
#ビニング
class BinningBlock(BaseBlock):
    
    def __init__(self,col,edges):
        self.lag = lag
        self.col = col
        self.edges = edges

    def transform(self,input_df):
        remain_df = input_df.copy()
        remain_df[col + '_bin'] = pd.cut(input_df[self.col], edges, labels = False)
        return remain_df[col + '_bin']
    
#n期前のLag特徴量
class LagBlock(BaseBlock):
    
    def __init__(self,lag:int,ids,cols):
        self.lag = lag
        self.ids = ids
        self.cols = cols

    def transform(self,input_df):
        output_df = input_df.groupby(self.ids)[self.cols].lag(self.lag)
        return output_df.add_prefix('Lag_{}_'.format(self.lag))
    
    
#n期前との差分
class Lag_DiffBlock(BaseBlock):
    
    def __init__(self,lag:int,ids,cols):
        self.lag = lag
        self.ids = ids
        self.cols = cols

    def transform(self,input_df):
        output_df = input_df.groupby(self.ids)[self.cols].diff(self.lag)
        return output_df.add_prefix('Lag_{}_'.format(self.lag))

