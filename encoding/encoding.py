#categorical encoding

#Label-Encoding
from sklearn.preprocessing import LabelEncoder
def label_encoding(df,label_cols):
    for col in label_cols:
        df[col].fillna('missing',inplace = True) #欠損値があるとエンコードできないので一度missingに変換
        le = LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
    return df 

#One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
def Onehot_encoding(df, one_cols):
    ohe = OneHotEncoder(sparse = False,categories = 'auto')
    ohe.fit(df[one_cols])
    columns = []
    for i,c in enumerate(one_cols):
        columns += [f'{c}_{v}' for v in ohe.categories_[i]]

    dummy_data = pd.DataFrame(ohe.transform(df[one_cols]),columns = columns)
    df = pd.concat([df.drop(one_cols,axis = 1),dummy_data],axis = 1)
    return df
  
#Frequency Encoding
def freq_encoding(df, freq_cols):
    for c in freq_cols:
        freq = df.query('part == "train"')[c].value_counts()
        df[c] = df[c].map(freq)
    return df

#Target Encoding
from sklearn.model_selection import KFold
def target_encoding(df, target, target_cols, folds):
  
    #trainとtestに分割  
    train = df.query('part == "train"')
    test = df.query('part == "test"')
    
    #最終提出用のデータフレームの保存
    train_save = train.copy()
                    
    # 変数をループしてtarget encoding
    for c in target_cols:
                    
        #事前に作成したfold毎に平均値を作成する
        for fold in folds:
                    
            #trainとvalidに分割
            trn_idx = train[train['fold'] != fold].index
            val_idx = train[train['fold'] == fold].index
            train_df = train.loc[trn_idx].reset_index(drop=True)
            valid_df = train.loc[val_idx].reset_index(drop=True)
                    
            # validに対するtarget encoding
            data_tmp = pd.DataFrame({c: train_df[c], 'target': train_df[target]})
            target_mean = data_tmp.groupby(c)['target'].mean()
            valid_df.loc[:, c] = valid_df[c].map(target_mean)

            # trainに対するtarget encoding
            # trainもKfoldで分割してtarget encodingする
            tmp = np.repeat(np.nan, train_df.shape[0])
            kf_encoding = KFold(n_splits=5, shuffle=True, random_state=37)
            for idx_1, idx_2 in kf_encoding.split(train_df):
                target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
                tmp[idx_2] = train_df[c].iloc[idx_2].map(target_mean)
            train_df.loc[:, c] = tmp
            
            #最終的なデータフレームに代入
            train_save.loc[trn_idx, c + f'_fold_{fold}'] = train_df.loc[:, c]
            train_save.loc[val_idx, c + f'_fold_{fold}'] = valid_df.loc[:, c]
            
        # 置換済みのカラムは不要なので削除
        train_save = train_save.drop(c,axis = 1)

    #testのtarget encoding
    for c in target_cols:
                    
        # 学習データ全体で各カテゴリにおけるtargetの平均を計算
        data_tmp = pd.DataFrame({c: train[c], 'target': train[target]})
        target_mean = data_tmp.groupby(c)['target'].mean()
                    
        # 置換
        for fold in folds:
            test.loc[:, c + f'_fold_{fold}'] = test[c].map(target_mean)
                    
        # 置換済みのカラムは不要なので削除
        test = test.drop(c,axis = 1)
    
    #trainとtestを結合して復元する               
    df = pd.concat([train_saved,test],axis = 0).reset_index(drop=True)
                    
    return df
                  
#numerical encoding

#auto-scaling
from sklearn.preprocessing import StandardScaler
def auto_scaling(df, auto_cols):
    #一時的にtrainとtestを分離
    train_tmp = df.query('part == "train"').reset_index(drop = True)
    test_tmp = df.query('part == "test"').reset_index(drop = True)

    #エンコーディングの実行
    scaler =  StandardScaler()
    scaler.fit(train_tmp[auto_cols])
    train_tmp[auto_cols] =  pd.DataFrame(scaler.transform(train_tmp[auto_cols]))
    test_tmp[auto_cols] =  pd.DataFrame(scaler.transform(test_tmp[auto_cols]))

    #trainとtestを再結合
    df = pd.concat([train_tmp,test_tmp],axis = 0).reset_index(drop = True)
    return df

#min-max scaling
from sklearn.preprocessing import MinMaxScaler
def minmac_scaling(df, minmax_cols):
    scaler = MinMaxScaler()
    df[minmax_cols] = scaler.fit_transform(df[minmax_cols])
    return df

#rank gauss:NNだとautoscalingより性能良いらしい
from sklearn.preprocessing import QuantileTransformer
def rankgauss_scaling(df,Rankgauss_cols):
    #一時的にtrainとtestを分離
    train_tmp = df.query('part == "train"').reset_index(drop = True)
    test_tmp = df.query('part == "test"').reset_index(drop = True)

    #エンコーディングの実行
    Rankgauss_scaler = QuantileTransformer(n_quantiles = 100,random_state = 37,output_distribution = 'normal')
    scaler.fit(train_tmp[Rankgauss_cols])
    train_tmp[Rankgauss_cols] =  pd.DataFrame(scaler.transform(train_tmp[Rankgauss_cols]))
    test_tmp[Rankgauss_cols] =  pd.DataFrame(scaler.transform(test_tmp[Rankgauss_cols]))

    #trainとtestを再結合
    df = pd.concat([train_tmp,test_tmp],axis = 0).reset_index(drop = True)
    return df

#対数変換
def log_transform(df, log_cols):
    for col in log_cols:
        df[col] = np.log1p(df[col])
    return df    

#定数倍
def multiple_transform(df, multiple_cols, multiple):
    for col in multiple_cols:
        df[cols] = df[col].apply(lambda x: np.floor(x * multiple))
    return df

#Clipping
def clipping(df,clip_cols,clip_min,clip_max):
    for col in clip_cols:
        df[cols] = df[col].clip(clip_min,clip_max)
    return df
