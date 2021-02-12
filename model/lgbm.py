import lightgbm as lgb
import pickle
import time
import pandas as pd

def train_lgb(folds, fold, param, features, target_col, LOGGER, get_score):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # dataset
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    trainData = lgb.Dataset(train_folds[features],train_folds[target_col])
    validData = lgb.Dataset(valid_folds[features],valid_folds[target_col])

    # ====================================================
    # train
    # ====================================================
   
    start_time = time.time()
    
    # train
    model = lgb.train(param,
                  trainData,
                  valid_sets = [trainData, validData],
                  num_boost_round = 10000,
                  early_stopping_rounds = 100,
                  verbose_eval = -1)

    # eval
    y_pred_valid = model.predict(valid_folds[features])
            
    # scoring
    score = get_score(valid_folds[target_col], y_pred_valid)

    elapsed = time.time() - start_time

    LOGGER.info(f'Score: {score} - time: {elapsed:.0f}s')

    # modelのsave
    pickle.dump(model, open(OUTPUT_DIR+f'lgbm_fold{fold}.sav','wb'))
    
    # 出力用データセットへの代入
    valid_folds['preds'] = y_pred_valid
    
    #重要度の出力
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = model.feature_importance()
    fold_importance_df["fold"] = fold

    return valid_folds, fold_importance_df

def inference_lgb(test, features):
    model = pickle.load(open(OUTPUT_DIR+f'lgbm_fold{fold}.sav','rb'))
    y_preds = model.predict(test[features])
    return y_preds
