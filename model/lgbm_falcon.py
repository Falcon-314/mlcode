def train_lgb(folds, fold):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # dataset
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    trainData = lgb.Dataset(x_train_tmp,y_train_tmp)
    validData = lgb.Dataset(x_valid,y_valid)
    
    y_pred_train = np.zeros((len(x_train), ))
    y_pred_test = np.zeros((len(x_test), ))
    feature_importance_df = pd.DataFrame()

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
                  verbose_eval = -1,
                  show_progress_bar = False)

    # eval
    y_pred_valid = model.predict(x_valid)
    y_pred_train[valid_index] = y_pred_valid
            
    # scoring
    score = get_score(y_valid, y_pred_valid)

    elapsed = time.time() - start_time

    LOGGER.info(f'MAE: {score} - time: {elapsed:.0f}s')

    # modelのsave
    torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth')
    
    # 出力用データセットへの代入
    valid_folds[[str(c) for c in range(5)]] = check_point['preds']
    valid_folds['preds'] = check_point['preds'].argmax(1)
    
    #重要度の出力
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = x_train.columns
    fold_importance_df["importance"] = model.feature_importance()
    fold_importance_df["fold"] = fold_n
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    return valid_folds, feature_importance_df

def inference_lgb():

    for i in range(5):
            #testへの予測
            y_pred_test += model.predict(x_test)/nfolds
  
    return y_pred_test


def lgbtuner():
    
    return best_params
