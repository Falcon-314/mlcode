#logistic regression

from sklearn.linear_model import LogisticRegression
import pickle

def logistic_regression_train(folds, fold, param, features):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # dataset
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    
    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    # ====================================================
    # train
    # ====================================================
   
    start_time = time.time()
    
    # train
    model = LogisticRegression(C = param)
    model.fit(train_folds[features],train_folds['label'])

    elapsed = time.time() - start_time
    
    # eval
    y_pred_valid = model.predict_proba(valid_folds[features])
    valid_folds[[str(c) for c in range(5)]] = y_pred_valid
    valid_folds['preds'] = y_pred_valid.argmax(1)  
            
    # scoring
    score = get_score(valid_folds['label'], valid_folds['preds'])   
    
    LOGGER.info(f'Score: {score} - time: {elapsed:.0f}s')
    
    # model save
    pickle.dump(model, open(OUTPUT_DIR+f'logistic_regression_fold{fold}.sav','wb'))

    return valid_folds

def logistic_regression_inference(test, states):
    avg_preds = []
    for state in states: 
        y_preds = state.predict_proba(test)
        avg_preds.append(y_preds)
    avg_preds = np.mean(avg_preds, axis=0)
    return avg_preds
