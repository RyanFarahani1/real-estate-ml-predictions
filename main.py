#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


# ---------------- Helper Functions ----------------

def impute_missing(df, target_column, predictors):
    available = df[target_column].notnull()
    if available.sum() == 0:
        return df
    from sklearn.ensemble import RandomForestRegressor
    X_train_sub = df.loc[available, predictors]
    y_train_sub = df.loc[available, target_column]
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_sub, y_train_sub)
    missing_mask = df[target_column].isnull()
    if missing_mask.sum() > 0:
        df.loc[missing_mask, target_column] = model.predict(df.loc[missing_mask, predictors])
    return df

def extract_first_ethnicity(x):
    if isinstance(x, str):
        return x.split(",")[0].split()[0]
    return x

def transform_continuous_features(df, continuous_features, skew_threshold=1):
    for col in continuous_features:
        if (df[col] >= 0).all() and abs(df[col].skew()) > skew_threshold:
            df[col] = np.log1p(df[col])
    return df

def transform_target(target, skew_threshold=1):
    if abs(target.skew()) > skew_threshold:
        return np.log1p(target), True
    return target, False

def preprocess_data(df, for_regression=False):
    df = impute_missing(df, "time_to_cbd_public_transport_town_hall_st", ["km_from_cbd"])
    df = impute_missing(df, "time_to_cbd_driving_town_hall_st", ["km_from_cbd"])
    df['date_sold'] = pd.to_datetime(df['date_sold'], format="%Y-%m-%d")
    df['year']  = df['date_sold'].dt.year
    df['month'] = df['date_sold'].dt.month
    df['day']   = df['date_sold'].dt.day
    df.drop("date_sold", axis=1, inplace=True)
    df['quarter'] = ((df['month'] - 1) // 3) + 1
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['km_drive_interaction'] = df['km_from_cbd'] * df['time_to_cbd_driving_town_hall_st']
    df['km_pt_interaction']    = df['km_from_cbd'] * df['time_to_cbd_public_transport_town_hall_st']
    df.drop(["nearest_train_station", "highlights_attractions", "ideal_for"], axis=1, inplace=True)
    df['ethnic_breakdown'] = df['ethnic_breakdown'].apply(extract_first_ethnicity)

    if not for_regression:
        df = pd.get_dummies(df, columns=["suburb", "region", "ethnic_breakdown"], drop_first=True)
    else:
        df = pd.get_dummies(df, columns=["type", "suburb", "region", "ethnic_breakdown"], drop_first=True)

    df['public_housing_pct'] = df['public_housing_pct'].str.rstrip("%").astype(float)
    if df['suburbpopulation'].dtype == object:
        df['suburbpopulation'] = df['suburbpopulation'].str.replace(',', '', regex=True).astype(float)
    return df

def safe_label_transform_full(y, le):
    return np.array([le.transform([v])[0] if v in le.classes_ else -1 for v in y])

# ---------------- Main Script ----------------

def main(train_file, test_file):
    ZID = "5432923"

  
   # PART II: Classification   
    
    df_train = pd.read_csv(train_file)
    df_train_cls = preprocess_data(df_train.copy(), for_regression=False)

    # Filter rare classes
    cls_counts = df_train_cls['type'].value_counts()
    valid = cls_counts[cls_counts >= 2].index.tolist()
    df_train_cls = df_train_cls[df_train_cls['type'].isin(valid)]

    # Encode labels and features
    y_cls_orig = df_train_cls['type']
    le = LabelEncoder().fit(y_cls_orig)
    y_cls = le.transform(y_cls_orig)
    X_cls = df_train_cls.drop('type', axis=1)
    cont_feats_cls = [c for c in X_cls.columns if X_cls[c].nunique() > 2]
    X_cls = transform_continuous_features(X_cls.copy(), cont_feats_cls)
    # 90/10 split
    X_train, X_int, y_train, y_int = train_test_split(
        X_cls, y_cls, test_size=0.1, stratify=y_cls, random_state=42
    )

    # Initialize and train classifiers
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softmax', max_depth=7,
        learning_rate=0.04, n_estimators=600,
        subsample=0.2, colsample_bytree=0.8,
        random_state=42, num_class=len(le.classes_)
    )
    lgb_clf = lgb.LGBMClassifier(
        learning_rate=0.07, max_depth=2,
        n_estimators=150, num_leaves=100,
        subsample=0.2, class_weight='balanced',
        random_state=42 , verbosity=-1
    )
    xgb_clf.fit(X_train, y_train)
    lgb_clf.fit(X_train, y_train)

    def eval_clf(model, X, y, tag):
        preds = model.predict(X)
        print(f"{tag} F1-weighted: {f1_score(y, preds, average='weighted'):.4f}")

    # Evaluate
    for model, name in [(xgb_clf, 'XGBoost'), (lgb_clf, 'LightGBM')]:
        eval_clf(model, X_train, y_train, f"{name} Train")
        eval_clf(model, X_int,  y_int,  f"{name} Internal")

    # Ensemble helper
    def ensemble_pred(X):
        p1 = xgb_clf.predict_proba(X)
        p2 = lgb_clf.predict_proba(X)
        return np.argmax(0.6 * p1 + 0.4 * p2, axis=1)

    for Xs, ys, lbl in [(X_train, y_train, 'Train'), (X_int, y_int, 'Internal')]:
        ens = ensemble_pred(Xs)
        print(f"Ensemble {lbl} F1: {f1_score(ys, ens, average='weighted'):.4f}")

    # External test for classification
    df_test = pd.read_csv(test_file)
    df_test_cls = preprocess_data(df_test.copy(), for_regression=False)

    if 'type' in df_test_cls:
        y_test_raw = df_test_cls['type'].values
        y_test = safe_label_transform_full(y_test_raw, le)
        X_test_eval = df_test_cls.drop('type', axis=1)
        X_test_eval = transform_continuous_features(X_test_eval.copy(), cont_feats_cls)
        X_test_eval = X_test_eval.reindex(columns=X_cls.columns, fill_value=0)
        eval_clf(xgb_clf, X_test_eval, y_test, 'XGBoost External')
        eval_clf(lgb_clf, X_test_eval, y_test, 'LightGBM External')
        pe = ensemble_pred(X_test_eval)
        print(f"Ensemble External F1: {f1_score(y_test, pe, average='weighted'):.4f}")

    # Write classification submission (uses models trained on 90% split)
    X_sub = df_test_cls.drop('type', axis=1) if 'type' in df_test_cls else df_test_cls.copy()
    X_sub = transform_continuous_features(X_sub.copy(), cont_feats_cls)
    X_sub = X_sub.reindex(columns=X_cls.columns, fill_value=0)
    preds_cls = ensemble_pred(X_sub)
    pd.DataFrame({
        'id': df_test['id'],
        'type': le.inverse_transform(preds_cls)
    }).to_csv(f"z{ZID}.classification.csv", index=False)
   

    # PART I: Regression         
    train_df = pd.read_csv(train_file)
    train_df = impute_missing(train_df, "time_to_cbd_public_transport_town_hall_st", ["km_from_cbd"])
    train_df = impute_missing(train_df, "time_to_cbd_driving_town_hall_st", ["km_from_cbd"])
    train_df['date_sold'] = pd.to_datetime(train_df['date_sold'], format="%Y-%m-%d")
    train_df['year']  = train_df['date_sold'].dt.year
    train_df['month'] = train_df['date_sold'].dt.month
    train_df['day']   = train_df['date_sold'].dt.day
    train_df.drop("date_sold", axis=1, inplace=True)
    train_df['quarter']     = ((train_df['month'] - 1) // 3) + 1
    train_df['month_sin']   = np.sin(2 * np.pi * train_df['month'] / 12)
    train_df['month_cos']   = np.cos(2 * np.pi * train_df['month'] / 12)
    train_df['day_sin']     = np.sin(2 * np.pi * train_df['day'] / 31)
    train_df['day_cos']     = np.cos(2 * np.pi * train_df['day'] / 31)
    train_df['km_drive_interaction'] = train_df['km_from_cbd'] * train_df['time_to_cbd_driving_town_hall_st']
    train_df['km_pt_interaction']    = train_df['km_from_cbd'] * train_df['time_to_cbd_public_transport_town_hall_st']
    train_df.drop(["nearest_train_station", "highlights_attractions", "ideal_for"], axis=1, inplace=True)
    train_df = pd.get_dummies(train_df, columns=["type", "suburb", "region"], drop_first=True)
    train_df['public_housing_pct'] = train_df['public_housing_pct'].str.rstrip("%").astype(float)
    train_df['ethnic_breakdown']   = train_df['ethnic_breakdown'].apply(extract_first_ethnicity)
    train_df = pd.get_dummies(train_df, columns=["ethnic_breakdown"], drop_first=True)
    if train_df['suburbpopulation'].dtype == object:
        train_df['suburbpopulation'] = train_df['suburbpopulation'].str.replace(',', '', regex=True).astype(float)

    X = train_df.drop("price", axis=1)
    y = train_df["price"]
    train_features = X.columns.tolist()

    # Tiny validation split
    X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
        X, y, test_size=0.002, random_state=42
    )

    continuous_features = [c for c in X_train_r.columns if X_train_r[c].nunique() > 2]
    X_train_r = transform_continuous_features(X_train_r.copy(), continuous_features)
    X_val_r   = transform_continuous_features(X_val_r.copy(),   continuous_features)

    y_train_trans, target_transformed = transform_target(y_train_r)
    if target_transformed:
        y_val_trans = np.log1p(y_val_r)
    else:
        y_train_trans = y_train_r
        y_val_trans   = y_val_r

    best_xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.03,
        'n_estimators': 400,
        'subsample': 0.2,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    best_lgb_params = {
        'learning_rate': 0.07,
        'max_depth': 5,
        'n_estimators': 285,
        'num_leaves': 31,
        'subsample': 0.7,
        'random_state': 42
         
    }

    final_xgb = xgb.XGBRegressor(**best_xgb_params)
    final_lgb = lgb.LGBMRegressor(**best_lgb_params,verbosity=-1)
    final_xgb.fit(X_train_r, y_train_trans)
    final_lgb.fit(X_train_r, y_train_trans)
    # External test for regression
    test_df = pd.read_csv(test_file)
    test_df = impute_missing(test_df, "time_to_cbd_public_transport_town_hall_st", ["km_from_cbd"])
    test_df = impute_missing(test_df, "time_to_cbd_driving_town_hall_st", ["km_from_cbd"])
    test_df['date_sold'] = pd.to_datetime(test_df['date_sold'], format="%Y-%m-%d")
    test_df['year']  = test_df['date_sold'].dt.year
    test_df['month'] = test_df['date_sold'].dt.month
    test_df['day']   = test_df['date_sold'].dt.day
    test_df.drop("date_sold", axis=1, inplace=True)
    test_df['quarter']     = ((test_df['month'] - 1) // 3) + 1
    test_df['month_sin']   = np.sin(2 * np.pi * test_df['month'] / 12)
    test_df['month_cos']   = np.cos(2 * np.pi * test_df['month'] / 12)
    test_df['day_sin']     = np.sin(2 * np.pi * test_df['day'] / 31)
    test_df['day_cos']     = np.cos(2 * np.pi * test_df['day'] / 31)
    test_df['km_drive_interaction'] = test_df['km_from_cbd'] * test_df['time_to_cbd_driving_town_hall_st']
    test_df['km_pt_interaction']    = test_df['km_from_cbd'] * test_df['time_to_cbd_public_transport_town_hall_st']
    test_df.drop(["nearest_train_station", "highlights_attractions", "ideal_for"], axis=1, inplace=True)
    test_df = pd.get_dummies(test_df, columns=["type", "suburb", "region"], drop_first=True)
    test_df['public_housing_pct'] = test_df['public_housing_pct'].str.rstrip("%").astype(float)
    test_df['ethnic_breakdown']   = test_df['ethnic_breakdown'].apply(extract_first_ethnicity)
    test_df = pd.get_dummies(test_df, columns=["ethnic_breakdown"], drop_first=True)
    if test_df['suburbpopulation'].dtype == object:
        test_df['suburbpopulation'] = test_df['suburbpopulation'].str.replace(',', '', regex=True).astype(float)

    # Align and predict
    ids = test_df['id']
    y_test_actual = test_df['price']
    test_df.drop("price", axis=1, inplace=True)
    extra = set(test_df.columns) - set(train_features)
    if extra:
        test_df.drop(columns=extra, inplace=True)
    for c in set(train_features) - set(test_df.columns):
        test_df[c] = 0
    test_df = test_df[train_features]
    test_df = transform_continuous_features(test_df.copy(), continuous_features)

    pred_xgb = final_xgb.predict(test_df)
    pred_lgb = final_lgb.predict(test_df)
    if target_transformed:
        pred_xgb = np.expm1(pred_xgb)
        pred_lgb = np.expm1(pred_lgb)

    print("XGBoost Test MAE:      {:.2f}".format(mean_absolute_error(y_test_actual, pred_xgb)))
    print("LightGBM Test MAE:     {:.2f}".format(mean_absolute_error(y_test_actual, pred_lgb)))
    w_xgb, w_lgb = 0.2, 0.7
    w_rest = 1 - w_xgb - w_lgb
    ensemble_reg = (
        w_xgb * pred_xgb
      + w_lgb * pred_lgb
      + w_rest * ((w_xgb * pred_xgb) + (w_lgb * pred_lgb)) / 2
    )
    print("Ensemble Test MAE:     {:.2f}".format(mean_absolute_error(y_test_actual, ensemble_reg)))

    # Save regression submission
    pd.DataFrame({
        'id': ids,
        'price': ensemble_reg
    }).to_csv(f"z{ZID}.regression.csv", index=False)

    print("Done! Files written:")
    print(f"  z{ZID}.classification.csv")
    print(f"  z{ZID}.regression.csv")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        train_path, test_path = "train.csv", "test.csv"
    else:
        train_path, test_path = sys.argv[1], sys.argv[2]
    main(train_path, test_path)