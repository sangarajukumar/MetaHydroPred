
# IMPORT LIBRARIES
import os
import gc
import json
import argparse
import scipy
import pandas as pd
import numpy as np
import joblib
import sklearn
from collections import defaultdict

# import seaborn as sns
# import matplotlib.pyplot as plt

import xgboost
import catboost
import lightgbm
import glob
import pprint

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold

# IMPORT MODElS
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,  HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, PowerTransformer, QuantileTransformer, MaxAbsScaler

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures


from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score, explained_variance_score,
                             max_error, median_absolute_error, mean_squared_log_error)
from scipy.stats import pearsonr

import warnings

warnings.filterwarnings('ignore')

def generate_meta_features(X_train, y_train, X_test, y_test, saved_baseline_models_dir):

    model_list = glob.glob(saved_baseline_models_dir)

    base_models = []

    for model_path in model_list:

        model_name = '_'.join(os.path.basename(model_path).replace('.joblib', '').split('_')[:2])
        model = joblib.load(model_path)
        base_models.append((model_name, model))

    X_train_values = X_train
    y_train_values = y_train.values
    X_test_values = X_test


    meta_features_train = np.zeros((X_train_values.shape[0], len(base_models)))
    meta_features_test = np.zeros((X_test_values.shape[0], len(base_models)))
    
    loo = LeaveOneOut()
    
    for i, (name, model) in enumerate(base_models):

        model = clone(model)

        print(model)

        oof_predictions = np.zeros(X_train.shape[0])
        test_predictions = np.zeros((X_test.shape[0], X_train.shape[0]))
        
        for j, (train_idx, valid_idx) in enumerate(loo.split(X_train)):
            X_train_fold, X_valid_fold = X_train_values[train_idx], X_train_values[valid_idx]
            y_train_fold, y_valid_fold = y_train_values[train_idx], y_train_values[valid_idx]
            
            print(f"Fitting model: {name}")
            
            model.fit(X_train_fold, y_train_fold)
            
            oof_predictions[valid_idx] = model.predict(X_valid_fold)
            
            test_predictions[:, j] = model.predict(X_test_values)
        
        meta_features_train[:, i] = oof_predictions
        
        meta_features_test[:, i] = test_predictions.mean(axis=1)
    
    meta_features_train_df = pd.DataFrame(meta_features_train, columns=[name for name, _ in base_models])
    meta_features_test_df = pd.DataFrame(meta_features_test, columns=[name for name, _ in base_models])
    
    X_train_combined = pd.concat([meta_features_train_df, y_train], axis=1)
    X_test_combined = pd.concat([meta_features_test_df, y_test], axis=1)
    
    return X_train_combined, X_test_combined

def meta_dataset_preparation(train_dataset_path, test_dataset_path, baseline_models_dir, drop_features=None, target_column='current_density'):

    if train_dataset_path is not None:
        train_data = pd.read_csv(train_dataset_path, header=0)
    else:
        raise ValueError("Training dataset path must be provided.")
    if test_dataset_path is not None:
        test_data = pd.read_csv(test_dataset_path, header=0)
    else:
        raise ValueError("Test dataset path must be provided.")
    
    if drop_features is not None:

        train_data = train_data.drop(columns=drop_features, errors='ignore')
        test_data = test_data.drop(columns=drop_features, errors='ignore')

    train_X = train_data.drop(columns=[target_column])
    train_y = train_data[target_column]

    test_X = test_data.drop(columns=[target_column])
    test_y = test_data[target_column]


    scaler = MinMaxScaler()
    train_X_norm = scaler.fit_transform(train_X)
    test_X_norm = scaler.transform(test_X)

    train_y_norm = train_y.apply(np.log1p)
    test_y_norm = test_y.apply(np.log1p)

    meta_train, meta_test = generate_meta_features(train_X_norm, train_y_norm, test_X_norm, test_y_norm, baseline_models_dir)

    return meta_train, meta_test

def predict_h2_production_rate_all_organic(
    train_dataset_path,
    test_dataset_path,
    meta_train_path,
    target_column,
    baseline_models_dir_BF1,
    baseline_models_dir_BF3,
    meta_model_path,
    output_csv_path,
):

    _ , MF1_test = meta_dataset_preparation(
        train_dataset_path, test_dataset_path,
        baseline_models_dir=baseline_models_dir_BF1,
        drop_features=None,
        target_column=target_column
    )

    _ , MF3_test = meta_dataset_preparation(
        train_dataset_path, test_dataset_path,
        baseline_models_dir=baseline_models_dir_BF3,
        drop_features=["S/V ratio", "Temperature"],
        target_column=target_column
    )


    MF1_test = MF1_test.rename(columns={col: f"{col}_stk" for col in MF1_test.columns if col != target_column})
    MF3_test = MF3_test.drop(columns=target_column, axis=1).rename(columns=lambda x: f"{x}_fi_stk")


    MF6_test = pd.concat([MF3_test, MF1_test], axis=1)


    MF6_train = pd.read_csv(meta_train_path, header=0)

    MF6_test = MF6_test.reindex(columns=MF6_train.columns)

    MF6_test_X = MF6_test.drop(columns=target_column, axis=1)
    MF6_train_X = MF6_train.drop(columns=target_column, axis=1)
    MF6_train_y = MF6_train[target_column]

    scaler = MinMaxScaler()
    MF6_train_X_norm = scaler.fit_transform(MF6_train_X)
    MF6_test_X_norm = scaler.transform(MF6_test_X)  

    meta_model = joblib.load(meta_model_path)
    print("Meta model loaded:", meta_model)

    predictions = meta_model.predict(MF6_test_X_norm)
    predictions = np.expm1(predictions)

    output_df = pd.DataFrame({'H2 production rate': predictions})
    output_df.to_csv(output_csv_path, index=False)

    return predictions

def predict_h2_production_rate_acetate(
    train_dataset_path,
    test_dataset_path,
    meta_train_path,
    target_column,
    baseline_models_dir_BF1,
    meta_model_path,
    output_csv_path,
):

    _, MF1_test = meta_dataset_preparation(
        train_dataset_path, test_dataset_path,
        baseline_models_dir=baseline_models_dir_BF1,
        drop_features=None,
        target_column=target_column
    )


    MF1_train = pd.read_csv(meta_train_path, header=0)

    MF1_test = MF1_test.reindex(columns=MF1_train.columns)

    MF1_test_X = MF1_test.drop(columns=target_column, axis=1)
    MF1_train_X = MF1_train.drop(columns=target_column, axis=1)
    MF1_train_y = MF1_train[target_column]

    scaler = MinMaxScaler()
    MF1_train_X_norm = scaler.fit_transform(MF1_train_X)
    MF1_test_X_norm = scaler.transform(MF1_test_X)  


    meta_model = joblib.load(meta_model_path)
    print("Meta model loaded:", meta_model)


    predictions = meta_model.predict(MF1_test_X_norm)
    predictions = np.expm1(predictions) 

    output_df = pd.DataFrame({'H2 Production Rate': predictions})
    output_df.to_csv(output_csv_path, index=False)

    return predictions

def predict_h2_production_rate_complex_substrate(
    train_dataset_path,
    test_dataset_path,
    meta_train_path,
    target_column,
    baseline_models_dir_BF1,
    meta_model_path,
    output_csv_path,
):
    _, MF1_test = meta_dataset_preparation(train_dataset_path,
                                                                  test_dataset_path,
                                                                    baseline_models_dir=baseline_models_dir_BF1,
                                                                      drop_features=None, 
                                                                        target_column=target_column)
    
    
    MF1_train = pd.read_csv(meta_train_path, header=0)

    MF1_test = MF1_test.reindex(columns=MF1_train.columns)

    MF1_test_X = MF1_test.drop(columns=target_column, axis=1)
    MF1_train_X = MF1_train.drop(columns=target_column, axis=1)
    MF1_train_y = MF1_train[target_column]

    scaler = MinMaxScaler()
    MF1_train_X_norm = scaler.fit_transform(MF1_train_X)
    MF1_test_X_norm = scaler.transform(MF1_test_X)  

    meta_model = joblib.load(meta_model_path)
    print("Meta model loaded:", meta_model)

    predictions = meta_model.predict(MF1_test_X_norm)
    predictions = np.expm1(predictions)

    output_df = pd.DataFrame({
        'H2 Production Rate': predictions
    })
    output_df.to_csv(output_csv_path, index=False)

    return predictions

def h2_production_rate_all_organic(test_csv_path):
    return predict_h2_production_rate_all_organic(
        train_dataset_path="dataset/H2_Production_Rate/benchmark-dataset/h2_all_organic_train.csv",
        test_dataset_path=test_csv_path,
        meta_train_path="dataset/H2_Production_Rate/meta-feature-train/all-organic/MF-6-train.csv",
        target_column="H2 production rate",
        baseline_models_dir_BF1="models/H2_Production_Rate/all-organic/baseline-models/BF-1/*.joblib",
        baseline_models_dir_BF3="models/H2_Production_Rate/all-organic/baseline-models/BF-3/*.joblib",
        meta_model_path="models/H2_Production_Rate/all-organic/meta-model/MetaHydroPred-h2-production-rate-all-organic.joblib",
        output_csv_path="temp_pred_all_organic.csv"
    )

def h2_production_rate_acetate(test_csv_path):
    return predict_h2_production_rate_acetate(
        train_dataset_path="dataset/H2_Production_Rate/benchmark-dataset/h2_acetate_train.csv",
        test_dataset_path=test_csv_path,
        meta_train_path="dataset/H2_Production_Rate/meta-feature-train/acetate/MF-1-train.csv",
        target_column="H2 production rate",
        baseline_models_dir_BF1="models/H2_Production_Rate/acetate/baseline-models/BF-1/*.joblib",
        meta_model_path="models/H2_Production_Rate/acetate/meta-model/MetaHydroPred-h2-production-rate-acetate.joblib",
        output_csv_path="temp_pred_acetate.csv"
    )

def h2_production_rate_complex_substrate(test_csv_path):
    return predict_h2_production_rate_complex_substrate(
        train_dataset_path="dataset/H2_Production_Rate/benchmark-dataset/h2_complex_substance_train.csv",
        test_dataset_path=test_csv_path,
        meta_train_path="dataset/H2_Production_Rate/meta-feature-train/complex-substrate/MF-1-train.csv",
        target_column="H2 production rate",
        baseline_models_dir_BF1="models/H2_Production_Rate/complex-substance/baseline-models/BF-1/*.joblib",
        meta_model_path="models/H2_Production_Rate/complex-substance/meta-model/MetaHydroPred-h2-production-rate-complex-substrate.joblib",
        output_csv_path="temp_pred_complex.csv"
    )

def get_h2_production_prediction(data_csv_path, type="complex-substrate"):
    if type == "all-organic":
        return h2_production_rate_all_organic(data_csv_path)
    elif type == "acetate":
        return h2_production_rate_acetate(data_csv_path)
    elif type == "complex-substrate":
        return h2_production_rate_complex_substrate(data_csv_path)
    else:
        raise ValueError("Type must be one of: 'all-organic', 'acetate', 'complex-substrate'")


def main():

    parser = argparse.ArgumentParser(
        description='MetaHydroPred: Predict H2 Production Rate using Meta-Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=['all-organic', 'acetate', 'complex-substrate'],
        help='Type of substrate for H2 production prediction'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file containing test data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='h2_production_predictions.csv',
        help='Path to output CSV file for predictions (default: h2_production_predictions.csv)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    print("="*60)
    print("MetaHydroPred - H2 Production Rate Prediction")
    print("="*60)
    print(f"Substrate Type: {args.type}")
    print(f"Input File: {args.input}")
    print(f"Output File: {args.output}")
    print("="*60)
    
    try:
        if args.type == "all-organic":
            predictions = h2_production_rate_all_organic(args.input)
            if os.path.exists("temp_pred_all_organic.csv"):
                os.rename("temp_pred_all_organic.csv", args.output)
                
        elif args.type == "acetate":
            predictions = h2_production_rate_acetate(args.input)
            if os.path.exists("temp_pred_acetate.csv"):
                os.rename("temp_pred_acetate.csv", args.output)
                
        elif args.type == "complex-substrate":
            predictions = h2_production_rate_complex_substrate(args.input)
            if os.path.exists("temp_pred_complex.csv"):
                os.rename("temp_pred_complex.csv", args.output)
        
        print("\n" + "="*60)
        print("PREDICTION COMPLETE")
        print("="*60)
        print(f"Total predictions: {len(predictions)}")
        print(f"Mean H2 Production Rate: {predictions.mean():.4f}")
        print(f"Results saved to: {args.output}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        print("Running default test...")
        results = get_h2_production_prediction(
            "dataset/H2_Production_Rate/benchmark-dataset/h2_complex_substance_test.csv",
            type="complex-substrate"
        )
        print(results)