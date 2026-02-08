
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

def predict_current_density_all_organic(
    train_dataset_path,
    test_dataset_path,
    meta_train_path,
    target_column,
    baseline_models_dir_BF1,
    baseline_models_dir_BF2,
    meta_model_path,
    output_csv_path,
):

    _ , MF1_test = meta_dataset_preparation(
        train_dataset_path, test_dataset_path,
        baseline_models_dir=baseline_models_dir_BF1,
        drop_features=None,
        target_column=target_column
    )

    _ , MF2_test = meta_dataset_preparation(
        train_dataset_path, test_dataset_path,
        baseline_models_dir=baseline_models_dir_BF2,
        drop_features=["Substrate concentration", "Reactor working volume"],
        target_column=target_column
    )

    MF1_test = MF1_test.rename(columns={col: f"{col}_stk" for col in MF1_test.columns if col != target_column})
    MF2_test = MF2_test.drop(columns=target_column, axis=1).rename(columns=lambda x: f"{x}_corr_stk")

    MF5_test = pd.concat([MF2_test, MF1_test], axis=1)

    MF5_train = pd.read_csv(meta_train_path, header=0)

    MF5_test = MF5_test.reindex(columns=MF5_train.columns)

    MF5_test_X = MF5_test.drop(columns=target_column, axis=1)
    MF5_train_X = MF5_train.drop(columns=target_column, axis=1)
    MF5_train_y = MF5_train[target_column]

    scaler = MinMaxScaler()
    MF5_train_X_norm = scaler.fit_transform(MF5_train_X)
    MF5_test_X_norm = scaler.transform(MF5_test_X)  

    meta_model = joblib.load(meta_model_path)
    print("Meta model loaded:", meta_model)

    predictions = meta_model.predict(MF5_test_X_norm)
    predictions = np.expm1(predictions) 

    output_df = pd.DataFrame({'Current density': predictions})
    output_df.to_csv(output_csv_path, index=False)

    return predictions

def predict_current_density_acetate(
    train_dataset_path,
    test_dataset_path,
    meta_train_path,
    target_column,
    baseline_models_dir_BF1,
    baseline_models_dir_BF2,
    meta_model_path,
    output_csv_path,
):

    _, MF1_test = meta_dataset_preparation(
        train_dataset_path, test_dataset_path,
        baseline_models_dir=baseline_models_dir_BF1,
        drop_features=None,
        target_column=target_column
    )

    _, MF2_test = meta_dataset_preparation(
        train_dataset_path, test_dataset_path,
        baseline_models_dir=baseline_models_dir_BF2,
        drop_features=["S/V ratio", "Temperature"],
        target_column=target_column
    )


    MF1_test = MF1_test.rename(columns={col: f"{col}_stk" for col in MF1_test.columns if col != target_column})
    MF2_test = MF2_test.drop(columns=target_column).rename(columns=lambda x: f"{x}_corr_stk")

    MF5_test = pd.concat([MF2_test, MF1_test], axis=1)


    MF5_train = pd.read_csv(meta_train_path, header=0)

    MF5_test = MF5_test.reindex(columns=MF5_train.columns)

    MF5_test_X = MF5_test.drop(columns=target_column, axis=1)
    MF5_train_X = MF5_train.drop(columns=target_column, axis=1)
    MF5_train_y = MF5_train[target_column]

    scaler = MinMaxScaler()
    MF5_train_X_norm = scaler.fit_transform(MF5_train_X)
    MF5_test_X_norm = scaler.transform(MF5_test_X)  

    meta_model = joblib.load(meta_model_path)
    print("Meta model loaded:", meta_model)


    predictions = meta_model.predict(MF5_test_X_norm)
    predictions = np.expm1(predictions)

    output_df = pd.DataFrame({'Current density': predictions})
    output_df.to_csv(output_csv_path, index=False)

    return predictions


def predict_current_density_complex_substrate(
    train_dataset_path,
    test_dataset_path,
    meta_train_path,
    target_column,
    baseline_models_dir_BF1,
    baseline_models_dir_BF2,
    baseline_models_dir_BF3,
    meta_model_path,
    output_csv_path,
):
    _, MF1_test = meta_dataset_preparation(train_dataset_path,
                                                                  test_dataset_path,
                                                                    baseline_models_dir=baseline_models_dir_BF1,
                                                                      drop_features=None, 
                                                                        target_column=target_column)
    
    _, MF2_test = meta_dataset_preparation(train_dataset_path,
                                                                  test_dataset_path,
                                                                    baseline_models_dir=baseline_models_dir_BF2,
                                                                      drop_features=["Applied voltage", "Temperature"], 
                                                                        target_column=target_column)
    

    _, MF3_test = meta_dataset_preparation(train_dataset_path,
                                                                test_dataset_path,
                                                                    baseline_models_dir=baseline_models_dir_BF3,
                                                                        drop_features=["Cathode projected surface area", "S/V ratio"], 
                                                                            target_column=target_column)
    

    


    MF1_test = MF1_test.rename(columns={col: f"{col}_stk" for col in MF1_test.columns if col not in target_column})

    MF2_test = MF2_test.drop(columns= target_column, axis=1)
    MF3_test = MF3_test.drop(columns= target_column, axis=1)


    MF2_test = MF2_test.rename(columns=lambda x: f"{x}_corr_stk")
    MF3_test = MF3_test.rename(columns=lambda x: f"{x}_fi_stk")

    MF4_test = pd.concat([MF1_test, MF2_test, MF3_test], axis=1)


    MF4_train = pd.read_csv(meta_train_path, header=0)


    MF4_test = MF4_test.reindex(columns=MF4_train.columns)

    MF4_test_X = MF4_test.drop(columns=target_column, axis=1)
    MF4_train_X = MF4_train.drop(columns=target_column, axis=1)
    MF4_train_y = MF4_train[target_column]

    scaler = MinMaxScaler()
    MF4_train_X_norm = scaler.fit_transform(MF4_train_X)
    MF4_test_X_norm = scaler.transform(MF4_test_X)  


    meta_model = joblib.load(meta_model_path)
    print("Meta model loaded:", meta_model)

    predictions = meta_model.predict(MF4_test_X_norm)
    predictions = np.expm1(predictions)

    output_df = pd.DataFrame({

        'Current density': predictions
    })
    output_df.to_csv(output_csv_path, index=False)

    return predictions

def current_density_all_organic(test_csv_path):
    return predict_current_density_all_organic(
        train_dataset_path="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_dataset/Current_density/benchmark-dataset/cd_all_organic_train.csv",
        test_dataset_path=test_csv_path,
        meta_train_path="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_dataset/Current_density/meta-feature-train/all-organic/MF-5-train.csv",
        target_column="Current density",
        baseline_models_dir_BF1="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_models/Current_density/all-organic/baseline-models/BF-1/*.joblib",
        baseline_models_dir_BF2="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_models/Current_density/all-organic/baseline-models/BF-2/*.joblib",
        meta_model_path="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_models/Current_density/all-organic/meta-model/MetaHydroPred-current-density-all-organic.joblib",
        output_csv_path="temp_pred_all_organic.csv"
    )

def current_density_acetate(test_csv_path):
    return predict_current_density_acetate(
        train_dataset_path="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_dataset/Current_density/benchmark-dataset/cd_acetate_train.csv",
        test_dataset_path=test_csv_path,
        meta_train_path="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_dataset/Current_density/meta-feature-train/acetate/MF-5-train.csv",
        target_column="Current density",
        baseline_models_dir_BF1="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_models/Current_density/acetate/baseline-models/BF-1/*.joblib",
        baseline_models_dir_BF2="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_models/Current_density/acetate/baseline-models/BF-2/*.joblib",
        meta_model_path="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_models/Current_density/acetate/meta-model/MetaHydroPred-current-density-acetate.joblib",
        output_csv_path="temp_pred_acetate.csv"
    )

def current_density_complex_substrate(test_csv_path):
    return predict_current_density_complex_substrate(
        train_dataset_path="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_dataset/Current_density/benchmark-dataset/cd_complex_substance_train.csv",
        test_dataset_path=test_csv_path,
        meta_train_path="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_dataset/Current_density/meta-feature-train/complex-substrate/MF-4-train.csv",
        target_column="Current density",
        baseline_models_dir_BF1="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_models/Current_density/complex-substance/baseline-models/BF-1/*.joblib",
        baseline_models_dir_BF2="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_models/Current_density/complex-substance/baseline-models/BF-2/*.joblib",
        baseline_models_dir_BF3="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_models/Current_density/complex-substance/baseline-models/BF-3/*.joblib",
        meta_model_path="/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_models/Current_density/complex-substance/meta-model/MetaHydroPred-current-density-complex-substrate.joblib",
        output_csv_path="temp_pred_complex.csv"
    )

def get_current_density_prediction(data_csv_path, type="complex-substrate"):
    if type == "all-organic":
        return current_density_all_organic(data_csv_path)
    elif type == "acetate":
        return current_density_acetate(data_csv_path)
    elif type == "complex-substrate":
        return current_density_complex_substrate(data_csv_path)
    else:
        raise ValueError("Type must be one of: 'all-organic', 'acetate', 'complex-substrate'")

def main():
    parser = argparse.ArgumentParser(
        description='MetaHydroPred: Predict Current Density using Meta-Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=['all-organic', 'acetate', 'complex-substrate'],
        help='Type of substrate for current density prediction'
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
        default='current_density_predictions.csv',
        help='Path to output CSV file for predictions (default: current_density_predictions.csv)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    print("="*60)
    print("MetaHydroPred - Current Density Prediction")
    print("="*60)
    print(f"Substrate Type: {args.type}")
    print(f"Input File: {args.input}")
    print(f"Output File: {args.output}")
    print("="*60)
    
    try:
        if args.type == "all-organic":
            predictions = current_density_all_organic(args.input)
            if os.path.exists("temp_pred_all_organic.csv"):
                os.rename("temp_pred_all_organic.csv", args.output)
                
        elif args.type == "acetate":
            predictions = current_density_acetate(args.input)
            if os.path.exists("temp_pred_acetate.csv"):
                os.rename("temp_pred_acetate.csv", args.output)
                
        elif args.type == "complex-substrate":
            predictions = current_density_complex_substrate(args.input)
            if os.path.exists("temp_pred_complex.csv"):
                os.rename("temp_pred_complex.csv", args.output)
        
        print("\n" + "="*60)
        print("PREDICTION COMPLETE")
        print("="*60)
        print(f"Total predictions: {len(predictions)}")
        print(f"Mean Current Density: {predictions.mean():.4f}")
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
        results = get_current_density_prediction(
            "/home/vinoth/SKKU-2026-Projects/MetaHydroPred_Prog/MetaHydroPred_dataset/Current_density/benchmark-dataset/cd_acetate_test.csv",
            type="acetate"
        )
        print(results)
