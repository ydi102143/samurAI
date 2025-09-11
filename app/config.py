# app/config.py — 段階解放テーブルとユーティリティ
from __future__ import annotations
from dataclasses import dataclass
from typing import List

# レベル 1–9：どのレベルでも「学習・評価」は可能にしておく（方針）
LEVEL_FEATURES = {
    1: ["EDA", "Preprocess", "Model", "Train", "Evaluate", "Submit"],
    2: ["EDA", "Preprocess", "Model", "Train", "Evaluate", "Submit", "Predict"],
    3: ["EDA", "Preprocess", "Model", "Train", "Evaluate", "Submit", "OneHot"],
    4: ["EDA", "Preprocess", "Model", "Train", "Evaluate", "Submit", "OneHot", "Scaling", "CV"],
    5: ["EDA", "Preprocess", "Model", "Train", "Evaluate", "Submit", "OneHot", "Scaling", "CV", "Hyperparam"],
    6: ["EDA", "Preprocess", "Model", "Train", "Evaluate", "Submit", "OneHot", "Scaling", "CV", "Hyperparam", "Boosting"],
    7: ["EDA", "Preprocess", "Model", "Train", "Evaluate", "Submit", "OneHot", "Scaling", "CV", "Hyperparam", "Boosting", "Ensemble"],
    8: ["EDA", "Preprocess", "Model", "Train", "Evaluate", "Submit", "OneHot", "Scaling", "CV", "Hyperparam", "Boosting", "Ensemble", "Stacking"],
    9: ["EDA", "Preprocess", "Model", "Train", "Evaluate", "Submit", "OneHot", "Scaling", "CV", "Hyperparam", "Boosting", "Ensemble", "Stacking"],
}

# モデルの段階解放（分類/回帰共通の考え方）
CLS_LEVEL_MODELS = {
    1: ["LogisticRegression"],
    2: ["LogisticRegression", "DecisionTree"],
    3: ["LogisticRegression", "DecisionTree", "RandomForest"],
    4: ["LogisticRegression", "DecisionTree", "RandomForest", "KNN"],
    5: ["LogisticRegression", "DecisionTree", "RandomForest", "KNN", "SVM"],
    6: ["LogisticRegression", "DecisionTree", "RandomForest", "KNN", "SVM", "GradientBoosting"],
    7: ["LogisticRegression", "DecisionTree", "RandomForest", "KNN", "SVM", "GradientBoosting", "Ensemble"],
    8: ["LogisticRegression", "DecisionTree", "RandomForest", "KNN", "SVM", "GradientBoosting", "Ensemble", "Stacking"],
    9: ["LogisticRegression", "DecisionTree", "RandomForest", "KNN", "SVM", "GradientBoosting", "Ensemble", "Stacking"],
}

REG_LEVEL_MODELS = {
    1: ["LinearRegression"],
    2: ["LinearRegression", "DecisionTreeRegressor"],
    3: ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor"],
    4: ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "KNNRegressor"],
    5: ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "KNNRegressor", "SVR"],
    6: ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "KNNRegressor", "SVR", "GradientBoostingRegressor"],
    7: ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "KNNRegressor", "SVR", "GradientBoostingRegressor", "Ensemble"],
    8: ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "KNNRegressor", "SVR", "GradientBoostingRegressor", "Ensemble", "Stacking"],
    9: ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "KNNRegressor", "SVR", "GradientBoostingRegressor", "Ensemble", "Stacking"],
}

def allowed_features(level: int) -> List[str]:
    level = max(1, min(9, int(level or 1)))
    return LEVEL_FEATURES[level][:]

def allowed_models(level: int, task_type: str) -> List[str]:
    level = max(1, min(9, int(level or 1)))
    if (task_type or "classification") == "regression":
        return REG_LEVEL_MODELS[level][:]
    return CLS_LEVEL_MODELS[level][:]
