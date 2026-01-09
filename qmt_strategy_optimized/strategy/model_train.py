#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型训练模块
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils.logger import setup_logger
import config
import os
import joblib

# 直接导入所有需要的模型
from xgboost import XGBClassifier

# 导入数据处理和特征工程模块
from .data_process import DataProcessor
from .feature_engineer import FeatureEngineer

logger = setup_logger()

class ModelWrapper:
    """模型包装器，封装模型、预处理逻辑和参数配置"""
    def __init__(self, model, data_processor=None, feature_engineer=None, config=None):
        self.model = model
        self.data_processor = data_processor
        self.feature_engineer = feature_engineer
        self.config = config or {}
        self.version = "1.0"
        self.created_at = pd.Timestamp.now().isoformat()
    
    def predict(self, data):
        """使用模型进行预测"""
        # 预处理数据
        processed_data = self.data_processor.process(data) if self.data_processor else data
        
        # 提取特征
        features = self.feature_engineer.extract_features(processed_data) if self.feature_engineer else processed_data
        
        # 分离特征和标签
        if isinstance(features, pd.DataFrame) and 'label' in features.columns:
            features = features.drop('label', axis=1)
        
        # 确保所有特征都是数值类型
        features = features.select_dtypes(include=[np.number])
        
        # 进行预测
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }

class ModelTrainer:
    """模型训练类，用于构建和训练机器学习模型"""
    
    def __init__(self):
        """初始化模型训练器"""
        logger.info("初始化模型训练器")
        
        # 创建模型保存目录
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        
        # 初始化为None，延迟加载模型
        self.model = None
    
    def _init_model(self):
        """初始化模型，添加正则化参数"""
        logger.info(f"初始化模型，模型类型：{config.MODEL_TYPE}")
        
        if config.MODEL_TYPE == "logistic":
            # 逻辑回归模型，添加L1/L2正则化
            model = LogisticRegression(
                penalty='elasticnet',  # 使用弹性网络正则化
                solver='saga',  # saga求解器支持elasticnet
                l1_ratio=0.5,  # L1正则化比例
                C=0.1,  # 正则化强度的倒数，值越小正则化越强
                max_iter=1000,  # 增加最大迭代次数
                random_state=42
            )
        elif config.MODEL_TYPE == "random_forest":
            # 随机森林模型，添加树深度、最小样本数等正则化参数
            model = RandomForestClassifier(
                n_estimators=200,  # 树的数量
                max_depth=10,  # 树的最大深度
                min_samples_split=10,  # 节点分裂所需的最小样本数
                min_samples_leaf=5,  # 叶节点所需的最小样本数
                max_features='sqrt',  # 分裂时考虑的最大特征数
                bootstrap=True,  # 使用bootstrap采样
                oob_score=True,  # 计算袋外分数
                class_weight='balanced',  # 处理不平衡数据
                random_state=42,  # 随机种子
                n_jobs=-1  # 使用所有CPU核心
            )
        elif config.MODEL_TYPE == "xgboost":
            # XGBoost模型，优化reg_alpha（L1）和reg_lambda（L2）参数
            model = XGBClassifier(
                n_estimators=200,  # 树的数量
                max_depth=6,  # 树的最大深度
                learning_rate=0.05,  # 学习率
                subsample=0.8,  # 训练样本采样比例
                colsample_bytree=0.8,  # 特征采样比例
                min_child_weight=3,  # 子节点最小权重
                gamma=0.1,  # 节点分裂所需的最小损失减少量
                reg_alpha=1.0,  # L1正则化参数
                reg_lambda=2.0,  # L2正则化参数
                scale_pos_weight=10,  # 处理不平衡数据
                objective='binary:logistic',  # 二分类目标函数
                eval_metric='auc',  # 评估指标
                random_state=42,  # 随机种子
                n_jobs=-1  # 使用所有CPU核心
            )
        else:
            # 默认使用XGBoost模型
            logger.warning(f"未知的模型类型：{config.MODEL_TYPE}，使用默认模型XGBoost")
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1
            )
        
        return model
    
    def train(self, features, labels=None):
        """训练模型"""
        logger.info("开始模型训练")
        
        if features is None or features.empty:
            logger.warning("特征数据为空，跳过模型训练")
            return None
        
        try:
            # 如果没有提供标签，则从特征数据中提取
            if labels is None:
                # 这里假设标签列名为'label'，1表示涨停，0表示非涨停
                if 'label' in features.columns:
                    labels = features['label']
                    features = features.drop('label', axis=1)
                else:
                    logger.error("特征数据中没有找到标签列'label'")
                    return None
            
            # 移除非数值特征，只保留数值特征用于模型训练
            numeric_features = features.select_dtypes(include=[np.number])
            logger.info(f"移除非数值特征，保留{len(numeric_features.columns)}个数值特征")
            features = numeric_features
            
            # 确保标签是整数类型的二分类标签（0和1）
            labels = labels.astype(int)
            # 处理多分类问题，只保留0和1
            labels = labels.clip(0, 1)
            # 检查唯一标签值
            unique_labels = np.unique(labels)
            logger.info(f"标签分布：{unique_labels}")
            
            # 计算标签分布
            label_counts = np.bincount(labels)
            logger.info(f"标签数量分布：{dict(zip(unique_labels, label_counts))}")
            
            # 确保只有两个类别
            if len(unique_labels) < 2:
                logger.warning(f"只有{len(unique_labels)}个类别，添加一些噪声以确保二分类")
                # 在少量样本中添加相反标签
                mask = np.random.choice([True, False], size=len(labels), p=[0.05, 0.95])
                labels[mask] = 1 - labels[mask]
                unique_labels = np.unique(labels)
                label_counts = np.bincount(labels)
                logger.info(f"添加噪声后标签分布：{dict(zip(unique_labels, label_counts))}")
            
            # 改进交叉验证方法：实现分层K折交叉验证
            logger.info("使用分层K折交叉验证进行模型评估")
            from sklearn.model_selection import StratifiedKFold
            
            # 划分训练集和验证集（用于快速验证）
            logger.info(f"划分训练集和验证集，比例：{config.TRAIN_VALID_RATIO}:{1-config.TRAIN_VALID_RATIO}")
            X_train, X_valid, y_train, y_valid = train_test_split(
                features, labels, test_size=1-config.TRAIN_VALID_RATIO, random_state=42, stratify=labels
            )
            
            # 分层K折交叉验证（用于更可靠的模型评估）
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
                logger.info(f"开始第{fold+1}/5折交叉验证")
                X_cv_train, X_cv_val = features.iloc[train_idx], features.iloc[val_idx]
                y_cv_train, y_cv_val = labels.iloc[train_idx], labels.iloc[val_idx]
                
                # 处理不平衡数据
                X_cv_train_resampled, y_cv_train_resampled = self._handle_imbalanced_data(X_cv_train, y_cv_train)
                
                # 训练模型
                temp_model = self._init_model()
                temp_model.fit(X_cv_train_resampled, y_cv_train_resampled)
                
                # 评估模型
                y_pred = temp_model.predict(X_cv_val)
                from sklearn.metrics import f1_score
                fold_score = f1_score(y_cv_val, y_pred)
                cv_scores.append(fold_score)
                logger.info(f"第{fold+1}/5折交叉验证F1分数：{fold_score:.4f}")
            
            logger.info(f"交叉验证平均F1分数：{np.mean(cv_scores):.4f}，标准差：{np.std(cv_scores):.4f}")
            
            # 时间序列交叉验证（适合金融数据）
            logger.info("使用时间序列交叉验证进行模型评估")
            try:
                from sklearn.model_selection import TimeSeriesSplit
                
                tscv = TimeSeriesSplit(n_splits=3)
                ts_cv_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
                    logger.info(f"开始第{fold+1}/3次时间序列交叉验证")
                    X_ts_train, X_ts_val = features.iloc[train_idx], features.iloc[val_idx]
                    y_ts_train, y_ts_val = labels.iloc[train_idx], labels.iloc[val_idx]
                    
                    # 处理不平衡数据
                    X_ts_train_resampled, y_ts_train_resampled = self._handle_imbalanced_data(X_ts_train, y_ts_train)
                    
                    # 训练模型
                    temp_model = self._init_model()
                    temp_model.fit(X_ts_train_resampled, y_ts_train_resampled)
                    
                    # 评估模型
                    y_pred = temp_model.predict(X_ts_val)
                    fold_score = f1_score(y_ts_val, y_pred)
                    ts_cv_scores.append(fold_score)
                    logger.info(f"第{fold+1}/3次时间序列交叉验证F1分数：{fold_score:.4f}")
                
                logger.info(f"时间序列交叉验证平均F1分数：{np.mean(ts_cv_scores):.4f}，标准差：{np.std(ts_cv_scores):.4f}")
            except Exception as e:
                logger.warning(f"时间序列交叉验证失败：{e}")
            
            # 处理不平衡数据
            X_train, y_train = self._handle_imbalanced_data(X_train, y_train)
            
            # 模型超参数优化
            logger.info("进行模型超参数优化")
            optimized_model = self._optimize_hyperparameters(X_train, y_train)
            
            # 训练模型，添加早停机制防止过拟合
            logger.info("训练优化后的模型，添加早停机制防止过拟合")
            
            # 计算类别权重
            class_weight = self._calculate_class_weight(y_train)
            logger.info(f"使用类别权重：{class_weight}")
            
            # 根据模型类型使用不同的训练方式
            if config.MODEL_TYPE == "xgboost":
                # XGBoost支持早停机制
                logger.info("使用XGBoost早停机制")
                # 使用训练集和验证集进行早停
                eval_set = [(X_train, y_train), (X_valid, y_valid)]
                # 尝试使用早停机制（根据XGBoost版本调整参数）
                try:
                    optimized_model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        early_stopping_rounds=50,
                        verbose=False
                    )
                    logger.info("XGBoost模型训练完成，使用早停机制")
                except TypeError:
                    # 如果当前版本不支持早停参数，使用基本训练
                    logger.warning("当前XGBoost版本不支持早停参数，使用基本训练")
                    optimized_model.fit(
                        X_train, y_train,
                        verbose=False
                    )
                    logger.info("XGBoost模型训练完成")
            elif config.MODEL_TYPE == "random_forest":
                # 随机森林使用袋外分数监控
                logger.info("使用随机森林袋外分数监控")
                optimized_model.fit(X_train, y_train)
                if hasattr(optimized_model, 'oob_score_'):
                    logger.info(f"随机森林袋外分数：{optimized_model.oob_score_:.4f}")
                else:
                    logger.warning("随机森林模型不支持袋外分数")
            else:
                # 其他模型直接训练
                optimized_model.fit(X_train, y_train)
            
            # 模型评估
            logger.info("评估模型性能")
            metrics = self._evaluate_model(optimized_model, X_valid, y_valid)
            
            # 保存基础模型
            self._save_model(optimized_model, metrics)
            
            # 尝试创建集成模型
            try:
                logger.info("尝试创建集成模型")
                ensemble_model = self._create_ensemble_model(features, labels)
                
                # 评估集成模型性能
                ensemble_metrics = self._evaluate_model(ensemble_model, X_valid, y_valid)
                
                # 如果集成模型性能更好，使用集成模型
                if ensemble_metrics['f1_score'] > metrics['f1_score']:
                    logger.info("集成模型性能优于基础模型，使用集成模型")
                    self.model = ensemble_model
                    # 保存集成模型
                    self._save_model(ensemble_model, ensemble_metrics)
                    final_model = ensemble_model
                    final_metrics = ensemble_metrics
                else:
                    logger.info("基础模型性能优于集成模型，使用基础模型")
                    self.model = optimized_model
                    final_model = optimized_model
                    final_metrics = metrics
            except Exception as e:
                logger.error(f"创建集成模型失败：{e}")
                # 如果集成模型创建失败，使用基础模型
                self.model = optimized_model
                final_model = optimized_model
                final_metrics = metrics
            
            logger.info(f"模型训练完成，最终模型性能指标：{final_metrics}")
            return final_model
            
        except Exception as e:
            logger.error(f"模型训练失败：{e}")
            return None
    
    def _handle_imbalanced_data(self, X, y):
        """处理不平衡数据"""
        logger.info("处理不平衡数据")
        
        try:
            # 使用SMOTE过采样处理不平衡数据
            from imblearn.over_sampling import SMOTE
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # 记录采样前后的标签分布
            before_counts = np.bincount(y)
            after_counts = np.bincount(y_resampled)
            logger.info(f"过采样前标签分布：{dict(zip(np.unique(y), before_counts))}")
            logger.info(f"过采样后标签分布：{dict(zip(np.unique(y_resampled), after_counts))}")
            
            return X_resampled, y_resampled
        except ImportError:
            logger.warning("imblearn库未安装，跳过过采样处理")
            return X, y
        except Exception as e:
            logger.error(f"过采样处理失败：{e}")
            return X, y
    
    def _calculate_class_weight(self, y):
        """计算类别权重"""
        logger.info("计算类别权重")
        
        # 计算每个类别的样本数量
        label_counts = np.bincount(y)
        total_samples = len(y)
        
        # 计算类别权重（反比于样本数量）
        class_weight = {}
        for i, count in enumerate(label_counts):
            if count > 0:
                class_weight[i] = total_samples / (len(label_counts) * count)
            else:
                class_weight[i] = 1.0
        
        return class_weight
    
    def _optimize_hyperparameters(self, X_train, y_train):
        """模型超参数优化"""
        logger.info("开始模型超参数优化")
        
        # 尝试导入贝叶斯优化库
        try:
            from skopt import BayesSearchCV
            from skopt.space import Real, Integer, Categorical
            bayesian_available = True
            logger.info("贝叶斯优化库导入成功，将使用贝叶斯优化进行超参数搜索")
        except ImportError:
            logger.warning("贝叶斯优化库未安装，将使用网格搜索进行超参数搜索")
            bayesian_available = False
        
        # 根据模型类型选择不同的超参数优化方法
        if config.MODEL_TYPE == "logistic":
            # 逻辑回归模型的超参数空间
            if bayesian_available:
                # 贝叶斯优化的超参数空间
                param_space = {
                    'C': Real(1e-4, 1e2, prior='log-uniform'),
                    'l1_ratio': Real(0.0, 1.0),
                    'penalty': Categorical(['elasticnet']),
                    'solver': Categorical(['saga']),
                    'max_iter': Integer(500, 2000)
                }
                
                # 使用贝叶斯优化进行超参数优化
                optimizer = BayesSearchCV(
                    estimator=self.model,
                    search_spaces=param_space,
                    scoring='f1',  # 使用F1分数作为评估指标
                    cv=3,  # 3折交叉验证
                    n_iter=30,  # 搜索30个参数组合
                    random_state=42,
                    verbose=0,
                    n_jobs=-1
                )
            else:
                # 网格搜索的超参数网格
                param_grid = {
                    'C': [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
                    'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
                    'penalty': ['elasticnet'],
                    'solver': ['saga'],
                    'max_iter': [1000, 1500, 2000]
                }
                
                # 使用网格搜索进行超参数优化
                from sklearn.model_selection import GridSearchCV
                optimizer = GridSearchCV(
                    estimator=self.model,
                    param_grid=param_grid,
                    scoring='f1',  # 使用F1分数作为评估指标
                    cv=3,  # 3折交叉验证
                    verbose=0,
                    n_jobs=-1
                )
            
            # 拟合数据
            optimizer.fit(X_train, y_train)
            
            logger.info(f"逻辑回归最佳超参数：{optimizer.best_params_}")
            logger.info(f"逻辑回归最佳F1分数：{optimizer.best_score_:.4f}")
            
            return optimizer.best_estimator()
        
        elif config.MODEL_TYPE == "random_forest":
            # 随机森林模型的超参数空间
            if bayesian_available:
                # 贝叶斯优化的超参数空间
                param_space = {
                    'n_estimators': Integer(100, 500),
                    'max_depth': Integer(3, 20),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 10),
                    'max_features': Categorical(['sqrt', 'log2', None]),
                    'bootstrap': Categorical([True, False]),
                    'class_weight': Categorical(['balanced', 'balanced_subsample', None])
                }
                
                # 使用贝叶斯优化进行超参数优化
                optimizer = BayesSearchCV(
                    estimator=self.model,
                    search_spaces=param_space,
                    scoring='f1',  # 使用F1分数作为评估指标
                    cv=3,  # 3折交叉验证
                    n_iter=30,  # 搜索30个参数组合
                    random_state=42,
                    verbose=0,
                    n_jobs=-1
                )
            else:
                # 网格搜索的超参数网格
                param_grid = {
                    'n_estimators': [100, 200, 300, 400],
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [5, 10, 15],
                    'min_samples_leaf': [2, 5, 8],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False],
                    'class_weight': ['balanced', 'balanced_subsample', None]
                }
                
                # 使用网格搜索进行超参数优化
                from sklearn.model_selection import GridSearchCV
                optimizer = GridSearchCV(
                    estimator=self.model,
                    param_grid=param_grid,
                    scoring='f1',  # 使用F1分数作为评估指标
                    cv=3,  # 3折交叉验证
                    verbose=0,
                    n_jobs=-1
                )
            
            # 拟合数据
            optimizer.fit(X_train, y_train)
            
            logger.info(f"随机森林最佳超参数：{optimizer.best_params_}")
            logger.info(f"随机森林最佳F1分数：{optimizer.best_score_:.4f}")
            
            return optimizer.best_estimator()
        
        elif config.MODEL_TYPE == "xgboost":
            # XGBoost模型的超参数空间
            if bayesian_available:
                # 贝叶斯优化的超参数空间
                param_space = {
                    'n_estimators': Integer(100, 500),
                    'max_depth': Integer(3, 12),
                    'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
                    'subsample': Real(0.6, 1.0),
                    'colsample_bytree': Real(0.6, 1.0),
                    'reg_alpha': Real(1e-3, 1e2, prior='log-uniform'),
                    'reg_lambda': Real(1e-3, 1e2, prior='log-uniform'),
                    'min_child_weight': Integer(1, 10),
                    'gamma': Real(0.0, 1.0)
                }
                
                # 使用贝叶斯优化进行超参数优化
                optimizer = BayesSearchCV(
                    estimator=self.model,
                    search_spaces=param_space,
                    scoring='f1',  # 使用F1分数作为评估指标
                    cv=3,  # 3折交叉验证
                    n_iter=30,  # 搜索30个参数组合
                    random_state=42,
                    verbose=0,
                    n_jobs=-1
                )
            else:
                # 扩展的网格搜索超参数网格
                param_grid = {
                    'n_estimators': [100, 200, 300, 400],
                    'max_depth': [3, 6, 9, 12],
                    'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0.01, 0.1, 1.0, 10.0],
                    'reg_lambda': [0.01, 0.1, 1.0, 10.0],
                    'min_child_weight': [1, 3, 5, 7],
                    'gamma': [0.0, 0.25, 0.5, 0.75]
                }
                
                # 使用网格搜索进行超参数优化
                from sklearn.model_selection import GridSearchCV
                optimizer = GridSearchCV(
                    estimator=self.model,
                    param_grid=param_grid,
                    scoring='f1',  # 使用F1分数作为评估指标
                    cv=3,  # 3折交叉验证
                    verbose=0,
                    n_jobs=-1
                )
            
            # 拟合数据
            optimizer.fit(X_train, y_train)
            
            logger.info(f"XGBoost最佳超参数：{optimizer.best_params_}")
            logger.info(f"XGBoost最佳F1分数：{optimizer.best_score_:.4f}")
            
            return optimizer.best_estimator()
        
        else:
            # 未知模型类型，返回原始模型
            logger.warning(f"未知的模型类型：{config.MODEL_TYPE}，跳过超参数优化")
            return self.model
    
    def _evaluate_model(self, model, X_valid, y_valid):
        """评估模型性能"""
        logger.info("评估模型性能")
        
        # 模型预测
        y_pred = model.predict(X_valid)
        y_pred_proba = model.predict_proba(X_valid)[:, 1]
        
        # 计算基本评估指标
        metrics = {
            'accuracy': accuracy_score(y_valid, y_pred),
            'precision': precision_score(y_valid, y_pred),
            'recall': recall_score(y_valid, y_pred),
            'f1_score': f1_score(y_valid, y_pred),
            'roc_auc': roc_auc_score(y_valid, y_pred_proba)
        }
        
        # 计算Precision@K、Recall@K等指标
        logger.info("计算Precision@K和Recall@K等指标")
        
        # 按预测概率排序
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        sorted_y = y_valid.iloc[sorted_indices] if isinstance(y_valid, pd.Series) else y_valid[sorted_indices]
        
        # 计算不同K值的Precision@K和Recall@K
        for k in [5, 10, 20, 50]:
            if len(sorted_y) >= k:
                top_k = sorted_y[:k]
                precision_at_k = np.sum(top_k) / k
                recall_at_k = np.sum(top_k) / np.sum(y_valid)
                metrics[f'precision@{k}'] = precision_at_k
                metrics[f'recall@{k}'] = recall_at_k
                metrics[f'f1@{k}'] = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0
                logger.info(f"Precision@{k}: {precision_at_k:.4f}, Recall@{k}: {recall_at_k:.4f}, F1@{k}: {metrics[f'f1@{k}']:.4f}")
        
        # 计算AUC-PR（精确率-召回率曲线下面积）
        from sklearn.metrics import average_precision_score
        metrics['auc_pr'] = average_precision_score(y_valid, y_pred_proba)
        logger.info(f"AUC-PR: {metrics['auc_pr']:.4f}")
        
        logger.info(f"模型性能指标：{metrics}")
        
        # 检查模型准确率是否低于阈值
        if metrics['accuracy'] < config.MODEL_ACCURACY_THRESHOLD:
            logger.warning(f"模型准确率({metrics['accuracy']})低于阈值({config.MODEL_ACCURACY_THRESHOLD})，建议更新模型")
        
        return metrics
    
    def _save_model(self, model, metrics):
        """保存模型"""
        logger.info(f"保存模型到本地，路径：{config.MODEL_PATH}")
        
        # 创建数据处理器和特征工程器实例
        data_processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        
        # 使用ModelWrapper封装模型、数据处理器和特征工程器
        model_wrapper = ModelWrapper(
            model=model,
            data_processor=data_processor,
            feature_engineer=feature_engineer,
            config={
                'model_type': config.MODEL_TYPE,
                'metrics': metrics,
                'train_time': pd.Timestamp.now().isoformat(),
                'version': '1.0'
            }
        )
        
        # 模型文件名
        model_file_name = f"model_wrapper_{config.MODEL_TYPE}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model_file_path = os.path.join(config.MODEL_PATH, model_file_name)
        
        # 保存模型
        joblib.dump(model_wrapper, model_file_path)
        logger.info(f"模型保存成功：{model_file_path}")
        
        # 保存模型性能指标
        metrics_file_name = f"metrics_{config.MODEL_TYPE}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        metrics_file_path = os.path.join(config.MODEL_PATH, metrics_file_name)
        
        import json
        with open(metrics_file_path, 'w') as f:
            json.dump(metrics, f)
        logger.info(f"模型性能指标保存成功：{metrics_file_path}")
    
    def _create_ensemble_model(self, features, labels):
        """创建集成模型，提高预测准确性"""
        logger.info("创建集成模型")
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.ensemble import StackingClassifier
        from xgboost import XGBClassifier
        
        try:
            # 定义基础模型，使用已优化的参数
            base_models = [
                ('lr', LogisticRegression(
                    penalty='elasticnet', solver='saga', l1_ratio=0.5, 
                    C=1.0, max_iter=1500, random_state=42
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=300, max_depth=15, min_samples_split=10,
                    min_samples_leaf=5, max_features='sqrt', bootstrap=True,
                    oob_score=True, class_weight='balanced', 
                    random_state=42, n_jobs=-1
                )),
                ('xgb', XGBClassifier(
                    n_estimators=300, max_depth=8, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                    gamma=0.1, reg_alpha=1.0, reg_lambda=2.0,
                    scale_pos_weight=10, objective='binary:logistic',
                    eval_metric='auc', random_state=42, n_jobs=-1
                ))
            ]
            
            # 尝试使用Stacking集成方法
            logger.info("创建Stacking集成模型")
            stacking_model = StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(
                    penalty='elasticnet', solver='saga', l1_ratio=0.5,
                    C=1.0, max_iter=1500, random_state=42
                ),
                cv=5,  # 5折交叉验证
                stack_method='predict_proba',  # 使用概率作为元特征
                n_jobs=-1,
                passthrough=False  # 不将原始特征传递给最终估计器
            )
            
            # 训练Stacking模型
            stacking_model.fit(features, labels)
            logger.info("Stacking集成模型创建成功")
            return stacking_model
        except Exception as e:
            logger.warning(f"Stacking集成模型创建失败：{e}，将使用投票法集成模型")
            
            # 投票法集成作为备选
            logger.info("创建投票法集成模型")
            voting_model = VotingClassifier(
                estimators=base_models,
                voting='soft',  # 软投票，使用概率
                n_jobs=-1
            )
            
            # 训练投票法模型
            voting_model.fit(features, labels)
            logger.info("投票法集成模型创建成功")
            return voting_model
    
    def load_model(self, model_file_path=None):
        """加载模型"""
        logger.info("加载模型")
        
        if model_file_path is None:
            # 首先尝试加载配置中指定的模型文件
            if hasattr(config, 'MODEL_FILE_NAME') and config.MODEL_FILE_NAME:
                model_file_path = os.path.join(config.MODEL_PATH, config.MODEL_FILE_NAME)
                logger.info(f"使用配置中指定的模型文件：{model_file_path}")
            else:
                # 如果没有配置具体模型文件名，则加载最新的模型
                model_files = [f for f in os.listdir(config.MODEL_PATH) if f.endswith('.joblib')]
                if not model_files:
                    logger.warning("模型目录中没有找到模型文件")
                    return None
                
                # 按修改时间排序，获取最新的模型文件
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(config.MODEL_PATH, x)), reverse=True)
                model_file_path = os.path.join(config.MODEL_PATH, model_files[0])
                logger.info(f"使用最新模型文件：{model_file_path}")
        
        try:
            # 加载模型
            model = joblib.load(model_file_path)
            logger.info(f"模型加载成功：{model_file_path}")
            
            # 检查模型类型
            if hasattr(model, 'predict_proba'):
                logger.info("模型支持概率预测")
            else:
                logger.warning("模型不支持概率预测，这可能导致预测概率全为0")
            
            # 更新模型
            self.model = model
            
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败：{e}")
            return None
    
    def predict(self, features):
        """使用模型进行预测"""
        logger.info("使用模型进行预测")
        
        if self.model is None:
            logger.warning("模型未初始化，尝试加载模型")
            self.load_model()
            
            if self.model is None:
                logger.error("无法加载模型，预测失败")
                return None
        
        if features is None or features.empty:
            logger.warning("特征数据为空，跳过预测")
            return None
        
        try:
            # 转换特征为numpy数组，忽略特征名称
            features_np = features.values
            
            # 进行预测
            predictions = self.model.predict(features_np)
            
            # 获取预测概率，处理单类别情况
            prob_array = self.model.predict_proba(features_np)
            if prob_array.shape[1] > 1:
                # 多类别情况，获取正类（1）的概率
                probabilities = prob_array[:, 1]
            else:
                # 单类别情况，根据预测结果设置概率
                # 如果预测是1，则概率为1.0，否则为0.0
                probabilities = np.where(predictions == 1, 1.0, 0.0)
            
            logger.info("模型预测完成")
            logger.info(f"预测结果统计：正例数={sum(predictions)}, 负例数={len(predictions)-sum(predictions)}")
            logger.info(f"预测概率统计：平均值={probabilities.mean()}, 最大值={probabilities.max()}, 最小值={probabilities.min()}")
            return {
                'predictions': predictions,
                'probabilities': probabilities
            }
            
        except Exception as e:
            logger.error(f"模型预测失败：{e}")
            import traceback
            logger.error(f"异常堆栈：{traceback.format_exc()}")
            return None
    
    def update_model(self, features, labels=None):
        """更新模型"""
        logger.info("更新模型")
        
        # 重新训练模型
        return self.train(features, labels)
