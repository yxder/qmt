#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型融合模块
负责实现多种模型融合策略，包括加权投票、Stacking等
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from utils.logger import setup_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = setup_logger()

class ModelFusion:
    """模型融合类"""
    
    def __init__(self, models=None, fusion_type='weighted_voting'):
        """
        初始化模型融合器
        
        Args:
            models: 基础模型列表
            fusion_type: 融合类型，可选值：weighted_voting, stacking, bagging
        """
        logger.info(f"初始化模型融合器，融合类型：{fusion_type}")
        
        # 默认基础模型配置
        self.base_models = models or self._get_default_base_models()
        self.fusion_type = fusion_type
        self.fusion_model = None
        self.model_weights = None
        self.model_performances = {}
        self.is_trained = False
    
    def _get_default_base_models(self):
        """
        获取默认基础模型
        
        Returns:
            默认基础模型列表
        """
        return [
            ('lr', LogisticRegression(
                penalty='elasticnet', solver='saga', l1_ratio=0.5,
                C=1.0, max_iter=2000, random_state=42, n_jobs=-1
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, max_features='sqrt', bootstrap=True,
                oob_score=True, class_weight='balanced',
                random_state=42, n_jobs=-1
            )),
            ('xgb', XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                gamma=0.1, reg_alpha=1.0, reg_lambda=2.0,
                scale_pos_weight=10, objective='binary:logistic',
                eval_metric='auc', random_state=42, n_jobs=-1
            ))
        ]
    
    def train(self, X, y):
        """
        训练融合模型
        
        Args:
            X: 训练特征
            y: 训练标签
        """
        logger.info(f"训练{self.fusion_type}融合模型")
        
        try:
            if self.fusion_type == 'weighted_voting':
                # 训练加权投票模型
                self._train_weighted_voting(X, y)
            elif self.fusion_type == 'stacking':
                # 训练Stacking模型
                self._train_stacking(X, y)
            elif self.fusion_type == 'bagging':
                # 训练Bagging模型
                self._train_bagging(X, y)
            else:
                logger.error(f"未知的融合类型：{self.fusion_type}")
                return False
            
            self.is_trained = True
            logger.info("模型融合训练完成")
            return True
            
        except Exception as e:
            logger.error(f"模型融合训练失败：{e}")
            return False
    
    def _train_weighted_voting(self, X, y):
        """
        训练加权投票模型
        
        Args:
            X: 训练特征
            y: 训练标签
        """
        logger.info("训练加权投票模型")
        
        # 先训练所有基础模型
        trained_models = []
        for name, model in self.base_models:
            logger.info(f"训练基础模型：{name}")
            model.fit(X, y)
            trained_models.append((name, model))
        
        # 计算每个模型的权重
        self._calculate_model_weights(X, y, trained_models)
        
        # 创建加权投票分类器
        self.fusion_model = VotingClassifier(
            estimators=trained_models,
            voting='soft',  # 使用软投票
            weights=self.model_weights,
            n_jobs=-1
        )
        
        # 训练加权投票模型
        logger.info("训练加权投票分类器")
        self.fusion_model.fit(X, y)
    
    def _train_stacking(self, X, y):
        """
        训练Stacking模型
        
        Args:
            X: 训练特征
            y: 训练标签
        """
        logger.info("训练Stacking模型")
        
        # 创建Stacking分类器
        self.fusion_model = StackingClassifier(
            estimators=self.base_models,
            final_estimator=LogisticRegression(
                penalty='elasticnet', solver='saga', l1_ratio=0.5,
                C=1.0, max_iter=2000, random_state=42, n_jobs=-1
            ),
            cv=5,  # 5折交叉验证
            stack_method='predict_proba',  # 使用概率作为元特征
            n_jobs=-1
        )
        
        # 训练Stacking模型
        self.fusion_model.fit(X, y)
    
    def _train_bagging(self, X, y):
        """
        训练Bagging模型
        
        Args:
            X: 训练特征
            y: 训练标签
        """
        logger.info("训练Bagging模型")
        
        # 这里使用随机森林作为Bagging的实现
        # 可以根据需要替换为其他Bagging模型
        self.fusion_model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt', bootstrap=True,
            oob_score=True, class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        
        self.fusion_model.fit(X, y)
    
    def _calculate_model_weights(self, X, y, models=None):
        """
        计算模型权重
        
        Args:
            X: 特征数据
            y: 标签数据
            models: 模型列表，默认为训练好的基础模型
        """
        logger.info("计算模型权重")
        
        if models is None:
            models = self.base_models
        
        # 评估每个模型的性能
        performances = []
        for name, model in models:
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 计算多个评估指标
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1_score': f1_score(y, y_pred)
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
            
            # 使用F1分数作为主要权重依据
            performances.append((name, metrics['f1_score']))
            self.model_performances[name] = metrics
            
            logger.info(f"模型{name}性能：{metrics}")
        
        # 计算权重（基于F1分数的归一化）
        f1_scores = np.array([score for _, score in performances])
        self.model_weights = list(f1_scores / np.sum(f1_scores)) if np.sum(f1_scores) > 0 else [1/len(models)] * len(models)
        
        logger.info(f"模型权重：{dict(zip([name for name, _ in performances], self.model_weights))}")
    
    def predict(self, X):
        """
        使用融合模型进行预测
        
        Args:
            X: 待预测的特征数据
            
        Returns:
            预测结果和预测概率
        """
        logger.info("使用融合模型进行预测")
        
        if not self.is_trained:
            logger.error("融合模型未训练，无法预测")
            return None, None
        
        try:
            # 预测类别
            y_pred = self.fusion_model.predict(X)
            
            # 预测概率
            y_pred_proba = None
            if hasattr(self.fusion_model, 'predict_proba'):
                y_pred_proba = self.fusion_model.predict_proba(X)[:, 1]
            
            logger.info(f"融合模型预测完成，预测结果样本：{y_pred[:5]}")
            return y_pred, y_pred_proba
            
        except Exception as e:
            logger.error(f"融合模型预测失败：{e}")
            return None, None
    
    def predict_with_base_models(self, X):
        """
        使用所有基础模型进行预测
        
        Args:
            X: 待预测的特征数据
            
        Returns:
            各基础模型的预测结果和融合结果
        """
        logger.info("使用所有基础模型进行预测")
        
        if not self.is_trained:
            logger.error("融合模型未训练，无法预测")
            return None
        
        try:
            # 获取所有基础模型的预测结果
            base_predictions = {}
            base_probabilities = {}
            
            # 根据融合类型获取基础模型
            models_to_use = self.base_models
            if hasattr(self.fusion_model, 'estimators_'):
                models_to_use = list(zip([name for name, _ in self.base_models], self.fusion_model.estimators_))
            
            for name, model in models_to_use:
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
                
                base_predictions[name] = y_pred
                base_probabilities[name] = y_pred_proba
            
            # 获取融合模型的预测结果
            fusion_pred, fusion_proba = self.predict(X)
            
            return {
                'base_predictions': base_predictions,
                'base_probabilities': base_probabilities,
                'fusion_prediction': fusion_pred,
                'fusion_probability': fusion_proba
            }
            
        except Exception as e:
            logger.error(f"基础模型预测失败：{e}")
            return None
    
    def evaluate(self, X, y):
        """
        评估融合模型性能
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            评估结果字典
        """
        logger.info("评估融合模型性能")
        
        if not self.is_trained:
            logger.error("融合模型未训练，无法评估")
            return None
        
        try:
            y_pred, y_pred_proba = self.predict(X)
            
            # 计算评估指标
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1_score': f1_score(y, y_pred)
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
            
            logger.info(f"融合模型评估结果：{metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"融合模型评估失败：{e}")
            return None
    
    def evaluate_base_models(self, X, y):
        """
        评估所有基础模型性能
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            各基础模型的评估结果
        """
        logger.info("评估所有基础模型性能")
        
        results = {}
        
        # 根据融合类型获取基础模型
        models_to_use = self.base_models
        if hasattr(self.fusion_model, 'estimators_'):
            models_to_use = list(zip([name for name, _ in self.base_models], self.fusion_model.estimators_))
        
        for name, model in models_to_use:
            logger.info(f"评估基础模型：{name}")
            
            try:
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # 计算评估指标
                metrics = {
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred),
                    'recall': recall_score(y, y_pred),
                    'f1_score': f1_score(y, y_pred)
                }
                
                if y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
                
                results[name] = metrics
                logger.info(f"模型{name}评估结果：{metrics}")
                
            except Exception as e:
                logger.error(f"模型{name}评估失败：{e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def update_model_weights(self, X, y):
        """
        更新模型权重
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            更新后的权重
        """
        logger.info("更新模型权重")
        
        if not self.is_trained:
            logger.error("融合模型未训练，无法更新权重")
            return None
        
        # 根据融合类型获取基础模型
        models_to_use = self.base_models
        if hasattr(self.fusion_model, 'estimators_'):
            models_to_use = list(zip([name for name, _ in self.base_models], self.fusion_model.estimators_))
        
        # 重新计算模型权重
        self._calculate_model_weights(X, y, models_to_use)
        
        # 如果是加权投票模型，更新权重
        if isinstance(self.fusion_model, VotingClassifier):
            self.fusion_model.weights = self.model_weights
            logger.info(f"已更新加权投票模型的权重：{self.model_weights}")
        
        return self.model_weights
    
    def dynamic_weight_adjustment(self, X, y, adjustment_period=5):
        """
        动态调整模型权重
        
        Args:
            X: 特征数据
            y: 标签数据
            adjustment_period: 调整周期
            
        Returns:
            是否成功调整权重
        """
        logger.info("动态调整模型权重")
        
        try:
            # 评估当前模型性能
            current_performance = self.evaluate(X, y)
            
            # 计算性能变化
            if self.model_performances.get('fusion'):
                performance_change = {
                    key: current_performance[key] - self.model_performances['fusion'][key]
                    for key in current_performance
                }
                logger.info(f"性能变化：{performance_change}")
            
            # 更新模型性能记录
            self.model_performances['fusion'] = current_performance
            
            # 更新基础模型权重
            self.update_model_weights(X, y)
            
            logger.info("动态权重调整完成")
            return True
            
        except Exception as e:
            logger.error(f"动态权重调整失败：{e}")
            return False
    
    def save_model(self, file_path):
        """
        保存融合模型
        
        Args:
            file_path: 保存路径
        """
        logger.info(f"保存融合模型到：{file_path}")
        
        try:
            import joblib
            joblib.dump({
                'fusion_model': self.fusion_model,
                'model_weights': self.model_weights,
                'fusion_type': self.fusion_type,
                'is_trained': self.is_trained,
                'model_performances': self.model_performances
            }, file_path)
            
            logger.info("融合模型保存成功")
            return True
            
        except Exception as e:
            logger.error(f"保存融合模型失败：{e}")
            return False
    
    def load_model(self, file_path):
        """
        加载融合模型
        
        Args:
            file_path: 加载路径
            
        Returns:
            是否成功加载
        """
        logger.info(f"加载融合模型从：{file_path}")
        
        try:
            import joblib
            model_data = joblib.load(file_path)
            
            self.fusion_model = model_data.get('fusion_model')
            self.model_weights = model_data.get('model_weights')
            self.fusion_type = model_data.get('fusion_type', 'weighted_voting')
            self.is_trained = model_data.get('is_trained', False)
            self.model_performances = model_data.get('model_performances', {})
            
            logger.info("融合模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"加载融合模型失败：{e}")
            return False
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'fusion_type': self.fusion_type,
            'base_models': [name for name, _ in self.base_models],
            'model_weights': self.model_weights,
            'is_trained': self.is_trained,
            'model_performances': self.model_performances
        }
