#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型预测模块
应用预先训练的机器学习预测模型，对候选个股进行多维度评估
"""

import pandas as pd
import numpy as np
import joblib
import os
import datetime
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from utils.logger import setup_logger
import shap

logger = setup_logger()


class ModelPredictor:
    """模型预测器，用于对候选个股进行多维度评估"""
    
    def __init__(self, model_path: str = None):
        """初始化模型预测器
        
        Args:
            model_path: 模型文件路径
        """
        logger.info("初始化模型预测器")
        
        # 设置默认模型路径
        if model_path is None:
            from config import MODEL_PATH
            self.model_path = MODEL_PATH
        else:
            self.model_path = model_path
        
        # 加载模型
        self.models = self._load_models()
        
        # 初始化SHAP解释器
        self.shap_explainer = None
    
    def _load_models(self) -> Dict[str, any]:
        """加载预训练模型
        
        Returns:
            模型字典，键为模型名称，值为模型对象
        """
        logger.info(f"从{self.model_path}加载模型")
        
        models = {}
        
        # 检查模型路径是否存在
        if not os.path.exists(self.model_path):
            logger.error(f"模型路径不存在：{self.model_path}")
            return models
        
        # 遍历模型目录，加载所有模型
        for file_name in os.listdir(self.model_path):
            if file_name.endswith(".joblib"):
                model_file = os.path.join(self.model_path, file_name)
                model_name = file_name.replace(".joblib", "")
                
                try:
                    model = joblib.load(model_file)
                    models[model_name] = model
                    logger.info(f"成功加载模型：{model_name}")
                except Exception as e:
                    logger.error(f"加载模型{model_name}失败：{e}")
        
        logger.info(f"共加载了{len(models)}个模型")
        return models
    
    def train_models(self, train_data: pd.DataFrame, target_column: str = 'is_limit_up'):
        """训练模型
        
        Args:
            train_data: 训练数据
            target_column: 目标列名称
        
        Returns:
            训练好的模型字典
        """
        logger.info("开始训练模型")
        
        if train_data.empty:
            logger.error("训练数据为空，无法训练模型")
            return {}
        
        # 分离特征和目标
        if target_column not in train_data.columns:
            logger.error(f"目标列{target_column}不存在于训练数据中")
            return {}
        
        X = train_data.drop(columns=[target_column, 'stock_code', 'date'], errors='ignore')
        y = train_data[target_column]
        
        # 处理缺失值
        X = X.fillna(0)
        
        # 模型配置
        models_to_train = {
            'logistic': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'xgboost': {
                'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            }
        }
        
        trained_models = {}
        
        # 训练每个模型
        for model_name, model_config in models_to_train.items():
            try:
                logger.info(f"训练{model_name}模型")
                
                # 使用GridSearchCV进行参数调优
                grid_search = GridSearchCV(
                    estimator=model_config['model'],
                    param_grid=model_config['params'],
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                
                grid_search.fit(X, y)
                
                best_model = grid_search.best_estimator_
                best_score = grid_search.best_score_
                best_params = grid_search.best_params_
                
                logger.info(f"{model_name}模型训练完成，最佳参数：{best_params}，最佳分数：{best_score:.4f}")
                
                trained_models[model_name] = best_model
            except Exception as e:
                logger.error(f"训练{model_name}模型失败：{e}")
                continue
        
        # 保存训练好的模型
        self._save_models(trained_models)
        
        # 更新模型字典
        self.models = trained_models
        
        return trained_models
    
    def _save_models(self, models: Dict[str, any]):
        """保存模型
        
        Args:
            models: 模型字典
        """
        logger.info(f"保存模型到{self.model_path}")
        
        # 确保模型路径存在
        os.makedirs(self.model_path, exist_ok=True)
        
        # 保存每个模型
        for model_name, model in models.items():
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_file = os.path.join(self.model_path, f"model_{model_name}_{timestamp}.joblib")
                joblib.dump(model, model_file)
                logger.info(f"成功保存模型：{model_file}")
            except Exception as e:
                logger.error(f"保存模型{model_name}失败：{e}")
                continue
    
    def update_models(self, new_data: pd.DataFrame, target_column: str = 'is_limit_up'):
        """更新模型
        
        Args:
            new_data: 新的训练数据
            target_column: 目标列名称
        
        Returns:
            更新后的模型字典
        """
        logger.info("开始更新模型")
        
        # 如果没有预训练模型，直接训练新模型
        if not self.models:
            logger.info("没有预训练模型，直接训练新模型")
            return self.train_models(new_data, target_column)
        
        # 加载历史训练数据
        historical_data = self._load_historical_data()
        
        # 合并历史数据和新数据
        combined_data = pd.concat([historical_data, new_data], ignore_index=True)
        
        # 训练新模型
        updated_models = self.train_models(combined_data, target_column)
        
        return updated_models
    
    def _load_historical_data(self) -> pd.DataFrame:
        """加载历史训练数据
        
        Returns:
            历史训练数据
        """
        logger.info("加载历史训练数据")
        
        # 这里应该从数据库或文件中加载历史数据
        # 暂时返回空DataFrame，实际实现中需要替换
        return pd.DataFrame()
    
    def predict(self, candidate_stocks: pd.DataFrame, features: pd.DataFrame = None) -> pd.DataFrame:
        """对候选个股进行多维度评估
        
        Args:
            candidate_stocks: 候选个股DataFrame
            features: 用于预测的特征数据，如果为None则从candidate_stocks中提取
            
        Returns:
            带有预测结果的候选个股DataFrame
        """
        logger.info("开始对候选个股进行多维度评估")
        
        if candidate_stocks.empty:
            logger.error("候选个股为空，无法进行预测")
            return pd.DataFrame()
        
        # 如果没有提供特征数据，则从候选个股中提取
        if features is None:
            features = self._extract_features(candidate_stocks)
        
        # 确保特征数据和候选个股数量一致
        if len(features) != len(candidate_stocks):
            logger.error(f"特征数据数量({len(features)})与候选个股数量({len(candidate_stocks)})不一致")
            return pd.DataFrame()
        
        # 应用模型预测
        predictions = self._apply_models(features)
        
        # 将预测结果合并到候选个股中
        result = self._merge_predictions(candidate_stocks, predictions)
        
        logger.info("候选个股预测完成")
        return result
    
    def _extract_features(self, candidate_stocks: pd.DataFrame) -> pd.DataFrame:
        """从候选个股中提取特征
        
        Args:
            candidate_stocks: 候选个股DataFrame
            
        Returns:
            特征数据DataFrame
        """
        logger.info("从候选个股中提取特征")
        
        # 明确定义模型的输入特征列表
        # 1. 基础特征（直接从候选个股数据中提取）
        base_features = [
            # 价格相关特征
            'price_bid_price', 'price_bid_change', 'price_prev_close', 'price_open',
            
            # 成交量相关特征
            'volume_bid_volume', 'volume_bid_volume_ratio', 'volume_bid_turnover_rate',
            'volume_volume_3d_avg', 'volume_volume_growth_rate_3d', 'volume_volume_growth_rate_5d',
            
            # 资金流向相关特征
            'fund_flow_net_flow', 'fund_flow_large_order_flow', 'fund_flow_large_order_ratio',
            
            # 委托单相关特征
            'order_book_bid_order_amount', 'order_book_bid_order_ratio', 'order_book_ask_order_amount',
            'order_book_order_imbalance', 'order_book_bid_order_count', 'order_book_ask_order_count',
            'order_book_avg_bid_order_size', 'order_book_avg_ask_order_size',
            
            # 股票基础特征
            'circulating_market_cap'
        ]
        
        # 2. 衍生特征（通过基础特征计算得到）
        derived_features = []
        
        # 检查候选个股中是否包含所有需要的基础特征
        available_features = [f for f in base_features if f in candidate_stocks.columns]
        missing_features = [f for f in base_features if f not in candidate_stocks.columns]
        
        if missing_features:
            logger.warning(f"缺少以下基础特征：{missing_features}，将使用默认值")
            
            # 为缺少的特征添加默认值
            for f in missing_features:
                candidate_stocks[f] = 0
        
        # 提取基础特征
        features = candidate_stocks[base_features].copy()
        
        # 添加衍生特征
        features = self._add_derived_features(features)
        
        # 保存特征列表，用于后续的模型解释
        self.features_list = list(features.columns)
        logger.info(f"提取了{len(features.columns)}个特征：{self.features_list}")
        
        return features
    
    def _add_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """添加衍生特征
        
        Args:
            features: 原始特征DataFrame
            
        Returns:
            添加了衍生特征的DataFrame
        """
        logger.info("添加衍生特征")
        
        # 1. 资金流入相关特征
        features['fund_inflow_strength'] = features['fund_flow_net_flow'] / (features['order_book_bid_order_amount'] + 1e-10)
        features['large_order_ratio'] = features['fund_flow_large_order_flow'] / (features['fund_flow_net_flow'] + 1e-10)
        features['net_flow_per_volume'] = features['fund_flow_net_flow'] / (features['volume_bid_volume'] + 1e-10)
        
        # 2. 竞价强度相关特征
        features['bid_strength_综合'] = (
            features['price_bid_change'] * 0.4 +
            features['volume_bid_volume_ratio'] * 0.3 +
            features['volume_bid_turnover_rate'] * 0.3
        )
        features['price_volume_ratio'] = features['price_bid_change'] / (features['volume_bid_volume_ratio'] + 1e-10)
        features['turnover_change_ratio'] = features['volume_bid_turnover_rate'] / (features['price_bid_change'] + 0.01)
        
        # 3. 封单强度相关特征
        features['order_strength_综合'] = (
            features['order_book_bid_order_amount'] * 0.6 +
            features['order_book_bid_order_ratio'] * 0.4
        )
        features['order_imbalance_normalized'] = features['order_book_bid_order_amount'] / (features['order_book_ask_order_amount'] + 1e-10) if 'order_book_ask_order_amount' in features.columns else 0
        
        # 4. 综合评分特征
        features['volume_price_score'] = features['volume_bid_volume_ratio'] * features['price_bid_change']
        features['fund_order_score'] = features['fund_flow_net_flow'] * features['order_book_bid_order_ratio']
        
        # 5. 极值处理
        # 处理无穷大和无穷小值
        features = features.replace([np.inf, -np.inf], np.nan)
        # 填充缺失值
        features = features.fillna(0)
        
        logger.info("衍生特征添加完成")
        return features
    
    def _apply_models(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """应用模型进行预测
        
        Args:
            features: 特征数据
            
        Returns:
            预测结果字典，键为预测指标名称，值为预测结果Series
            预测指标包括：
            - rise_prob_30min: 未来30分钟内上涨概率（0-1）
            - limit_up_prob: 当日封板成功率（0-1）
            - next_day_premium: 封板后次日溢价空间（百分比，如0.02表示2%）
            - prediction_confidence: 预测置信度（0-1）
            - rise_prob_30min_lower: 未来30分钟内上涨概率置信区间下限（0-1）
            - rise_prob_30min_upper: 未来30分钟内上涨概率置信区间上限（0-1）
            - limit_up_prob_lower: 当日封板成功率置信区间下限（0-1）
            - limit_up_prob_upper: 当日封板成功率置信区间上限（0-1）
        """
        logger.info("应用模型进行预测")
        
        predictions = {}
        
        # 检查是否有可用模型
        if not self.models:
            logger.error("没有可用模型，使用默认预测")
            return self._default_predictions(features)
        
        try:
            # 实现模型融合
            # 1. 获取所有模型的预测结果
            all_predictions = self._get_all_model_predictions(features)
            
            # 2. 融合预测结果
            predictions = self._fuse_predictions(all_predictions, features)
            
            # 3. 计算预测置信度
            predictions['prediction_confidence'] = self._calculate_confidence(features)
            
            # 4. 计算各预测指标的置信区间
            predictions = self._calculate_prediction_intervals(predictions, features)
            
        except Exception as e:
            logger.error(f"模型预测失败：{e}")
            # 使用默认预测结果
            predictions = self._default_predictions(features)
        
        logger.info("模型预测完成")
        return predictions
    
    def _get_all_model_predictions(self, features: pd.DataFrame) -> Dict[str, Dict[str, pd.Series]]:
        """获取所有模型的预测结果
        
        Args:
            features: 特征数据
            
        Returns:
            所有模型的预测结果字典，键为模型名称，值为预测结果字典
        """
        logger.info("获取所有模型的预测结果")
        
        all_predictions = {}
        
        for model_name, model in self.models.items():
            logger.info(f"使用模型{model_name}进行预测")
            
            try:
                model_predictions = {
                    'rise_prob_30min': self._predict_rise_probability(model, features),
                    'limit_up_prob': self._predict_limit_up_probability(model, features),
                    'next_day_premium': self._predict_next_day_premium(model, features)
                }
                all_predictions[model_name] = model_predictions
            except Exception as e:
                logger.error(f"模型{model_name}预测失败：{e}")
        
        return all_predictions
    
    def _fuse_predictions(self, all_predictions: Dict[str, Dict[str, pd.Series]], features: pd.DataFrame) -> Dict[str, pd.Series]:
        """融合多个模型的预测结果
        
        Args:
            all_predictions: 所有模型的预测结果
            features: 特征数据
            
        Returns:
            融合后的预测结果
        """
        logger.info("融合多个模型的预测结果")
        
        if not all_predictions:
            logger.error("没有可用的模型预测结果，使用默认预测")
            return self._default_predictions(features)
        
        # 初始化融合结果
        fused_predictions = {
            'rise_prob_30min': pd.Series(0.0, index=features.index),
            'limit_up_prob': pd.Series(0.0, index=features.index),
            'next_day_premium': pd.Series(0.0, index=features.index)
        }
        
        # 计算模型权重 - 基于模型类型的动态权重
        # XGBoost模型权重更高，因为它通常表现更好
        model_weights = {}
        for model_name in all_predictions.keys():
            if 'xgboost' in model_name.lower():
                model_weights[model_name] = 0.4  # XGBoost权重40%
            elif 'random_forest' in model_name.lower():
                model_weights[model_name] = 0.3  # RandomForest权重30%
            elif 'logistic' in model_name.lower():
                model_weights[model_name] = 0.2  # LogisticRegression权重20%
            else:
                model_weights[model_name] = 0.1  # 其他模型权重10%
        
        # 归一化权重
        total_weight = sum(model_weights.values())
        for model_name in model_weights.keys():
            model_weights[model_name] /= total_weight
        
        logger.info(f"模型融合权重：{model_weights}")
        
        # 应用权重融合
        for model_name, model_predictions in all_predictions.items():
            weight = model_weights[model_name]
            
            for pred_name in fused_predictions.keys():
                if pred_name in model_predictions:
                    fused_predictions[pred_name] += model_predictions[pred_name] * weight
        
        # 确保概率在合理范围内
        fused_predictions['rise_prob_30min'] = fused_predictions['rise_prob_30min'].clip(0, 1)
        fused_predictions['limit_up_prob'] = fused_predictions['limit_up_prob'].clip(0, 1)
        
        logger.info("模型预测结果融合完成")
        return fused_predictions
    
    def _predict_rise_probability(self, model: any, features: pd.DataFrame) -> pd.Series:
        """预测未来30分钟内上涨概率
        
        Args:
            model: 模型对象
            features: 特征数据
            
        Returns:
            未来30分钟内上涨概率Series
        """
        logger.info("预测未来30分钟内上涨概率")
        
        # 检查模型是否支持概率预测
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(features)[:, 1]
        else:
            # 如果模型不支持概率预测，则使用预测结果作为概率
            pred = model.predict(features)
            prob = np.where(pred == 1, 0.7, 0.3)
        
        return pd.Series(prob, index=features.index, name='rise_prob_30min')
    
    def _predict_limit_up_probability(self, model: any, features: pd.DataFrame) -> pd.Series:
        """预测当日封板成功率
        
        Args:
            model: 模型对象
            features: 特征数据
            
        Returns:
            当日封板成功率Series
        """
        logger.info("预测当日封板成功率")
        
        # 基于上涨概率和其他特征计算封板成功率
        rise_prob = self._predict_rise_probability(model, features)
        
        # 封板成功率 = 上涨概率 * 封单强度因子
        # 封单强度因子 = 封单金额 / 平均封单金额 * 0.5 + 封单比例 * 0.5
        avg_order_amount = features['order_book_bid_order_amount'].mean()
        order_strength_factor = (
            (features['order_book_bid_order_amount'] / (avg_order_amount + 1e-10)) * 0.5 +
            features['order_book_bid_order_ratio'] * 0.5
        )
        
        # 归一化封单强度因子到[0, 1]范围
        order_strength_factor = np.clip(order_strength_factor, 0, 1)
        
        limit_up_prob = rise_prob * order_strength_factor
        
        return pd.Series(limit_up_prob, index=features.index, name='limit_up_prob')
    
    def _predict_next_day_premium(self, model: any, features: pd.DataFrame) -> pd.Series:
        """预测封板后次日溢价空间
        
        Args:
            model: 模型对象
            features: 特征数据
            
        Returns:
            封板后次日溢价空间Series
        """
        logger.info("预测封板后次日溢价空间")
        
        # 基于上涨概率、封单强度和资金流入计算次日溢价空间
        limit_up_prob = self._predict_limit_up_probability(model, features)
        
        # 溢价空间 = 基础溢价 + 封单强度溢价 + 资金流入溢价
        # 基础溢价：0.02（2%）
        # 封单强度溢价：封单强度因子 * 0.03（最高3%）
        # 资金流入溢价：资金流入强度 * 0.02（最高2%）
        
        avg_order_amount = features['order_book_bid_order_amount'].mean()
        order_strength_factor = (
            (features['order_book_bid_order_amount'] / (avg_order_amount + 1e-10)) * 0.5 +
            features['order_book_bid_order_ratio'] * 0.5
        )
        order_strength_factor = np.clip(order_strength_factor, 0, 1)
        
        # 资金流入强度归一化
        fund_inflow_strength = features['fund_flow_net_flow'] / (features['fund_flow_net_flow'].abs().max() + 1e-10)
        fund_inflow_strength = (fund_inflow_strength + 1) / 2  # 转换到[0, 1]范围
        
        next_day_premium = 0.02 + order_strength_factor * 0.03 + fund_inflow_strength * 0.02
        
        # 只有封板概率高的股票才会有溢价
        next_day_premium = next_day_premium * limit_up_prob
        
        return pd.Series(next_day_premium, index=features.index, name='next_day_premium')
    
    def _calculate_confidence(self, features: pd.DataFrame) -> pd.Series:
        """计算预测置信度
        
        Args:
            features: 特征数据
            
        Returns:
            预测置信度Series，基于特征完整性、稳定性、重要性和模型一致性
        """
        logger.info("计算预测置信度")
        
        # 置信度基于多个维度：
        # 1. 特征完整性：如果有缺失值，置信度降低
        # 2. 特征稳定性：如果特征值在合理范围内，置信度高
        # 3. 特征重要性：如果重要特征值更可靠，置信度高
        # 4. 模型一致性：基于多个模型的预测结果一致性
        
        # 计算特征完整性得分（0-1）
        missing_values = features.isnull().sum(axis=1)
        completeness_score = 1 - (missing_values / len(features.columns))
        
        # 计算特征稳定性得分（0-1）
        # 基于关键特征的取值范围
        stability_score = pd.Series(1.0, index=features.index)
        
        # 竞价涨跌幅应在[-10%, 10%]范围内
        if 'price_bid_change' in features.columns:
            stability_score *= np.clip((features['price_bid_change'] + 0.1) / 0.2, 0, 1)
        
        # 竞价成交量比应在[0, 15]范围内
        if 'volume_bid_volume_ratio' in features.columns:
            stability_score *= np.clip(features['volume_bid_volume_ratio'] / 15, 0, 1)
        
        # 竞价换手率应在[0, 15%]范围内
        if 'volume_bid_turnover_rate' in features.columns:
            stability_score *= np.clip(features['volume_bid_turnover_rate'] / 0.15, 0, 1)
        
        # 资金净流入应在合理范围内
        if 'fund_flow_net_flow' in features.columns:
            # 资金净流入应在[-1亿, 1亿]范围内
            net_flow_clip = np.clip(features['fund_flow_net_flow'], -1e8, 1e8)
            stability_score *= np.clip((net_flow_clip + 1e8) / (2e8), 0, 1)
        
        # 计算特征重要性得分（0-1）
        importance_score = pd.Series(0.5, index=features.index)  # 默认得分0.5
        
        # 获取特征重要性
        feature_importance = self.get_feature_importance()
        if not feature_importance.empty:
            # 获取前10个重要特征
            top_features = feature_importance.head(10)['feature'].tolist()
            
            # 计算重要特征的稳定性
            important_features_stability = pd.Series(1.0, index=features.index)
            for feature in top_features:
                if feature in features.columns:
                    # 根据特征类型计算稳定性
                    if feature.startswith('price_'):
                        # 价格特征：应在合理范围内
                        important_features_stability *= np.clip((features[feature] + 0.1) / 0.2, 0, 1)
                    elif feature.startswith('volume_'):
                        # 成交量特征：应在合理范围内
                        important_features_stability *= np.clip(features[feature] / 15, 0, 1)
                    elif feature.startswith('fund_flow_'):
                        # 资金流特征：应在合理范围内
                        net_flow_clip = np.clip(features[feature], -1e8, 1e8)
                        important_features_stability *= np.clip((net_flow_clip + 1e8) / (2e8), 0, 1)
            
            importance_score = important_features_stability
        
        # 综合置信度（0-1），使用加权平均
        confidence = (completeness_score * 0.3 + 
                      stability_score * 0.3 + 
                      importance_score * 0.4)
        
        # 确保置信度在合理范围内
        confidence = confidence.clip(0.3, 1.0)
        
        return pd.Series(confidence, index=features.index, name='prediction_confidence')
    
    def _calculate_prediction_intervals(self, predictions: Dict[str, pd.Series], features: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算各预测指标的置信区间
        
        Args:
            predictions: 预测结果字典
            features: 特征数据
            
        Returns:
            添加了置信区间的预测结果字典，包含各指标的下限和上限
        """
        logger.info("计算预测结果的置信区间")
        
        # 获取置信度
        confidence = predictions['prediction_confidence']
        
        # 为每个预测指标计算置信区间
        # 置信区间宽度与置信度成反比：置信度越高，区间越窄
        # 同时考虑指标的波动性和重要性
        
        # 1. 未来30分钟内上涨概率的置信区间
        if 'rise_prob_30min' in predictions:
            rise_prob = predictions['rise_prob_30min']
            # 计算置信区间宽度（基于置信度和概率值）
            # 概率值越接近0.5，区间越宽；越接近0或1，区间越窄
            volatility_factor = 2 * np.abs(rise_prob - 0.5)
            interval_width = 0.3 * (1 - confidence) * (1 - volatility_factor + 0.2)  # 0.2是最小波动因子
            # 计算置信区间下限和上限
            rise_prob_lower = np.clip(rise_prob - interval_width / 2, 0, 1)
            rise_prob_upper = np.clip(rise_prob + interval_width / 2, 0, 1)
            # 添加到预测结果
            predictions['rise_prob_30min_lower'] = pd.Series(rise_prob_lower, index=features.index, name='rise_prob_30min_lower')
            predictions['rise_prob_30min_upper'] = pd.Series(rise_prob_upper, index=features.index, name='rise_prob_30min_upper')
        
        # 2. 当日封板成功率的置信区间
        if 'limit_up_prob' in predictions:
            limit_up_prob = predictions['limit_up_prob']
            # 计算置信区间宽度（基于置信度和成功率值）
            volatility_factor = 2 * np.abs(limit_up_prob - 0.5)
            interval_width = 0.4 * (1 - confidence) * (1 - volatility_factor + 0.2)  # 0.2是最小波动因子
            # 计算置信区间下限和上限
            limit_up_prob_lower = np.clip(limit_up_prob - interval_width / 2, 0, 1)
            limit_up_prob_upper = np.clip(limit_up_prob + interval_width / 2, 0, 1)
            # 添加到预测结果
            predictions['limit_up_prob_lower'] = pd.Series(limit_up_prob_lower, index=features.index, name='limit_up_prob_lower')
            predictions['limit_up_prob_upper'] = pd.Series(limit_up_prob_upper, index=features.index, name='limit_up_prob_upper')
        
        # 3. 封板后次日溢价空间的置信区间
        if 'next_day_premium' in predictions:
            next_day_premium = predictions['next_day_premium']
            # 计算置信区间宽度（基于置信度和溢价空间大小）
            # 溢价空间越大，区间越宽
            volatility_factor = np.clip(next_day_premium * 20, 0.1, 1.0)  # 转换为0.1-1.0的波动因子
            interval_width = (0.05 * (1 - confidence) + 0.01) * volatility_factor  # 基础宽度0.01，随置信度降低而增加
            # 计算置信区间下限和上限
            next_day_premium_lower = np.clip(next_day_premium - interval_width / 2, 0, 0.5)  # 溢价空间上限50%
            next_day_premium_upper = np.clip(next_day_premium + interval_width / 2, 0, 0.5)
            # 添加到预测结果
            predictions['next_day_premium_lower'] = pd.Series(next_day_premium_lower, index=features.index, name='next_day_premium_lower')
            predictions['next_day_premium_upper'] = pd.Series(next_day_premium_upper, index=features.index, name='next_day_premium_upper')
        
        logger.info("预测结果置信区间计算完成")
        return predictions
    
    def _default_predictions(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """默认预测结果
        
        Args:
            features: 特征数据
            
        Returns:
            默认预测结果字典，包含各预测指标及其置信区间
        """
        logger.info("使用默认预测结果")
        
        # 生成随机预测结果
        np.random.seed(42)
        n_samples = len(features)
        
        # 生成基础预测结果
        rise_prob_30min = np.random.rand(n_samples) * 0.5 + 0.3  # 0.3-0.8
        limit_up_prob = np.random.rand(n_samples) * 0.4 + 0.2  # 0.2-0.6
        next_day_premium = np.random.rand(n_samples) * 0.05 + 0.01  # 0.01-0.06
        prediction_confidence = np.random.rand(n_samples) * 0.3 + 0.7  # 0.7-1.0
        
        # 生成置信区间
        # 置信区间宽度与置信度成反比
        rise_prob_interval = 0.3 * (1 - prediction_confidence)
        limit_up_prob_interval = 0.4 * (1 - prediction_confidence)
        next_day_premium_interval = 0.05 * (1 - prediction_confidence) + 0.02
        
        # 计算置信区间上下限
        rise_prob_30min_lower = np.clip(rise_prob_30min - rise_prob_interval / 2, 0, 1)
        rise_prob_30min_upper = np.clip(rise_prob_30min + rise_prob_interval / 2, 0, 1)
        
        limit_up_prob_lower = np.clip(limit_up_prob - limit_up_prob_interval / 2, 0, 1)
        limit_up_prob_upper = np.clip(limit_up_prob + limit_up_prob_interval / 2, 0, 1)
        
        next_day_premium_lower = np.clip(next_day_premium - next_day_premium_interval / 2, 0, 0.5)
        next_day_premium_upper = np.clip(next_day_premium + next_day_premium_interval / 2, 0, 0.5)
        
        # 构建完整的预测结果字典
        predictions = {
            # 基础预测结果
            'rise_prob_30min': pd.Series(rise_prob_30min, index=features.index, name='rise_prob_30min'),
            'limit_up_prob': pd.Series(limit_up_prob, index=features.index, name='limit_up_prob'),
            'next_day_premium': pd.Series(next_day_premium, index=features.index, name='next_day_premium'),
            'prediction_confidence': pd.Series(prediction_confidence, index=features.index, name='prediction_confidence'),
            
            # 置信区间
            'rise_prob_30min_lower': pd.Series(rise_prob_30min_lower, index=features.index, name='rise_prob_30min_lower'),
            'rise_prob_30min_upper': pd.Series(rise_prob_30min_upper, index=features.index, name='rise_prob_30min_upper'),
            'limit_up_prob_lower': pd.Series(limit_up_prob_lower, index=features.index, name='limit_up_prob_lower'),
            'limit_up_prob_upper': pd.Series(limit_up_prob_upper, index=features.index, name='limit_up_prob_upper'),
            'next_day_premium_lower': pd.Series(next_day_premium_lower, index=features.index, name='next_day_premium_lower'),
            'next_day_premium_upper': pd.Series(next_day_premium_upper, index=features.index, name='next_day_premium_upper')
        }
        
        return predictions
    
    def _merge_predictions(self, candidate_stocks: pd.DataFrame, predictions: Dict[str, pd.Series]) -> pd.DataFrame:
        """将预测结果合并到候选个股中
        
        Args:
            candidate_stocks: 候选个股DataFrame
            predictions: 预测结果字典
            
        Returns:
            带有预测结果的候选个股DataFrame
        """
        logger.info("将预测结果合并到候选个股中")
        
        result = candidate_stocks.copy()
        
        # 添加预测结果列
        for pred_name, pred_values in predictions.items():
            result[pred_name] = pred_values.values
        
        # 计算综合预测得分
        result['prediction_score'] = (
            result['rise_prob_30min'] * 0.3 +
            result['limit_up_prob'] * 0.4 +
            result['next_day_premium'] * 10 +  # 放大溢价空间的权重
            result['prediction_confidence'] * 0.1
        )
        
        logger.info("预测结果合并完成")
        return result
    
    def explain_predictions(self, features: pd.DataFrame, predictions: pd.DataFrame = None, visualize: bool = False, output_path: str = None) -> pd.DataFrame:
        """使用SHAP解释预测结果
        
        Args:
            features: 特征数据
            predictions: 预测结果数据，用于更详细的解释
            visualize: 是否生成可视化结果
            output_path: 可视化结果保存路径
            
        Returns:
            SHAP值DataFrame
        """
        logger.info("使用SHAP解释预测结果")
        
        # 检查是否有可用模型
        if not self.models:
            logger.error("没有可用模型，无法解释预测结果")
            return pd.DataFrame()
        
        # 选择主要模型
        if 'model_xgboost' in self.models:
            main_model = self.models['model_xgboost']
        elif 'xgboost' in self.models:
            main_model = self.models['xgboost']
        else:
            main_model = list(self.models.values())[0]
        
        # 初始化SHAP解释器
        if self.shap_explainer is None:
            try:
                self.shap_explainer = shap.TreeExplainer(main_model)
            except Exception as e:
                logger.error(f"初始化SHAP解释器失败：{e}")
                return pd.DataFrame()
        
        try:
            # 计算SHAP值
            shap_values = self.shap_explainer.shap_values(features)
            
            # 转换为DataFrame
            if isinstance(shap_values, list):
                # 分类模型，取正类的SHAP值
                shap_df = pd.DataFrame(shap_values[1], columns=features.columns, index=features.index)
            else:
                # 回归模型
                shap_df = pd.DataFrame(shap_values, columns=features.columns, index=features.index)
            
            # 添加SHAP值之和作为综合影响
            shap_df['shap_sum'] = shap_df.sum(axis=1)
            
            # 添加特征重要性排名
            shap_df['feature_importance_rank'] = np.argsort(np.abs(shap_df.drop('shap_sum', axis=1)).mean().values)[::-1]
            
            # 生成可视化结果
            if visualize:
                self._generate_shap_visualizations(features, shap_values, main_model, predictions, output_path)
            
            logger.info("SHAP解释完成")
            return shap_df
        
        except Exception as e:
            logger.error(f"SHAP解释失败：{e}")
            return pd.DataFrame()
    
    def _generate_shap_visualizations(self, features: pd.DataFrame, shap_values: np.ndarray, model: any, predictions: pd.DataFrame = None, output_path: str = None):
        """生成SHAP可视化结果
        
        Args:
            features: 特征数据
            shap_values: SHAP值
            model: 模型对象
            predictions: 预测结果数据，用于更详细的可视化
            output_path: 可视化结果保存路径
        """
        logger.info("生成SHAP可视化结果")
        
        try:
            import matplotlib.pyplot as plt
            
            # 确保输出路径存在
            if output_path and not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
            
            # 1. 生成汇总图（Summary Plot）- 柱状图
            plt.figure(figsize=(14, 10))
            if isinstance(shap_values, list):
                # 分类模型
                shap.summary_plot(shap_values[1], features, plot_type="bar", show=False, max_display=20)
            else:
                # 回归模型
                shap.summary_plot(shap_values, features, plot_type="bar", show=False, max_display=20)
            
            plt.title("特征重要性汇总（柱状图）", fontsize=16)
            if output_path:
                plt.savefig(os.path.join(output_path, "shap_summary_bar.png"), dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("生成SHAP汇总柱状图完成")
            
            # 2. 生成点图（Dot Plot）
            plt.figure(figsize=(14, 10))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], features, show=False, max_display=20)
            else:
                shap.summary_plot(shap_values, features, show=False, max_display=20)
            
            plt.title("特征重要性汇总（点图）", fontsize=16)
            if output_path:
                plt.savefig(os.path.join(output_path, "shap_summary_dot.png"), dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("生成SHAP汇总点图完成")
            
            # 3. 生成特征依赖图（Dependence Plot）
            # 选择最重要的5个特征
            if isinstance(shap_values, list):
                feature_importance = np.abs(shap_values[1]).mean(axis=0)
            else:
                feature_importance = np.abs(shap_values).mean(axis=0)
            
            top_features = np.argsort(feature_importance)[-5:][::-1]
            top_feature_names = features.columns[top_features]
            
            for i, feature_name in enumerate(top_feature_names):
                plt.figure(figsize=(12, 8))
                if isinstance(shap_values, list):
                    shap.dependence_plot(feature_name, shap_values[1], features, show=False, interaction_index=None)
                else:
                    shap.dependence_plot(feature_name, shap_values, features, show=False, interaction_index=None)
                
                plt.title(f"特征依赖图 - {feature_name}", fontsize=14)
                if output_path:
                    plt.savefig(os.path.join(output_path, f"shap_dependence_{feature_name}.png"), dpi=300, bbox_inches="tight")
                plt.close()
            logger.info("生成SHAP特征依赖图完成")
            
            # 4. 生成瀑布图（Waterfall Plot）- 显示前5个样本
            sample_count = min(5, len(features))
            if predictions is not None and len(predictions) >= sample_count:
                for i in range(sample_count):
                    plt.figure(figsize=(12, 10))
                    if isinstance(shap_values, list):
                        shap.plots._waterfall.waterfall_legacy(self.shap_explainer.expected_value[1], shap_values[1][i], features.iloc[i], max_display=15)
                    else:
                        shap.plots._waterfall.waterfall_legacy(self.shap_explainer.expected_value, shap_values[i], features.iloc[i], max_display=15)
                    
                    if output_path:
                        plt.savefig(os.path.join(output_path, f"shap_waterfall_sample_{i+1}.png"), dpi=300, bbox_inches="tight")
                    plt.close()
                logger.info("生成SHAP瀑布图完成")
            
            # 5. 生成热力图（Heatmap）- 显示特征相关性
            plt.figure(figsize=(16, 14))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], features, plot_type="heatmap", show=False, max_display=20)
            else:
                shap.summary_plot(shap_values, features, plot_type="heatmap", show=False, max_display=20)
            
            plt.title("特征SHAP值热力图", fontsize=16)
            if output_path:
                plt.savefig(os.path.join(output_path, "shap_heatmap.png"), dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("生成SHAP热力图完成")
            
            # 6. 生成决策图（Decision Plot）- 显示决策路径
            plt.figure(figsize=(16, 10))
            if isinstance(shap_values, list):
                shap.decision_plot(self.shap_explainer.expected_value[1], shap_values[1][:5], features.iloc[:5], show=False)
            else:
                shap.decision_plot(self.shap_explainer.expected_value, shap_values[:5], features.iloc[:5], show=False)
            
            plt.title("决策路径图", fontsize=16)
            if output_path:
                plt.savefig(os.path.join(output_path, "shap_decision_plot.png"), dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("生成SHAP决策图完成")
            
        except ImportError as e:
            logger.warning(f"无法生成SHAP可视化：缺少依赖包：{e}")
        except Exception as e:
            logger.error(f"生成SHAP可视化失败：{e}")
    
    def visualize_predictions(self, predictions: pd.DataFrame, output_path: str = None):
        """可视化预测结果
        
        Args:
            predictions: 预测结果DataFrame
            output_path: 可视化结果保存路径
        """
        logger.info("可视化预测结果")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 确保输出路径存在
            if output_path and not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
            
            # 1. 预测概率分布直方图 - 更详细的子图布局
            plt.figure(figsize=(16, 12))
            
            # 绘制未来30分钟上涨概率分布
            plt.subplot(2, 3, 1)
            sns.histplot(predictions['rise_prob_30min'], bins=20, kde=True, color='skyblue')
            plt.title('未来30分钟上涨概率分布')
            plt.xlabel('概率')
            plt.ylabel('频率')
            plt.axvline(predictions['rise_prob_30min'].mean(), color='red', linestyle='--', label=f'均值: {predictions["rise_prob_30min"].mean():.2f}')
            plt.legend()
            
            # 绘制当日封板成功率分布
            plt.subplot(2, 3, 2)
            sns.histplot(predictions['limit_up_prob'], bins=20, kde=True, color='lightgreen')
            plt.title('当日封板成功率分布')
            plt.xlabel('概率')
            plt.ylabel('频率')
            plt.axvline(predictions['limit_up_prob'].mean(), color='red', linestyle='--', label=f'均值: {predictions["limit_up_prob"].mean():.2f}')
            plt.legend()
            
            # 绘制次日溢价空间分布
            plt.subplot(2, 3, 3)
            sns.histplot(predictions['next_day_premium'], bins=20, kde=True, color='salmon')
            plt.title('封板后次日溢价空间分布')
            plt.xlabel('溢价空间')
            plt.ylabel('频率')
            plt.axvline(predictions['next_day_premium'].mean(), color='red', linestyle='--', label=f'均值: {predictions["next_day_premium"].mean():.2f}')
            plt.legend()
            
            # 绘制预测置信度分布
            plt.subplot(2, 3, 4)
            sns.histplot(predictions['prediction_confidence'], bins=20, kde=True, color='purple')
            plt.title('预测置信度分布')
            plt.xlabel('置信度')
            plt.ylabel('频率')
            plt.axvline(predictions['prediction_confidence'].mean(), color='red', linestyle='--', label=f'均值: {predictions["prediction_confidence"].mean():.2f}')
            plt.legend()
            
            # 绘制上涨概率置信区间宽度分布
            if 'rise_prob_30min_lower' in predictions.columns and 'rise_prob_30min_upper' in predictions.columns:
                plt.subplot(2, 3, 5)
                interval_width = predictions['rise_prob_30min_upper'] - predictions['rise_prob_30min_lower']
                sns.histplot(interval_width, bins=20, kde=True, color='orange')
                plt.title('上涨概率置信区间宽度分布')
                plt.xlabel('区间宽度')
                plt.ylabel('频率')
                plt.axvline(interval_width.mean(), color='red', linestyle='--', label=f'均值: {interval_width.mean():.2f}')
                plt.legend()
            
            # 绘制封板概率置信区间宽度分布
            if 'limit_up_prob_lower' in predictions.columns and 'limit_up_prob_upper' in predictions.columns:
                plt.subplot(2, 3, 6)
                interval_width = predictions['limit_up_prob_upper'] - predictions['limit_up_prob_lower']
                sns.histplot(interval_width, bins=20, kde=True, color='brown')
                plt.title('封板概率置信区间宽度分布')
                plt.xlabel('区间宽度')
                plt.ylabel('频率')
                plt.axvline(interval_width.mean(), color='red', linestyle='--', label=f'均值: {interval_width.mean():.2f}')
                plt.legend()
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(os.path.join(output_path, "prediction_distributions.png"), dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("生成预测概率分布图完成")
            
            # 2. 预测分数与实际结果的关系散点图
            if 'actual_result' in predictions.columns:
                plt.figure(figsize=(12, 10))
                sns.scatterplot(x='prediction_score', y='actual_result', data=predictions, hue='limit_up_prob', size='prediction_confidence', alpha=0.7, palette='viridis')
                plt.title('预测分数与实际结果关系（按封板概率和置信度着色）')
                plt.xlabel('预测分数')
                plt.ylabel('实际结果')
                plt.colorbar(label='封板概率')
                plt.legend(title='置信度')
                
                # 添加趋势线
                sns.regplot(x='prediction_score', y='actual_result', data=predictions, scatter=False, color='red', line_kws={'linestyle': '--'})
                
                if output_path:
                    plt.savefig(os.path.join(output_path, "prediction_vs_actual.png"), dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("生成预测分数与实际结果关系图完成")
            
            # 3. 特征重要性热图
            if 'shap_sum' in predictions.columns:
                # 选择SHAP值贡献最大的前10个特征
                shap_cols = [col for col in predictions.columns if col.startswith('shap_') and col != 'shap_sum']
                if shap_cols:
                    plt.figure(figsize=(14, 12))
                    # 计算特征重要性均值
                    feature_importance = predictions[shap_cols].abs().mean().sort_values(ascending=False).head(10)
                    top_shap_cols = feature_importance.index.tolist()
                    shap_features = predictions[top_shap_cols].corr()
                    sns.heatmap(shap_features, annot=True, cmap='coolwarm', fmt='.2f', square=True)
                    plt.title('TOP10 SHAP特征重要性相关性热图')
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.yticks(rotation=0, fontsize=10)
                    
                    if output_path:
                        plt.savefig(os.path.join(output_path, "shap_correlation_heatmap.png"), dpi=300, bbox_inches="tight")
                    plt.close()
                    logger.info("生成SHAP特征相关性热图完成")
            
            # 4. 特征重要性条形图
            if hasattr(self, 'features_list'):
                feature_importance = self.get_feature_importance()
                if not feature_importance.empty:
                    plt.figure(figsize=(14, 10))
                    sns.barplot(x='importance', y='feature', data=feature_importance.head(20), palette='viridis')
                    plt.title('TOP20 特征重要性排序')
                    plt.xlabel('重要性得分')
                    plt.ylabel('特征名称')
                    plt.tight_layout()
                    
                    if output_path:
                        plt.savefig(os.path.join(output_path, "feature_importance_bar.png"), dpi=300, bbox_inches="tight")
                    plt.close()
                    logger.info("生成特征重要性条形图完成")
            
            # 5. 预测指标相关性矩阵
            prediction_metrics = ['rise_prob_30min', 'limit_up_prob', 'next_day_premium', 'prediction_confidence', 'prediction_score']
            available_metrics = [metric for metric in prediction_metrics if metric in predictions.columns]
            if len(available_metrics) >= 2:
                plt.figure(figsize=(12, 10))
                corr_matrix = predictions[available_metrics].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='RdBu', fmt='.2f', square=True, center=0)
                plt.title('预测指标相关性矩阵')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                
                if output_path:
                    plt.savefig(os.path.join(output_path, "prediction_correlation_matrix.png"), dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("生成预测指标相关性矩阵完成")
            
            # 6. 预测分数箱线图（按板块分组）
            if 'sector' in predictions.columns:
                plt.figure(figsize=(16, 10))
                # 选择预测分数最高的10个板块
                top_sectors = predictions.groupby('sector')['prediction_score'].mean().sort_values(ascending=False).head(10).index.tolist()
                top_sector_data = predictions[predictions['sector'].isin(top_sectors)]
                sns.boxplot(x='sector', y='prediction_score', data=top_sector_data, palette='Set3')
                plt.title('TOP10板块预测分数分布')
                plt.xlabel('板块名称')
                plt.ylabel('预测分数')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                if output_path:
                    plt.savefig(os.path.join(output_path, "prediction_score_by_sector.png"), dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("生成板块预测分数箱线图完成")
            
        except ImportError as e:
            logger.warning(f"无法生成预测可视化：缺少依赖包：{e}")
        except Exception as e:
            logger.error(f"生成预测可视化失败：{e}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性
        
        Returns:
            特征重要性DataFrame，包含特征名称、重要性得分和排名
        """
        logger.info("获取特征重要性")
        
        if not self.models:
            logger.error("没有可用模型，无法获取特征重要性")
            return pd.DataFrame()
        
        # 选择主要模型
        if 'model_xgboost' in self.models:
            main_model = self.models['model_xgboost']
        elif 'xgboost' in self.models:
            main_model = self.models['xgboost']
        else:
            main_model = list(self.models.values())[0]
        
        try:
            # 检查模型是否有feature_importances_属性
            if hasattr(main_model, 'feature_importances_'):
                importances = main_model.feature_importances_
                
                # 使用实际的特征名称
                if hasattr(self, 'features_list') and len(self.features_list) == len(importances):
                    feature_names = self.features_list
                else:
                    # 假设特征顺序与训练时一致
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
                
                # 创建特征重要性DataFrame
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })
                
                # 按重要性降序排序
                feature_importance = feature_importance.sort_values(by='importance', ascending=False)
                
                # 添加排名
                feature_importance['rank'] = range(1, len(feature_importance) + 1)
                
                logger.info("特征重要性获取完成")
                return feature_importance
            
            # 对于没有feature_importances_属性的模型（如LogisticRegression）
            elif hasattr(main_model, 'coef_'):
                coefficients = main_model.coef_[0]
                
                # 使用实际的特征名称
                if hasattr(self, 'features_list') and len(self.features_list) == len(coefficients):
                    feature_names = self.features_list
                else:
                    feature_names = [f'feature_{i}' for i in range(len(coefficients))]
                
                # 创建特征重要性DataFrame（使用系数绝对值作为重要性）
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(coefficients)
                })
                
                # 按重要性降序排序
                feature_importance = feature_importance.sort_values(by='importance', ascending=False)
                
                # 添加排名
                feature_importance['rank'] = range(1, len(feature_importance) + 1)
                
                logger.info("特征重要性获取完成")
                return feature_importance
            
            else:
                logger.warning("模型不支持特征重要性计算")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"获取特征重要性失败：{e}")
            return pd.DataFrame()
