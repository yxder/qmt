#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征标准化与归一化模块
负责对股票特征进行标准化、归一化和特征选择处理
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils.logger import setup_logger

logger = setup_logger()

class FeatureStandardizer:
    """特征标准化与归一化类"""
    
    def __init__(self):
        """初始化特征标准化器"""
        logger.info("初始化特征标准化器")
        self.scalers = {}
        self.selectors = {}
        self.feature_columns = None
    
    def standardize_features(self, features, method='zscore', columns=None):
        """
        对特征进行标准化处理
        
        Args:
            features: 待标准化的特征数据
            method: 标准化方法，可选值：zscore, minmax, robust
            columns: 指定需要标准化的列，默认为所有数值列
            
        Returns:
            标准化后的特征数据
        """
        logger.info(f"使用{method}方法进行特征标准化")
        
        if features is None or features.empty:
            logger.warning("特征数据为空，跳过标准化")
            return features
        
        standardized_features = features.copy()
        
        # 确定需要标准化的列
        if columns is None:
            # 选择所有数值列
            columns = standardized_features.select_dtypes(include=[np.number]).columns
        
        if not columns.any():
            logger.warning("没有找到数值列，跳过标准化")
            return features
        
        try:
            # 根据方法选择标准化器
            if method == 'zscore':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                logger.warning(f"未知的标准化方法：{method}，使用默认的zscore方法")
                scaler = StandardScaler()
            
            # 保存特征列
            if self.feature_columns is None:
                self.feature_columns = columns
            
            # 对指定列进行标准化
            standardized_features[columns] = scaler.fit_transform(standardized_features[columns])
            
            # 保存标准化器
            self.scalers[method] = scaler
            
            logger.info(f"特征标准化完成，处理了{len(columns)}个特征")
            return standardized_features
            
        except Exception as e:
            logger.error(f"特征标准化失败：{e}")
            return features
    
    def normalize_features(self, features, columns=None, min_val=0, max_val=1):
        """
        对特征进行归一化处理
        
        Args:
            features: 待归一化的特征数据
            columns: 指定需要归一化的列，默认为所有数值列
            min_val: 归一化后的最小值
            max_val: 归一化后的最大值
            
        Returns:
            归一化后的特征数据
        """
        logger.info(f"对特征进行归一化处理，范围：[{min_val}, {max_val}]")
        
        if features is None or features.empty:
            logger.warning("特征数据为空，跳过归一化")
            return features
        
        normalized_features = features.copy()
        
        # 确定需要归一化的列
        if columns is None:
            # 选择所有数值列
            columns = normalized_features.select_dtypes(include=[np.number]).columns
        
        if not columns.any():
            logger.warning("没有找到数值列，跳过归一化")
            return features
        
        try:
            scaler = MinMaxScaler(feature_range=(min_val, max_val))
            normalized_features[columns] = scaler.fit_transform(normalized_features[columns])
            
            # 保存归一化器
            self.scalers['normalize'] = scaler
            
            logger.info(f"特征归一化完成，处理了{len(columns)}个特征")
            return normalized_features
            
        except Exception as e:
            logger.error(f"特征归一化失败：{e}")
            return features
    
    def select_features(self, features, labels, method='variance', threshold=0.01, k=100):
        """
        对特征进行选择
        
        Args:
            features: 待选择的特征数据
            labels: 对应的标签数据
            method: 特征选择方法，可选值：variance, kbest, embedded
            threshold: 方差阈值，仅用于variance方法
            k: 选择的特征数量，仅用于kbest方法
            
        Returns:
            选择后的特征数据
        """
        logger.info(f"使用{method}方法进行特征选择")
        
        if features is None or features.empty:
            logger.warning("特征数据为空，跳过特征选择")
            return features
        
        selected_features = features.copy()
        
        try:
            if method == 'variance':
                # 方差过滤
                logger.info(f"使用方差过滤，阈值：{threshold}")
                selector = VarianceThreshold(threshold=threshold)
                selector.fit(selected_features)
                selected_columns = selected_features.columns[selector.get_support()]
                selected_features = selected_features[selected_columns]
            
            elif method == 'kbest':
                # 基于统计测试的特征选择
                logger.info(f"使用KBest特征选择，选择前{k}个特征")
                selector = SelectKBest(score_func=f_classif, k=min(k, len(selected_features.columns)))
                selector.fit(selected_features, labels)
                selected_columns = selected_features.columns[selector.get_support()]
                selected_features = selected_features[selected_columns]
            
            elif method == 'embedded':
                # 嵌入式特征选择
                logger.info("使用嵌入式特征选择")
                # 使用随机森林进行特征选择
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                selector = SelectFromModel(estimator=rf)
                selector.fit(selected_features, labels)
                selected_columns = selected_features.columns[selector.get_support()]
                selected_features = selected_features[selected_columns]
            
            else:
                logger.warning(f"未知的特征选择方法：{method}，跳过特征选择")
                return features
            
            # 保存选择器
            self.selectors[method] = selector
            
            logger.info(f"特征选择完成，保留{len(selected_features.columns)}个特征，删除{len(features.columns) - len(selected_features.columns)}个特征")
            return selected_features
            
        except Exception as e:
            logger.error(f"特征选择失败：{e}")
            return features
    
    def filter_low_correlation_features(self, features, labels, correlation_threshold=0.05):
        """
        过滤与标签相关性低的特征
        
        Args:
            features: 待过滤的特征数据
            labels: 对应的标签数据
            correlation_threshold: 相关性阈值，低于此值的特征将被过滤
            
        Returns:
            过滤后的特征数据
        """
        logger.info(f"过滤与标签相关性低于{correlation_threshold}的特征")
        
        if features is None or features.empty:
            logger.warning("特征数据为空，跳过相关性过滤")
            return features
        
        try:
            # 计算每个特征与标签的相关系数
            correlations = []
            for col in features.columns:
                if pd.api.types.is_numeric_dtype(features[col]):
                    # 使用Spearman相关系数，适合非线性关系
                    corr = features[col].corr(labels, method='spearman')
                    correlations.append((col, abs(corr)))
            
            # 过滤相关性低于阈值的特征
            selected_columns = [col for col, corr in correlations if corr >= correlation_threshold]
            
            if not selected_columns:
                logger.warning("所有特征相关性都低于阈值，保留所有特征")
                return features
            
            filtered_features = features[selected_columns]
            logger.info(f"相关性过滤完成，保留{len(selected_columns)}个特征，删除{len(features.columns) - len(selected_columns)}个特征")
            return filtered_features
            
        except Exception as e:
            logger.error(f"相关性过滤失败：{e}")
            return features
    
    def remove_highly_correlated_features(self, features, correlation_threshold=0.8):
        """
        移除高度相关的特征
        
        Args:
            features: 待处理的特征数据
            correlation_threshold: 相关性阈值，高于此值的特征对将被处理
            
        Returns:
            处理后的特征数据
        """
        logger.info(f"移除相关性高于{correlation_threshold}的特征")
        
        if features is None or features.empty:
            logger.warning("特征数据为空，跳过相关性处理")
            return features
        
        try:
            # 计算特征相关性矩阵
            corr_matrix = features.corr().abs()
            
            # 创建一个布尔矩阵，标记需要删除的特征
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # 找到相关性高于阈值的列
            to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
            
            if to_drop:
                features = features.drop(to_drop, axis=1)
                logger.info(f"移除高度相关特征完成，删除{len(to_drop)}个特征")
            else:
                logger.info("没有找到高度相关的特征")
            
            return features
            
        except Exception as e:
            logger.error(f"移除高度相关特征失败：{e}")
            return features
    
    def process_features(self, features, labels=None, config=None):
        """
        完整的特征处理流水线
        
        Args:
            features: 待处理的特征数据
            labels: 对应的标签数据
            config: 处理配置
            
        Returns:
            处理后的特征数据
        """
        logger.info("开始特征处理流水线")
        
        if features is None or features.empty:
            logger.warning("特征数据为空，跳过特征处理")
            return features
        
        # 默认配置
        default_config = {
            'standardize_method': 'zscore',
            'normalize': False,
            'feature_selection': True,
            'selection_method': 'variance',
            'variance_threshold': 0.01,
            'correlation_threshold': 0.05,
            'remove_high_corr': True,
            'high_corr_threshold': 0.8
        }
        
        config = config or default_config
        
        processed_features = features.copy()
        
        try:
            # 1. 标准化处理
            if config.get('standardize', True):
                processed_features = self.standardize_features(
                    processed_features, 
                    method=config['standardize_method']
                )
            
            # 2. 归一化处理
            if config.get('normalize', False):
                processed_features = self.normalize_features(processed_features)
            
            # 3. 过滤低相关性特征（需要标签）
            if config.get('filter_low_corr', True) and labels is not None:
                processed_features = self.filter_low_correlation_features(
                    processed_features, 
                    labels, 
                    correlation_threshold=config['correlation_threshold']
                )
            
            # 4. 移除高度相关特征
            if config.get('remove_high_corr', True):
                processed_features = self.remove_highly_correlated_features(
                    processed_features, 
                    correlation_threshold=config['high_corr_threshold']
                )
            
            # 5. 特征选择（需要标签）
            if config.get('feature_selection', True) and labels is not None:
                processed_features = self.select_features(
                    processed_features, 
                    labels, 
                    method=config['selection_method'],
                    threshold=config['variance_threshold']
                )
            
            logger.info("特征处理流水线完成")
            return processed_features
            
        except Exception as e:
            logger.error(f"特征处理流水线失败：{e}")
            return features
    
    def transform_new_features(self, features):
        """
        使用已训练的标准化器和选择器处理新的特征数据
        
        Args:
            features: 待处理的新特征数据
            
        Returns:
            处理后的特征数据
        """
        logger.info("处理新的特征数据")
        
        if features is None or features.empty:
            logger.warning("特征数据为空，跳过处理")
            return features
        
        transformed_features = features.copy()
        
        try:
            # 1. 应用标准化
            for method, scaler in self.scalers.items():
                if self.feature_columns is not None:
                    # 只处理之前处理过的列
                    common_columns = [col for col in self.feature_columns if col in transformed_features.columns]
                    if common_columns:
                        transformed_features[common_columns] = scaler.transform(transformed_features[common_columns])
            
            # 2. 应用特征选择
            for method, selector in self.selectors.items():
                if hasattr(selector, 'get_support'):
                    selected_columns = transformed_features.columns[selector.get_support()]
                    transformed_features = transformed_features[selected_columns]
            
            logger.info("新特征数据处理完成")
            return transformed_features
            
        except Exception as e:
            logger.error(f"处理新特征数据失败：{e}")
            return features
    
    def get_feature_importance(self, features, labels, method='random_forest'):
        """
        获取特征重要性
        
        Args:
            features: 特征数据
            labels: 标签数据
            method: 特征重要性计算方法
            
        Returns:
            特征重要性字典
        """
        logger.info(f"使用{method}方法计算特征重要性")
        
        try:
            if method == 'random_forest':
                # 使用随机森林计算特征重要性
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(features, labels)
                importances = rf.feature_importances_
            
            elif method == 'logistic':
                # 使用逻辑回归计算特征重要性
                lr = LogisticRegression(penalty='l1', solver='saga', random_state=42)
                lr.fit(features, labels)
                importances = np.abs(lr.coef_[0])
            
            else:
                logger.warning(f"未知的特征重要性计算方法：{method}")
                return {}
            
            # 构建特征重要性字典
            feature_importance = {}
            for col, importance in zip(features.columns, importances):
                feature_importance[col] = importance
            
            # 按重要性排序
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            logger.info("特征重要性计算完成")
            return feature_importance
            
        except Exception as e:
            logger.error(f"计算特征重要性失败：{e}")
            return {}
    
    def save_scalers(self, file_path):
        """
        保存标准化器和选择器
        
        Args:
            file_path: 保存路径
        """
        import joblib
        
        try:
            save_data = {
                'scalers': self.scalers,
                'selectors': self.selectors,
                'feature_columns': self.feature_columns
            }
            
            joblib.dump(save_data, file_path)
            logger.info(f"标准化器和选择器保存成功：{file_path}")
            
        except Exception as e:
            logger.error(f"保存标准化器和选择器失败：{e}")
    
    def load_scalers(self, file_path):
        """
        加载标准化器和选择器
        
        Args:
            file_path: 加载路径
        """
        import joblib
        
        try:
            load_data = joblib.load(file_path)
            self.scalers = load_data.get('scalers', {})
            self.selectors = load_data.get('selectors', {})
            self.feature_columns = load_data.get('feature_columns', None)
            
            logger.info(f"标准化器和选择器加载成功：{file_path}")
            
        except Exception as e:
            logger.error(f"加载标准化器和选择器失败：{e}")
