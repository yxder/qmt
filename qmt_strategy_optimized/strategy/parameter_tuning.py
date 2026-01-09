#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数调优模块
负责对模型和策略的关键参数进行系统性优化
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils.logger import setup_logger
import json
import os

# 先初始化logger
logger = setup_logger()

# 尝试导入optuna，如果失败则设置为None
try:
    import optuna
except ImportError:
    optuna = None
    logger.warning("optuna库未安装，将无法使用optuna_tuning方法")

class ParameterTuner:
    """参数调优类"""
    
    def __init__(self):
        """初始化参数调优器"""
        logger.info("初始化参数调优器")
        self.tuning_results = {}
    
    def grid_search_tuning(self, model, X, y, param_grid, scoring='f1', cv=5):
        """
        使用网格搜索进行参数调优
        
        Args:
            model: 待调优的模型
            X: 特征数据
            y: 标签数据
            param_grid: 参数网格
            scoring: 评估指标
            cv: 交叉验证折数
            
        Returns:
            调优结果
        """
        logger.info("开始网格搜索参数调优")
        
        try:
            # 创建时间序列交叉验证分割器
            tscv = TimeSeriesSplit(n_splits=cv)
            
            # 创建评分器
            scorers = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1': make_scorer(f1_score),
                'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
            }
            
            # 进行网格搜索
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=scorers,
                refit=scoring,
                cv=tscv,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            # 保存调优结果
            result = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_,
                'best_estimator': grid_search.best_estimator_
            }
            
            self.tuning_results['grid_search'] = result
            
            logger.info(f"网格搜索调优完成，最佳参数：{result['best_params']}，最佳{scoring}分数：{result['best_score']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"网格搜索调优失败：{e}")
            return None
    
    def random_search_tuning(self, model, X, y, param_distributions, n_iter=100, scoring='f1', cv=5):
        """
        使用随机搜索进行参数调优
        
        Args:
            model: 待调优的模型
            X: 特征数据
            y: 标签数据
            param_distributions: 参数分布
            n_iter: 迭代次数
            scoring: 评估指标
            cv: 交叉验证折数
            
        Returns:
            调优结果
        """
        logger.info("开始随机搜索参数调优")
        
        try:
            # 创建时间序列交叉验证分割器
            tscv = TimeSeriesSplit(n_splits=cv)
            
            # 创建评分器
            scorers = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1': make_scorer(f1_score),
                'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
            }
            
            # 进行随机搜索
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring=scorers,
                refit=scoring,
                cv=tscv,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            
            random_search.fit(X, y)
            
            # 保存调优结果
            result = {
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'cv_results': random_search.cv_results_,
                'best_estimator': random_search.best_estimator_
            }
            
            self.tuning_results['random_search'] = result
            
            logger.info(f"随机搜索调优完成，最佳参数：{result['best_params']}，最佳{scoring}分数：{result['best_score']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"随机搜索调优失败：{e}")
            return None
    
    def optuna_tuning(self, model_class, X, y, param_space_func, n_trials=100, scoring='f1', cv=5):
        """
        使用Optuna进行参数调优
        
        Args:
            model_class: 模型类
            X: 特征数据
            y: 标签数据
            param_space_func: 参数空间生成函数
            n_trials: 试验次数
            scoring: 评估指标
            cv: 交叉验证折数
        
        Returns:
            调优结果
        """
        logger.info("开始Optuna参数调优")
        
        # 检查optuna是否可用
        if optuna is None:
            logger.error("optuna库未安装，无法使用optuna_tuning方法")
            return None
        
        def objective(trial):
            """Optuna目标函数"""
            # 获取参数
            params = param_space_func(trial)
            
            # 创建模型
            model = model_class(**params)
            
            # 进行时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=cv)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测
                y_pred = model.predict(X_test)
                
                # 计算分数
                if scoring == 'accuracy':
                    score = accuracy_score(y_test, y_pred)
                elif scoring == 'precision':
                    score = precision_score(y_test, y_pred)
                elif scoring == 'recall':
                    score = recall_score(y_test, y_pred)
                elif scoring == 'f1':
                    score = f1_score(y_test, y_pred)
                elif scoring == 'roc_auc':
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    score = roc_auc_score(y_test, y_pred_proba)
                else:
                    score = accuracy_score(y_test, y_pred)
                
                scores.append(score)
            
            return np.mean(scores)
        
        try:
            # 创建研究对象
            study = optuna.create_study(direction='maximize', study_name='stock_prediction')
            
            # 运行优化
            study.optimize(objective, n_trials=n_trials, n_jobs=-1)
            
            # 保存调优结果
            result = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'trials': study.trials_dataframe().to_dict()
            }
            
            self.tuning_results['optuna'] = result
            
            logger.info(f"Optuna调优完成，最佳参数：{result['best_params']}，最佳{scoring}分数：{result['best_score']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Optuna调优失败：{e}")
            return None
    
    def tune_threshold_params(self, model, X, y, thresholds, scoring='f1'):
        """
        调优阈值参数（如BUY_THRESHOLD）
        
        Args:
            model: 训练好的模型
            X: 特征数据
            y: 标签数据
            thresholds: 阈值列表
            scoring: 评估指标
            
        Returns:
            最佳阈值和对应的评估结果
        """
        logger.info("开始阈值参数调优")
        
        try:
            # 预测概率
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            results = []
            
            for threshold in thresholds:
                # 根据阈值生成预测标签
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                # 计算各评估指标
                metrics = {
                    'threshold': threshold,
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred),
                    'recall': recall_score(y, y_pred),
                    'f1': f1_score(y, y_pred),
                    'roc_auc': roc_auc_score(y, y_pred_proba)
                }
                
                results.append(metrics)
            
            # 转换为DataFrame
            results_df = pd.DataFrame(results)
            
            # 找到最佳阈值
            best_result = results_df.sort_values(by=scoring, ascending=False).iloc[0]
            
            # 保存结果
            self.tuning_results['threshold_tuning'] = {
                'best_threshold': best_result['threshold'],
                'best_metrics': best_result.to_dict(),
                'all_results': results_df.to_dict()
            }
            
            logger.info(f"阈值调优完成，最佳{scoring}分数对应的阈值：{best_result['threshold']:.4f}")
            logger.info(f"最佳阈值对应的指标：{best_result.to_dict()}")
            
            return best_result
            
        except Exception as e:
            logger.error(f"阈值调优失败：{e}")
            return None
    
    def tune_strategy_params(self, strategy_func, param_grid, X, y, scoring='f1'):
        """
        调优策略参数
        
        Args:
            strategy_func: 策略函数
            param_grid: 参数网格
            X: 特征数据
            y: 标签数据
            scoring: 评估指标
            
        Returns:
            调优结果
        """
        logger.info("开始策略参数调优")
        
        try:
            best_score = -1
            best_params = None
            
            # 遍历参数网格
            for params in self._generate_param_combinations(param_grid):
                # 执行策略
                y_pred = strategy_func(X, **params)
                
                # 计算分数
                if scoring == 'accuracy':
                    score = accuracy_score(y, y_pred)
                elif scoring == 'precision':
                    score = precision_score(y, y_pred)
                elif scoring == 'recall':
                    score = recall_score(y, y_pred)
                elif scoring == 'f1':
                    score = f1_score(y, y_pred)
                else:
                    score = accuracy_score(y, y_pred)
                
                # 更新最佳结果
                if score > best_score:
                    best_score = score
                    best_params = params
            
            # 保存结果
            result = {
                'best_params': best_params,
                'best_score': best_score
            }
            
            self.tuning_results['strategy_tuning'] = result
            
            logger.info(f"策略参数调优完成，最佳参数：{best_params}，最佳{scoring}分数：{best_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"策略参数调优失败：{e}")
            return None
    
    def sensitivity_analysis(self, model, X, y, param_names, param_ranges, scoring='f1'):
        """
        进行参数敏感性分析
        
        Args:
            model: 训练好的模型
            X: 特征数据
            y: 标签数据
            param_names: 参数名称列表
            param_ranges: 参数范围列表
            scoring: 评估指标
            
        Returns:
            敏感性分析结果
        """
        logger.info("开始参数敏感性分析")
        
        try:
            results = {}
            
            for param_name, param_range in zip(param_names, param_ranges):
                param_results = []
                
                for param_value in param_range:
                    # 创建临时模型
                    temp_model = model.__class__(**{**model.get_params(), param_name: param_value})
                    
                    # 训练模型
                    temp_model.fit(X, y)
                    
                    # 预测
                    y_pred = temp_model.predict(X)
                    
                    # 计算分数
                    if scoring == 'accuracy':
                        score = accuracy_score(y, y_pred)
                    elif scoring == 'precision':
                        score = precision_score(y, y_pred)
                    elif scoring == 'recall':
                        score = recall_score(y, y_pred)
                    elif scoring == 'f1':
                        score = f1_score(y, y_pred)
                    elif scoring == 'roc_auc':
                        y_pred_proba = temp_model.predict_proba(X)[:, 1]
                        score = roc_auc_score(y, y_pred_proba)
                    else:
                        score = accuracy_score(y, y_pred)
                    
                    param_results.append({
                        param_name: param_value,
                        scoring: score
                    })
                
                results[param_name] = param_results
            
            # 保存结果
            self.tuning_results['sensitivity_analysis'] = results
            
            logger.info("参数敏感性分析完成")
            return results
            
        except Exception as e:
            logger.error(f"参数敏感性分析失败：{e}")
            return None
    
    def _generate_param_combinations(self, param_grid):
        """
        生成参数组合
        
        Args:
            param_grid: 参数网格
            
        Returns:
            参数组合生成器
        """
        from itertools import product
        
        # 获取参数名称和值列表
        keys = param_grid.keys()
        values = param_grid.values()
        
        # 生成所有组合
        for combination in product(*values):
            yield dict(zip(keys, combination))
    
    def save_tuning_results(self, file_path):
        """
        保存调优结果
        
        Args:
            file_path: 保存路径
        """
        logger.info(f"保存调优结果到：{file_path}")
        
        try:
            # 创建目录
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存结果
            with open(file_path, 'w') as f:
                json.dump(self.tuning_results, f, indent=2, default=str)
            
            logger.info("调优结果保存成功")
            
        except Exception as e:
            logger.error(f"保存调优结果失败：{e}")
    
    def load_tuning_results(self, file_path):
        """
        加载调优结果
        
        Args:
            file_path: 加载路径
            
        Returns:
            调优结果
        """
        logger.info(f"加载调优结果从：{file_path}")
        
        try:
            with open(file_path, 'r') as f:
                self.tuning_results = json.load(f)
            
            logger.info("调优结果加载成功")
            return self.tuning_results
            
        except Exception as e:
            logger.error(f"加载调优结果失败：{e}")
            return None
    
    def get_best_params(self, tuning_type=None):
        """
        获取最佳参数
        
        Args:
            tuning_type: 调优类型
            
        Returns:
            最佳参数
        """
        if tuning_type is None:
            # 返回所有调优类型的最佳参数
            return {
                tuning_type: result.get('best_params', result.get('best_threshold'))
                for tuning_type, result in self.tuning_results.items()
            }
        else:
            # 返回指定调优类型的最佳参数
            if tuning_type in self.tuning_results:
                result = self.tuning_results[tuning_type]
                return result.get('best_params', result.get('best_threshold'))
            else:
                logger.warning(f"未找到{result_type}类型的调优结果")
                return None
    
    def analyze_tuning_results(self):
        """
        分析调优结果
        
        Returns:
            分析报告
        """
        logger.info("分析调优结果")
        
        report = {
            'tuning_summary': {},
            'key_insights': []
        }
        
        # 汇总调优结果
        for tuning_type, result in self.tuning_results.items():
            if tuning_type == 'grid_search' or tuning_type == 'random_search' or tuning_type == 'optuna':
                report['tuning_summary'][tuning_type] = {
                    'best_score': result['best_score'],
                    'best_params': result['best_params'],
                    'tuning_method': tuning_type
                }
            elif tuning_type == 'threshold_tuning':
                report['tuning_summary'][tuning_type] = {
                    'best_threshold': result['best_threshold'],
                    'best_metrics': result['best_metrics']
                }
            elif tuning_type == 'sensitivity_analysis':
                report['tuning_summary'][tuning_type] = {
                    'analyzed_params': list(result.keys())
                }
        
        # 生成关键洞察
        if 'threshold_tuning' in self.tuning_results:
            best_threshold = self.tuning_results['threshold_tuning']['best_threshold']
            report['key_insights'].append(f"最佳BUY_THRESHOLD为{best_threshold:.4f}")
        
        logger.info(f"调优结果分析报告：{report}")
        return report
