#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数调优模块
实现自动化参数调优框架，支持多种优化算法
"""

import pandas as pd
import numpy as np
import os
import datetime
import json
from typing import Dict, List, Tuple, Any
from utils.logger import setup_logger
from sklearn.model_selection import ParameterGrid, ParameterSampler
from scipy.optimize import minimize
import concurrent.futures

logger = setup_logger()


class ParameterTuner:
    """参数调优器，用于自动寻找策略最优参数组合"""
    
    def __init__(self, backtest_module, optimization_target: str = 'annual_return'):
        """初始化参数调优器
        
        Args:
            backtest_module: 回测模块对象，用于评估参数表现
            optimization_target: 优化目标，如'annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'等
        """
        logger.info("初始化参数调优器")
        self.backtest = backtest_module
        self.optimization_target = optimization_target
        self.best_params = {}
        self.tuning_history = []
        self.search_space = self._define_search_space()
        
        # 确保结果保存目录存在
        self.result_dir = 'reports/parameter_tuning'
        os.makedirs(self.result_dir, exist_ok=True)
    
    def _define_search_space(self) -> Dict[str, List[Any]]:
        """定义参数搜索空间
        
        Returns:
            参数搜索空间字典，键为参数名，值为候选值列表
        """
        logger.info("定义参数搜索空间")
        
        # 根据不同策略模块定义参数搜索空间
        search_space = {
            # 板块识别模块参数
            'sector_top_n': [3, 5, 7],
            'sector_hotness_threshold': [0.5, 0.6, 0.7],
            
            # 个股筛选模块参数
            'price_bid_change_lower': [0.04, 0.05, 0.06],
            'price_bid_change_upper': [0.09, 0.095, 0.1],
            'volume_bid_volume_ratio': [1.2, 1.5, 2.0],
            'volume_3d_growth_ratio': [0.2, 0.3, 0.4],
            'circulating_market_cap_lower': [5000000000, 10000000000],
            'circulating_market_cap_upper': [15000000000, 20000000000],
            'volume_bid_turnover_rate_lower': [0.005, 0.008],
            'volume_bid_turnover_rate_upper': [0.06, 0.08, 0.1],
            'order_book_bid_order_amount': [5000000, 10000000, 20000000],
            'order_book_bid_order_ratio': [0.12, 0.15, 0.2],
            
            # 模型预测模块参数
            'prediction_confidence_threshold': [0.7, 0.75, 0.8],
            
            # 风险评估模块参数
            'market_risk_weight': [0.2, 0.3, 0.4],
            'volatility_risk_weight': [0.2, 0.3, 0.4],
            'liquidity_risk_weight': [0.1, 0.2, 0.3],
            'policy_risk_weight': [0.1, 0.2, 0.3],
            
            # 决策生成模块参数
            'max_single_position': [0.15, 0.2, 0.25],
            'max_total_position': [0.7, 0.8, 0.9],
            'profit_target': [0.08, 0.1, 0.12],
            'stop_loss_ratio': [0.03, 0.04, 0.05],
            'trailing_stop_ratio': [0.04, 0.05, 0.06]
        }
        
        logger.info(f"参数搜索空间：{search_space}")
        return search_space
    
    def grid_search(self, param_subset: List[str] = None, max_workers: int = 4) -> Dict[str, Any]:
        """网格搜索优化参数
        
        Args:
            param_subset: 要优化的参数子集，默认为所有参数
            max_workers: 并行计算的最大线程数
            
        Returns:
            最优参数组合
        """
        logger.info("开始网格搜索参数优化")
        
        # 确定要优化的参数
        if param_subset is None:
            param_subset = list(self.search_space.keys())
        
        # 构建参数网格
        param_grid = {param: self.search_space[param] for param in param_subset}
        grid = ParameterGrid(param_grid)
        
        logger.info(f"参数网格大小：{len(grid)} 个组合")
        
        # 使用并行计算评估参数组合
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self._evaluate_params, params): params
                for params in grid
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                params = futures[future]
                try:
                    result = future.result()
                    results.append((params, result))
                    logger.info(f"参数组合评估完成：{params}, 结果：{result}")
                except Exception as e:
                    logger.error(f"参数组合 {params} 评估失败：{e}")
        
        # 找到最优参数
        best_params, best_result = self._find_best_params(results)
        
        # 保存调优历史
        self._save_tuning_history(results)
        
        logger.info(f"网格搜索完成，最优参数：{best_params}, 最优结果：{best_result}")
        return best_params
    
    def random_search(self, n_iter: int = 50, param_subset: List[str] = None, max_workers: int = 4) -> Dict[str, Any]:
        """随机搜索优化参数
        
        Args:
            n_iter: 随机采样次数
            param_subset: 要优化的参数子集，默认为所有参数
            max_workers: 并行计算的最大线程数
            
        Returns:
            最优参数组合
        """
        logger.info(f"开始随机搜索参数优化，采样次数：{n_iter}")
        
        # 确定要优化的参数
        if param_subset is None:
            param_subset = list(self.search_space.keys())
        
        # 构建参数空间
        param_space = {param: self.search_space[param] for param in param_subset}
        
        # 随机采样参数组合
        param_samples = list(ParameterSampler(param_space, n_iter=n_iter, random_state=42))
        
        logger.info(f"随机采样参数组合数量：{len(param_samples)}")
        
        # 使用并行计算评估参数组合
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self._evaluate_params, params): params
                for params in param_samples
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                params = futures[future]
                try:
                    result = future.result()
                    results.append((params, result))
                    logger.info(f"参数组合评估完成：{params}, 结果：{result}")
                except Exception as e:
                    logger.error(f"参数组合 {params} 评估失败：{e}")
        
        # 找到最优参数
        best_params, best_result = self._find_best_params(results)
        
        # 保存调优历史
        self._save_tuning_history(results)
        
        logger.info(f"随机搜索完成，最优参数：{best_params}, 最优结果：{best_result}")
        return best_params
    
    def bayesian_optimization(self, n_iter: int = 30, param_subset: List[str] = None) -> Dict[str, Any]:
        """贝叶斯优化参数
        
        Args:
            n_iter: 优化迭代次数
            param_subset: 要优化的参数子集，默认为所有参数
            
        Returns:
            最优参数组合
        """
        logger.info(f"开始贝叶斯优化参数，迭代次数：{n_iter}")
        
        # 确定要优化的参数
        if param_subset is None:
            param_subset = list(self.search_space.keys())
        
        # 贝叶斯优化需要连续参数，这里简化处理
        # 实际实现中应使用GPyOpt或scikit-optimize等库
        logger.warning("贝叶斯优化暂未完全实现，将使用随机搜索替代")
        return self.random_search(n_iter=n_iter, param_subset=param_subset)
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, float]:
        """评估单个参数组合的表现
        
        Args:
            params: 参数组合
            
        Returns:
            回测结果字典
        """
        logger.info(f"评估参数组合：{params}")
        
        try:
            # 使用回测模块评估参数表现
            backtest_results = self.backtest.run_backtest(params)
            
            # 提取关键评估指标
            performance = {
                'annual_return': backtest_results.get('annual_return', 0),
                'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                'max_drawdown': backtest_results.get('max_drawdown', 0),
                'win_rate': backtest_results.get('win_rate', 0),
                'profit_factor': backtest_results.get('profit_factor', 0),
                'total_trades': backtest_results.get('total_trades', 0)
            }
            
            logger.info(f"参数组合评估结果：{performance}")
            return performance
        except Exception as e:
            logger.error(f"评估参数组合 {params} 失败：{e}")
            # 返回最差结果
            return {
                'annual_return': -1,
                'sharpe_ratio': -1,
                'max_drawdown': -1,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0
            }
    
    def _find_best_params(self, results: List[Tuple[Dict[str, Any], Dict[str, float]]]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """从评估结果中找到最优参数组合
        
        Args:
            results: 参数评估结果列表，每个元素为(参数组合, 表现指标)
            
        Returns:
            最优参数组合和对应的表现指标
        """
        logger.info("寻找最优参数组合")
        
        if not results:
            logger.error("没有评估结果，无法找到最优参数")
            return {}, {}
        
        # 根据优化目标选择最优参数
        best_result = None
        best_params = None
        
        for params, performance in results:
            # 初始化为第一个结果
            if best_result is None:
                best_params = params
                best_result = performance
                continue
            
            # 根据不同优化目标选择不同的比较逻辑
            if self.optimization_target == 'annual_return':
                # 年化收益率越高越好
                if performance['annual_return'] > best_result['annual_return']:
                    best_params = params
                    best_result = performance
            elif self.optimization_target == 'sharpe_ratio':
                # 夏普比率越高越好
                if performance['sharpe_ratio'] > best_result['sharpe_ratio']:
                    best_params = params
                    best_result = performance
            elif self.optimization_target == 'max_drawdown':
                # 最大回撤越小越好（绝对值）
                if abs(performance['max_drawdown']) < abs(best_result['max_drawdown']):
                    best_params = params
                    best_result = performance
            elif self.optimization_target == 'win_rate':
                # 胜率越高越好
                if performance['win_rate'] > best_result['win_rate']:
                    best_params = params
                    best_result = performance
            elif self.optimization_target == 'profit_factor':
                # 盈亏比越高越好
                if performance['profit_factor'] > best_result['profit_factor']:
                    best_params = params
                    best_result = performance
        
        # 保存最佳参数
        self.best_params = best_params
        
        logger.info(f"最优参数组合：{best_params}")
        logger.info(f"最优表现指标：{best_result}")
        
        return best_params, best_result
    
    def _save_tuning_history(self, results: List[Tuple[Dict[str, Any], Dict[str, float]]]):
        """保存参数调优历史
        
        Args:
            results: 参数评估结果列表
        """
        logger.info("保存参数调优历史")
        
        # 转换结果格式便于保存
        history = []
        for params, performance in results:
            history.append({
                'params': params,
                'performance': performance,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # 更新调优历史
        self.tuning_history.extend(history)
        
        # 保存到文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(self.result_dir, f"tuning_history_{timestamp}.json")
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        logger.info(f"参数调优历史已保存到：{history_file}")
        
        # 保存最佳参数
        best_params_file = os.path.join(self.result_dir, f"best_params_{timestamp}.json")
        with open(best_params_file, 'w', encoding='utf-8') as f:
            json.dump({
                'best_params': self.best_params,
                'optimization_target': self.optimization_target,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"最佳参数已保存到：{best_params_file}")
    
    def visualize_tuning_results(self):
        """可视化参数调优结果"""
        logger.info("可视化参数调优结果")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 转换调优历史为DataFrame便于可视化
            if not self.tuning_history:
                logger.warning("没有调优历史，无法可视化")
                return
            
            # 提取所有参数和性能指标
            param_names = list(self.search_space.keys())
            performance_metrics = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
            
            # 创建可视化结果
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 各参数对优化目标的影响热力图
            for metric in performance_metrics:
                plt.figure(figsize=(12, 8))
                
                # 提取参数和对应指标值
                param_values = []
                metric_values = []
                
                for entry in self.tuning_history:
                    params = entry['params']
                    performance = entry['performance']
                    
                    # 只考虑完整的参数组合
                    if set(params.keys()) == set(param_names):
                        param_values.append(params)
                        metric_values.append(performance[metric])
                
                if not param_values:
                    continue
                
                # 转换为DataFrame
                df = pd.DataFrame(param_values)
                df[metric] = metric_values
                
                # 计算相关性矩阵
                corr_matrix = df.corr()
                
                # 绘制热力图
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
                plt.title(f'{metric}与各参数的相关性热力图')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                
                # 保存图表
                plt.savefig(os.path.join(self.result_dir, f"{metric}_correlation_heatmap_{timestamp}.png"), dpi=300, bbox_inches="tight")
                plt.close()
            
            # 2. 优化目标随迭代的变化
            plt.figure(figsize=(12, 6))
            
            # 提取优化目标值
            target_values = [entry['performance'][self.optimization_target] for entry in self.tuning_history]
            iterations = range(1, len(target_values) + 1)
            
            # 绘制优化曲线
            plt.plot(iterations, target_values, marker='o', linestyle='-', color='b')
            plt.xlabel('迭代次数')
            plt.ylabel(self.optimization_target)
            plt.title(f'{self.optimization_target}随迭代次数的变化')
            plt.grid(True)
            
            # 保存图表
            plt.savefig(os.path.join(self.result_dir, f"optimization_curve_{timestamp}.png"), dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info("参数调优结果可视化完成")
            
        except ImportError as e:
            logger.warning(f"无法生成可视化：缺少依赖包：{e}")
        except Exception as e:
            logger.error(f"生成可视化失败：{e}")
    
    def load_best_params(self, file_path: str = None) -> Dict[str, Any]:
        """加载最佳参数
        
        Args:
            file_path: 最佳参数文件路径
            
        Returns:
            最佳参数字典
        """
        logger.info("加载最佳参数")
        
        if file_path is None:
            # 查找最新的最佳参数文件
            files = [f for f in os.listdir(self.result_dir) if f.startswith('best_params_')]
            if not files:
                logger.warning("没有找到最佳参数文件")
                return {}
            
            # 按时间戳排序，取最新的
            files.sort(reverse=True)
            file_path = os.path.join(self.result_dir, files[0])
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.best_params = data['best_params']
                logger.info(f"从文件加载最佳参数：{file_path}")
                logger.info(f"最佳参数：{self.best_params}")
                return self.best_params
        except Exception as e:
            logger.error(f"加载最佳参数失败：{e}")
            return {}
    
    def get_best_params(self) -> Dict[str, Any]:
        """获取最佳参数
        
        Returns:
            最佳参数字典
        """
        return self.best_params
    
    def get_tuning_history(self) -> List[Dict[str, Any]]:
        """获取调优历史
        
        Returns:
            调优历史列表
        """
        return self.tuning_history


if __name__ == "__main__":
    # 测试参数调优器
    logger.info("测试参数调优器")
    
    # 模拟回测模块
    class MockBacktest:
        def run_backtest(self, params):
            return {
                'annual_return': np.random.rand() * 0.5 + 0.1,
                'sharpe_ratio': np.random.rand() * 2 + 1,
                'max_drawdown': -np.random.rand() * 0.2,
                'win_rate': np.random.rand() * 0.3 + 0.5,
                'profit_factor': np.random.rand() * 1 + 1.5,
                'total_trades': np.random.randint(50, 200)
            }
    
    backtest = MockBacktest()
    tuner = ParameterTuner(backtest)
    
    # 运行网格搜索
    best_params = tuner.grid_search(param_subset=['sector_top_n', 'sector_hotness_threshold'], max_workers=2)
    logger.info(f"网格搜索最佳参数：{best_params}")
    
    # 运行随机搜索
    best_params = tuner.random_search(n_iter=10, param_subset=['price_bid_change_lower', 'price_bid_change_upper'])
    logger.info(f"随机搜索最佳参数：{best_params}")
    
    # 可视化结果
    tuner.visualize_tuning_results()
    
    logger.info("参数调优器测试完成")
