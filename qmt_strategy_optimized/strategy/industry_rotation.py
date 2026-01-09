#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
行业轮动策略模块
负责实现行业轮动策略，包括行业景气度评估、资金流动监测和权重动态调整
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = setup_logger()

class IndustryRotation:
    """行业轮动策略类"""
    
    def __init__(self):
        """初始化行业轮动策略"""
        logger.info("初始化行业轮动策略")
        self.industry_scores = None
        self.industry_weights = None
        self.scaler = MinMaxScaler()
    
    def evaluate_industry_boom(self, industry_data):
        """
        评估行业景气度
        
        Args:
            industry_data: 行业数据，包含财务指标、估值指标等
            
        Returns:
            行业景气度评分
        """
        logger.info("评估行业景气度")
        
        try:
            if industry_data is None or industry_data.empty:
                logger.warning("行业数据为空，无法评估景气度")
                return None
            
            # 计算行业景气度评分
            boom_factors = []
            
            # 1. 营收增长率
            if 'revenue_growth' in industry_data.columns:
                boom_factors.append('revenue_growth')
            
            # 2. 净利润增长率
            if 'net_profit_growth' in industry_data.columns:
                boom_factors.append('net_profit_growth')
            
            # 3. 毛利率
            if 'gross_margin' in industry_data.columns:
                boom_factors.append('gross_margin')
            
            # 4. ROE
            if 'roe' in industry_data.columns:
                boom_factors.append('roe')
            
            # 5. 行业资金流入
            if 'capital_flow' in industry_data.columns:
                boom_factors.append('capital_flow')
            
            # 6. 行业成交量变化率
            if 'volume_change_rate' in industry_data.columns:
                boom_factors.append('volume_change_rate')
            
            # 7. 行业涨跌幅
            if 'industry_change' in industry_data.columns:
                boom_factors.append('industry_change')
            
            # 8. 行业换手率
            if 'turnover_rate' in industry_data.columns:
                boom_factors.append('turnover_rate')
            
            # 9. 估值指标（市盈率相对行业历史水平）
            if 'pe_ratio' in industry_data.columns and 'industry_pe_ratio' in industry_data.columns:
                industry_data['pe_relative'] = industry_data['pe_ratio'] / industry_data['industry_pe_ratio']
                boom_factors.append('pe_relative')
            
            if not boom_factors:
                logger.warning("没有找到合适的景气度因子")
                return None
            
            # 标准化处理
            scaled_data = self.scaler.fit_transform(industry_data[boom_factors])
            scaled_df = pd.DataFrame(scaled_data, index=industry_data.index, columns=boom_factors)
            
            # 计算景气度评分（加权平均）
            # 根据因子重要性设置权重
            weights = {
                'revenue_growth': 0.15,
                'net_profit_growth': 0.15,
                'gross_margin': 0.1,
                'roe': 0.1,
                'capital_flow': 0.15,
                'volume_change_rate': 0.1,
                'industry_change': 0.1,
                'turnover_rate': 0.05,
                'pe_relative': 0.1
            }
            
            # 只保留存在的因子权重
            used_weights = {factor: weights[factor] for factor in boom_factors if factor in weights}
            
            # 归一化权重
            total_weight = sum(used_weights.values())
            normalized_weights = {factor: weight / total_weight for factor, weight in used_weights.items()}
            
            # 计算加权评分
            industry_data['boom_score'] = 0
            for factor, weight in normalized_weights.items():
                industry_data['boom_score'] += scaled_df[factor] * weight
            
            # 行业分类
            industry_data['boom_level'] = pd.qcut(industry_data['boom_score'], 5, labels=[1, 2, 3, 4, 5])
            
            self.industry_scores = industry_data[['boom_score', 'boom_level'] + boom_factors]
            
            logger.info("行业景气度评估完成")
            return self.industry_scores
            
        except Exception as e:
            logger.error(f"行业景气度评估失败：{e}")
            return None
    
    def monitor_industry_capital_flow(self, capital_flow_data, window=5):
        """
        监测行业资金流动
        
        Args:
            capital_flow_data: 资金流动数据
            window: 时间窗口
            
        Returns:
            资金流动监测结果
        """
        logger.info("监测行业资金流动")
        
        try:
            if capital_flow_data is None or capital_flow_data.empty:
                logger.warning("资金流动数据为空，无法监测")
                return None
            
            # 计算资金流动指标
            capital_flow_metrics = pd.DataFrame(index=capital_flow_data.index)
            
            # 1. 当日资金流入
            if 'capital_flow' in capital_flow_data.columns:
                capital_flow_metrics['daily_flow'] = capital_flow_data['capital_flow']
            
            # 2. 5日累计资金流入
            if 'capital_flow' in capital_flow_data.columns:
                capital_flow_metrics['5d_flow'] = capital_flow_data['capital_flow'].rolling(window=window).sum()
            
            # 3. 资金流入变化率
            if 'capital_flow' in capital_flow_data.columns:
                capital_flow_metrics['flow_change_rate'] = capital_flow_data['capital_flow'].pct_change()
            
            # 4. 主力资金流入
            if 'main_capital_flow' in capital_flow_data.columns:
                capital_flow_metrics['main_flow'] = capital_flow_data['main_capital_flow']
            
            # 5. 散户资金流入
            if 'retail_capital_flow' in capital_flow_data.columns:
                capital_flow_metrics['retail_flow'] = capital_flow_data['retail_capital_flow']
            
            # 6. 资金流入占比
            if 'capital_flow' in capital_flow_data.columns and 'market_cap' in capital_flow_data.columns:
                capital_flow_metrics['flow_ratio'] = capital_flow_data['capital_flow'] / capital_flow_data['market_cap']
            
            # 7. 资金流入排名
            if 'capital_flow' in capital_flow_data.columns:
                capital_flow_metrics['flow_rank'] = capital_flow_data['capital_flow'].rank(ascending=False)
            
            # 8. 资金流入动量
            if 'capital_flow' in capital_flow_data.columns:
                capital_flow_metrics['flow_momentum'] = capital_flow_data['capital_flow'].rolling(window=window).mean()
            
            logger.info("行业资金流动监测完成")
            return capital_flow_metrics
            
        except Exception as e:
            logger.error(f"行业资金流动监测失败：{e}")
            return None
    
    def adjust_industry_weights(self, industry_scores, capital_flow_metrics, strategy='momentum', top_n=5):
        """
        动态调整行业权重
        
        Args:
            industry_scores: 行业景气度评分
            capital_flow_metrics: 资金流动指标
            strategy: 权重调整策略，可选值：momentum, mean_reversion, balanced
            top_n: 选择前n个行业
            
        Returns:
            行业权重
        """
        logger.info(f"使用{strategy}策略动态调整行业权重")
        
        try:
            # 合并数据
            if industry_scores is not None and capital_flow_metrics is not None:
                combined_data = pd.concat([industry_scores, capital_flow_metrics], axis=1)
            elif industry_scores is not None:
                combined_data = industry_scores.copy()
            elif capital_flow_metrics is not None:
                combined_data = capital_flow_metrics.copy()
            else:
                logger.warning("没有可用的数据进行权重调整")
                return None
            
            # 计算综合评分
            combined_data['composite_score'] = 0
            
            # 1. 景气度评分权重
            if 'boom_score' in combined_data.columns:
                combined_data['composite_score'] += combined_data['boom_score'] * 0.5
            
            # 2. 资金流入权重
            if '5d_flow' in combined_data.columns:
                combined_data['composite_score'] += combined_data['5d_flow'] * 0.3
            
            # 3. 涨跌幅权重
            if 'industry_change' in combined_data.columns:
                combined_data['composite_score'] += combined_data['industry_change'] * 0.2
            
            # 根据策略调整权重
            if strategy == 'momentum':
                # 动量策略：选择景气度高、资金流入多的行业
                selected_industries = combined_data.sort_values(by='composite_score', ascending=False).head(top_n)
            elif strategy == 'mean_reversion':
                # 反转策略：选择景气度低但有改善迹象的行业
                combined_data['reversion_score'] = -combined_data['composite_score']
                selected_industries = combined_data.sort_values(by='reversion_score', ascending=False).head(top_n)
            elif strategy == 'balanced':
                # 平衡策略：兼顾不同景气度的行业
                # 使用K-Means聚类
                kmeans = KMeans(n_clusters=3, random_state=42)
                combined_data['cluster'] = kmeans.fit_predict(combined_data[['boom_score', '5d_flow']])
                
                # 从每个聚类中选择前top_n/3个行业
                selected_industries = pd.DataFrame()
                for cluster in combined_data['cluster'].unique():
                    cluster_industries = combined_data[combined_data['cluster'] == cluster]
                    selected = cluster_industries.sort_values(by='composite_score', ascending=False).head(top_n//3 + 1)
                    selected_industries = pd.concat([selected_industries, selected])
                
                # 限制总数量
                selected_industries = selected_industries.sort_values(by='composite_score', ascending=False).head(top_n)
            else:
                logger.warning(f"未知的权重调整策略：{strategy}，使用默认的momentum策略")
                selected_industries = combined_data.sort_values(by='composite_score', ascending=False).head(top_n)
            
            # 计算权重
            total_score = selected_industries['composite_score'].sum()
            selected_industries['weight'] = selected_industries['composite_score'] / total_score
            
            # 处理可能的NaN值
            selected_industries['weight'] = selected_industries['weight'].fillna(1/top_n)
            
            # 确保权重和为1
            selected_industries['weight'] = selected_industries['weight'] / selected_industries['weight'].sum()
            
            self.industry_weights = selected_industries[['weight', 'composite_score']]
            
            logger.info(f"行业权重调整完成，选择了{len(selected_industries)}个行业")
            logger.info(f"行业权重分布：{self.industry_weights['weight'].to_dict()}")
            
            return self.industry_weights
            
        except Exception as e:
            logger.error(f"行业权重调整失败：{e}")
            return None
    
    def generate_rotation_signals(self, industry_weights, threshold=0.1):
        """
        生成行业轮动信号
        
        Args:
            industry_weights: 行业权重
            threshold: 权重变化阈值
            
        Returns:
            行业轮动信号
        """
        logger.info("生成行业轮动信号")
        
        try:
            if industry_weights is None or industry_weights.empty:
                logger.warning("行业权重为空，无法生成轮动信号")
                return None
            
            # 生成轮动信号
            rotation_signals = pd.DataFrame(index=industry_weights.index)
            
            # 1. 买入信号：权重高于平均权重
            avg_weight = 1 / len(industry_weights)
            rotation_signals['buy_signal'] = (industry_weights['weight'] > avg_weight + threshold).astype(int)
            
            # 2. 卖出信号：权重低于平均权重
            rotation_signals['sell_signal'] = (industry_weights['weight'] < avg_weight - threshold).astype(int)
            
            # 3. 持有信号：权重在平均权重附近
            rotation_signals['hold_signal'] = ((industry_weights['weight'] >= avg_weight - threshold) & 
                                             (industry_weights['weight'] <= avg_weight + threshold)).astype(int)
            
            # 4. 信号强度
            rotation_signals['signal_strength'] = industry_weights['weight'] - avg_weight
            
            logger.info("行业轮动信号生成完成")
            return rotation_signals
            
        except Exception as e:
            logger.error(f"行业轮动信号生成失败：{e}")
            return None
    
    def combine_with_stock_selection(self, rotation_signals, stock_predictions):
        """
        将行业轮动信号与个股选择结合
        
        Args:
            rotation_signals: 行业轮动信号
            stock_predictions: 个股预测结果
            
        Returns:
            结合后的投资组合
        """
        logger.info("将行业轮动信号与个股选择结合")
        
        try:
            if rotation_signals is None or rotation_signals.empty or stock_predictions is None or stock_predictions.empty:
                logger.warning("输入数据为空，无法结合行业轮动与个股选择")
                return None
            
            # 假设stock_predictions包含industry列
            if 'industry' not in stock_predictions.columns:
                logger.warning("个股预测结果中没有行业信息，无法结合")
                return None
            
            # 合并数据
            combined = stock_predictions.merge(
                rotation_signals, 
                left_on='industry', 
                right_index=True, 
                how='left'
            )
            
            # 计算最终评分
            combined['final_score'] = 0
            
            # 1. 个股预测分数权重
            if 'prediction_prob' in combined.columns:
                combined['final_score'] += combined['prediction_prob'] * 0.6
            
            # 2. 行业轮动信号权重
            if 'signal_strength' in combined.columns:
                combined['final_score'] += combined['signal_strength'] * 0.4
            
            # 3. 买入信号增强
            if 'buy_signal' in combined.columns:
                combined['final_score'] += combined['buy_signal'] * 0.1
            
            # 4. 卖出信号减弱
            if 'sell_signal' in combined.columns:
                combined['final_score'] -= combined['sell_signal'] * 0.1
            
            # 过滤掉卖出信号强的股票
            combined = combined[combined['sell_signal'] != 1]
            
            logger.info("行业轮动与个股选择结合完成")
            return combined.sort_values(by='final_score', ascending=False)
            
        except Exception as e:
            logger.error(f"结合行业轮动与个股选择失败：{e}")
            return None
    
    def backtest_industry_rotation(self, industry_data, strategy_params=None):
        """
        回测行业轮动策略
        
        Args:
            industry_data: 行业历史数据
            strategy_params: 策略参数
            
        Returns:
            回测结果
        """
        logger.info("回测行业轮动策略")
        
        try:
            if industry_data is None or industry_data.empty:
                logger.warning("行业历史数据为空，无法回测")
                return None
            
            # 默认参数
            default_params = {
                'top_n': 5,
                'rebalance_period': 20,  # 20个交易日调整一次
                'strategy': 'momentum'
            }
            
            params = strategy_params or default_params
            
            # 模拟回测
            results = []
            dates = industry_data.index.unique()
            
            for i in range(0, len(dates), params['rebalance_period']):
                # 选择回测日期
                date = dates[i]
                
                # 获取当日行业数据
                daily_data = industry_data.loc[date]
                
                # 评估景气度
                boom_scores = self.evaluate_industry_boom(daily_data)
                
                # 调整权重
                industry_weights = self.adjust_industry_weights(
                    boom_scores, 
                    None, 
                    strategy=params['strategy'],
                    top_n=params['top_n']
                )
                
                if industry_weights is not None:
                    # 计算下一期收益
                    if i + params['rebalance_period'] < len(dates):
                        next_date = dates[i + params['rebalance_period']]
                        next_data = industry_data.loc[next_date]
                        
                        # 计算行业收益
                        industry_returns = next_data['industry_change']
                        
                        # 计算组合收益
                        portfolio_return = 0
                        for industry in industry_weights.index:
                            if industry in industry_returns.index:
                                portfolio_return += industry_weights.loc[industry, 'weight'] * industry_returns.loc[industry]
                        
                        results.append({
                            'date': date,
                            'portfolio_return': portfolio_return,
                            'selected_industries': list(industry_weights.index),
                            'weights': industry_weights['weight'].to_dict()
                        })
            
            # 计算回测指标
            if results:
                results_df = pd.DataFrame(results)
                
                # 计算累计收益
                results_df['cumulative_return'] = (1 + results_df['portfolio_return']).cumprod() - 1
                
                # 计算年化收益
                total_days = (dates[-1] - dates[0]).days
                annual_return = (results_df['cumulative_return'].iloc[-1] + 1) ** (365 / total_days) - 1
                
                # 计算夏普比率（假设无风险利率为3%）
                risk_free_rate = 0.03
                excess_returns = results_df['portfolio_return'] - risk_free_rate / 252
                sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                
                # 计算最大回撤
                cumulative_returns = results_df['cumulative_return']
                max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
                
                backtest_results = {
                    'annual_return': annual_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'cumulative_return': results_df['cumulative_return'].iloc[-1],
                    'daily_returns': results_df['portfolio_return'],
                    'full_results': results_df
                }
                
                logger.info(f"行业轮动策略回测完成")
                logger.info(f"年化收益：{annual_return:.4f}")
                logger.info(f"夏普比率：{sharpe_ratio:.4f}")
                logger.info(f"最大回撤：{max_drawdown:.4f}")
                logger.info(f"累计收益：{results_df['cumulative_return'].iloc[-1]:.4f}")
                
                return backtest_results
            else:
                logger.warning("没有回测结果")
                return None
                
        except Exception as e:
            logger.error(f"行业轮动策略回测失败：{e}")
            return None
    
    def optimize_rotation_strategy(self, industry_data, param_grid):
        """
        优化行业轮动策略参数
        
        Args:
            industry_data: 行业数据
            param_grid: 参数网格
            
        Returns:
            最优参数
        """
        logger.info("优化行业轮动策略参数")
        
        try:
            best_score = -np.inf
            best_params = None
            
            # 遍历参数组合
            for top_n in param_grid.get('top_n', [5]):
                for strategy in param_grid.get('strategy', ['momentum']):
                    for rebalance_period in param_grid.get('rebalance_period', [20]):
                        
                        params = {
                            'top_n': top_n,
                            'strategy': strategy,
                            'rebalance_period': rebalance_period
                        }
                        
                        # 回测
                        backtest_result = self.backtest_industry_rotation(industry_data, params)
                        
                        if backtest_result:
                            # 使用夏普比率作为优化目标
                            if backtest_result['sharpe_ratio'] > best_score:
                                best_score = backtest_result['sharpe_ratio']
                                best_params = params
            
            logger.info(f"行业轮动策略参数优化完成")
            logger.info(f"最优参数：{best_params}")
            logger.info(f"最优夏普比率：{best_score:.4f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"行业轮动策略参数优化失败：{e}")
            return None
    
    def get_industry_performance(self, industry_data, period=20):
        """
        获取行业表现
        
        Args:
            industry_data: 行业数据
            period: 时间周期
            
        Returns:
            行业表现数据
        """
        logger.info(f"获取行业近{period}天表现")
        
        try:
            if industry_data is None or industry_data.empty:
                logger.warning("行业数据为空，无法获取表现")
                return None
            
            # 计算行业表现指标
            industry_performance = pd.DataFrame(index=industry_data.index.unique())
            
            # 1. 平均涨跌幅
            industry_performance['avg_return'] = industry_data.groupby(level=0)['industry_change'].mean()
            
            # 2. 最大涨幅
            industry_performance['max_return'] = industry_data.groupby(level=0)['industry_change'].max()
            
            # 3. 最小涨幅
            industry_performance['min_return'] = industry_data.groupby(level=0)['industry_change'].min()
            
            # 4. 胜率
            industry_performance['win_rate'] = industry_data[industry_data['industry_change'] > 0].groupby(level=0).size() / \
                                              industry_data.groupby(level=0).size()
            
            # 5. 平均资金流入
            if 'capital_flow' in industry_data.columns:
                industry_performance['avg_capital_flow'] = industry_data.groupby(level=0)['capital_flow'].mean()
            
            logger.info("行业表现获取完成")
            return industry_performance
            
        except Exception as e:
            logger.error(f"获取行业表现失败：{e}")
            return None
