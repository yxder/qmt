#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征工程模块
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger
import config
from utils.tools import normalize_data, standardize_data

logger = setup_logger()

class FeatureEngineer:
    """特征工程类，用于提取和计算量化特征"""
    
    def __init__(self):
        """初始化特征工程师"""
        logger.info("初始化特征工程师")
    
    def extract_features_simple(self, data):
        """简化的特征提取方法，用于回测，跳过复杂的特征选择和重要性评估"""
        logger.info("开始简化特征提取")
        
        if data is None:
            logger.warning("输入数据为空，跳过特征提取")
            return None
        
        try:
            # 提取集合竞价特征
            bid_features = self._extract_bid_features(data)
            
            # 提取板块热度特征
            sector_features = self._extract_sector_features(data)
            
            # 提取市场情绪特征
            market_features = self._extract_market_features(data)
            
            # 提取个股基本面特征
            fundamental_features = self._extract_fundamental_features(data)
            
            # 提取技术指标特征
            technical_features = self._extract_technical_features(data)
            
            # 合并所有特征
            all_features = self._merge_features([
                bid_features,
                sector_features,
                market_features,
                fundamental_features,
                technical_features
            ])
            
            logger.info("简化特征提取完成")
            return all_features
            
        except Exception as e:
            logger.error(f"简化特征提取失败：{e}")
            return None
    
    def extract_features(self, data):
        """提取特征"""
        logger.info("开始特征提取")
        
        if data is None:
            logger.warning("输入数据为空，跳过特征提取")
            return None
        
        try:
            # 提取集合竞价特征
            bid_features = self._extract_bid_features(data)
            
            # 提取板块热度特征
            sector_features = self._extract_sector_features(data)
            
            # 提取市场情绪特征
            market_features = self._extract_market_features(data)
            
            # 提取个股基本面特征
            fundamental_features = self._extract_fundamental_features(data)
            
            # 提取技术指标特征
            technical_features = self._extract_technical_features(data)
            
            # 提取label列
            label_data = pd.DataFrame()
            if isinstance(data, pd.DataFrame) and 'label' in data.columns:
                label_data['label'] = data['label']
            
            # 合并所有特征
            all_features = self._merge_features([
                bid_features,
                sector_features,
                market_features,
                fundamental_features,
                technical_features,
                label_data
            ])
            
            # 特征选择和重要性评估
            selected_features = self._select_features(all_features)
            
            # 保存特征数据
            self._save_feature_data(selected_features)
            
            logger.info("特征提取完成")
            return selected_features
            
        except Exception as e:
            logger.error(f"特征提取失败：{e}")
            return None
    
    def _extract_bid_features(self, data):
        """提取集合竞价特征"""
        logger.info("提取集合竞价特征")
        
        bid_features = pd.DataFrame(index=data.index) if isinstance(data, pd.DataFrame) else pd.DataFrame()
        
        if isinstance(data, pd.DataFrame):
            # 竞价价格走势相关特征
            if 'bid_price_915' in data.columns and 'bid_price_925' in data.columns:
                # 竞价期间价格涨幅
                bid_features['bid_price_change'] = (data['bid_price_925'] - data['bid_price_915']) / data['bid_price_915']
                # 竞价价格波动率
                bid_features['bid_price_volatility'] = (data['bid_price_925'] - data['bid_price_915']) / data['bid_price_915'] * 100
            else:
                bid_features['bid_price_change'] = 0.0
                bid_features['bid_price_volatility'] = 0.0
            
            # 9:20后不可撤单阶段的关键特征
            if 'bid_price_920' in data.columns and 'bid_price_925' in data.columns:
                # 9:20-9:25价格涨幅
                bid_features['bid_price_change_920_925'] = (data['bid_price_925'] - data['bid_price_920']) / data['bid_price_920']
                # 9:20-9:25价格变化绝对值
                bid_features['bid_price_change_abs_920_925'] = abs(data['bid_price_925'] - data['bid_price_920']) / data['bid_price_920']
            else:
                bid_features['bid_price_change_920_925'] = 0.0
                bid_features['bid_price_change_abs_920_925'] = 0.0
            
            # 9:15-9:20可撤单阶段特征
            if 'bid_price_915' in data.columns and 'bid_price_920' in data.columns:
                # 9:15-9:20价格涨幅
                bid_features['bid_price_change_915_920'] = (data['bid_price_920'] - data['bid_price_915']) / data['bid_price_915']
            else:
                bid_features['bid_price_change_915_920'] = 0.0
            
            # 成交量变化特征
            if 'bid_volume_915' in data.columns and 'bid_volume_925' in data.columns:
                # 竞价期间成交量变化率
                bid_features['bid_volume_change'] = (data['bid_volume_925'] - data['bid_volume_915']) / data['bid_volume_915']
                # 竞价期间成交量绝对值变化
                bid_features['bid_volume_abs_change'] = data['bid_volume_925'] - data['bid_volume_915']
            else:
                bid_features['bid_volume_change'] = 0.0
                bid_features['bid_volume_abs_change'] = 0.0
            
            # 9:20后不可撤单阶段的成交量变化
            if 'bid_volume_920' in data.columns and 'bid_volume_925' in data.columns:
                bid_features['bid_volume_change_920_925'] = (data['bid_volume_925'] - data['bid_volume_920']) / data['bid_volume_920']
                bid_features['bid_volume_abs_change_920_925'] = data['bid_volume_925'] - data['bid_volume_920']
                # 9:20后成交量占总竞价成交量比例
                bid_features['bid_volume_ratio_920_925'] = data['bid_volume_925'] / (data['bid_volume_925'] + data['bid_volume_920']) if 'bid_volume_920' in data.columns else 1.0
                # 处理可能的除零情况
                bid_features['bid_volume_ratio_920_925'] = bid_features['bid_volume_ratio_920_925'].fillna(0.0)
            else:
                bid_features['bid_volume_change_920_925'] = 0.0
                bid_features['bid_volume_abs_change_920_925'] = 0.0
                bid_features['bid_volume_ratio_920_925'] = 0.0
            
            # 委托单量比
            if 'bid_volume_925' in data.columns and 'avg_volume_5d' in data.columns:
                bid_features['bid_volume_ratio'] = data['bid_volume_925'] / data['avg_volume_5d']
                # 委托单量与昨日成交量比
                if 'volume' in data.columns and data['volume'].shift(1).sum() > 0:
                    bid_features['bid_volume_yesterday_ratio'] = data['bid_volume_925'] / data['volume'].shift(1)
                else:
                    bid_features['bid_volume_yesterday_ratio'] = 0.0
            else:
                bid_features['bid_volume_ratio'] = 0.0
                bid_features['bid_volume_yesterday_ratio'] = 0.0
            
            # 竞价强度
            if 'bid_price_925' in data.columns and 'prev_close' in data.columns:
                bid_features['bid_intensity'] = (data['bid_price_925'] - data['prev_close']) / data['prev_close']
                # 竞价强度归一化
                bid_features['bid_intensity_normalized'] = (data['bid_price_925'] - data['prev_close']) / (data['prev_close'] * 0.1)  # 归一化到[-1, 1]附近
            else:
                bid_features['bid_intensity'] = 0.0
                bid_features['bid_intensity_normalized'] = 0.0
            
            # 封单量特征
            if 'buy_order_size_925' in data.columns:
                bid_features['buy_order_size_925'] = data['buy_order_size_925']
                # 封单量变化率（如果有前一天的数据）
                if 'buy_order_size_925_prev' in data.columns:
                    bid_features['buy_order_size_change'] = (data['buy_order_size_925'] - data['buy_order_size_925_prev']) / data['buy_order_size_925_prev']
                else:
                    bid_features['buy_order_size_change'] = 0.0
            else:
                bid_features['buy_order_size_925'] = 0.0
                bid_features['buy_order_size_change'] = 0.0
            
            # 封单量占流通盘比例
            if 'buy_order_size_925' in data.columns and 'circulating_cap' in data.columns:
                bid_features['buy_order_size_ratio'] = data['buy_order_size_925'] / data['circulating_cap']
                # 封单量占昨日成交量比例
                if 'volume' in data.columns and data['volume'].shift(1).sum() > 0:
                    bid_features['buy_order_size_volume_ratio'] = data['buy_order_size_925'] / data['volume'].shift(1)
                else:
                    bid_features['buy_order_size_volume_ratio'] = 0.0
            else:
                bid_features['buy_order_size_ratio'] = 0.0
                bid_features['buy_order_size_volume_ratio'] = 0.0
            
            # 新增：竞价量价配合特征
            if 'bid_price_change' in bid_features.columns and 'bid_volume_change' in bid_features.columns:
                # 量价配合度：价格上涨同时成交量放大为正，否则为负
                bid_features['price_volume_fit'] = bid_features['bid_price_change'] * bid_features['bid_volume_change']
            
            # 新增：开盘价与竞价价关系
            if 'open' in data.columns and 'bid_price_925' in data.columns:
                bid_features['open_bid_price_diff'] = (data['open'] - data['bid_price_925']) / data['bid_price_925']
            else:
                bid_features['open_bid_price_diff'] = 0.0
            
            # 新增：竞价换手率
            if 'bid_volume_925' in data.columns and 'circulating_cap' in data.columns:
                bid_features['bid_turnover_rate'] = data['bid_volume_925'] / data['circulating_cap'] * 100
            else:
                bid_features['bid_turnover_rate'] = 0.0
            
            # 新增：竞价金额特征
            if 'bid_price_925' in data.columns and 'bid_volume_925' in data.columns:
                bid_features['bid_amount_925'] = data['bid_price_925'] * data['bid_volume_925']
            
            # 新增：价格稳定性特征
            if 'bid_price_915' in data.columns and 'bid_price_920' in data.columns and 'bid_price_925' in data.columns:
                # 计算竞价期间价格的标准差（衡量稳定性）
                bid_prices = pd.concat([data['bid_price_915'], data['bid_price_920'], data['bid_price_925']], axis=1)
                bid_features['bid_price_std'] = bid_prices.std(axis=1)
                # 价格稳定性指标（标准差/平均价格）
                bid_features['bid_price_stability'] = bid_features['bid_price_std'] / bid_prices.mean(axis=1)
            
            # 新增：9:15-9:20价格波动幅度
            if 'bid_price_915' in data.columns and 'bid_price_920' in data.columns:
                bid_features['bid_price_volatility_915_920'] = abs(data['bid_price_920'] - data['bid_price_915']) / data['bid_price_915'] * 100
            else:
                bid_features['bid_price_volatility_915_920'] = 0.0
            
            # 新增：9:20-9:25价格波动幅度
            if 'bid_price_920' in data.columns and 'bid_price_925' in data.columns:
                bid_features['bid_price_volatility_920_925'] = abs(data['bid_price_925'] - data['bid_price_920']) / data['bid_price_920'] * 100
            else:
                bid_features['bid_price_volatility_920_925'] = 0.0
            
            # 新增：竞价总量
            if 'bid_volume_915' in data.columns and 'bid_volume_925' in data.columns:
                bid_features['total_bid_volume'] = data['bid_volume_925'] + data['bid_volume_915']
            else:
                bid_features['total_bid_volume'] = 0.0
            
            # 新增：竞价相对前收盘价的涨跌幅
            if 'bid_price_925' in data.columns and 'prev_close' in data.columns:
                bid_features['bid_price_to_prev_close'] = (data['bid_price_925'] - data['prev_close']) / data['prev_close']
            else:
                bid_features['bid_price_to_prev_close'] = 0.0
            
            # 新增：9:20-9:25阶段的量价配合
            if 'bid_price_change_920_925' in bid_features.columns and 'bid_volume_change_920_925' in bid_features.columns:
                bid_features['price_volume_fit_920_925'] = bid_features['bid_price_change_920_925'] * bid_features['bid_volume_change_920_925']
            
            # 新增：竞价成交量占当日总成交量比例（预估）
            if 'bid_volume_925' in data.columns and 'volume' in data.columns:
                bid_features['bid_volume_to_day_ratio'] = data['bid_volume_925'] / (data['volume'].shift(-1) + 1e-6)
            else:
                bid_features['bid_volume_to_day_ratio'] = 0.0
            
            # 新增：多日竞价数据趋势特征
            if 'bid_intensity' in bid_features.columns:
                # 竞价强度5日趋势
                bid_features['bid_intensity_trend_5d'] = bid_features['bid_intensity'].rolling(5).mean()
                # 竞价强度5日变化率
                bid_features['bid_intensity_change_5d'] = bid_features['bid_intensity'].pct_change(5).fillna(0)
                # 竞价强度10日趋势
                bid_features['bid_intensity_trend_10d'] = bid_features['bid_intensity'].rolling(10).mean()
            
            if 'bid_volume_ratio' in bid_features.columns:
                # 竞价量比5日趋势
                bid_features['bid_volume_ratio_trend_5d'] = bid_features['bid_volume_ratio'].rolling(5).mean()
                # 竞价量比5日变化率
                bid_features['bid_volume_ratio_change_5d'] = bid_features['bid_volume_ratio'].pct_change(5).fillna(0)
            
            if 'bid_price_change' in bid_features.columns:
                # 竞价价格变化5日趋势
                bid_features['bid_price_change_trend_5d'] = bid_features['bid_price_change'].rolling(5).mean()
                # 竞价价格变化5日标准差
                bid_features['bid_price_change_std_5d'] = bid_features['bid_price_change'].rolling(5).std()
        
        return bid_features
    
    def _extract_sector_features(self, data):
        """提取板块热度特征"""
        logger.info("提取板块热度特征")
        
        sector_features = pd.DataFrame(index=data.index) if isinstance(data, pd.DataFrame) else pd.DataFrame()
        
        if isinstance(data, pd.DataFrame):
            # 板块涨幅
            if 'sector_change' in data.columns:
                sector_features['sector_change'] = data['sector_change']
                # 板块涨幅5日趋势
                sector_features['sector_change_trend_5d'] = data['sector_change'].rolling(5).mean()
                # 板块涨幅10日趋势
                sector_features['sector_change_trend_10d'] = data['sector_change'].rolling(10).mean()
                # 板块涨幅变化率
                sector_features['sector_change_change'] = data['sector_change'].pct_change().fillna(0)
                # 板块涨幅5日变化率
                sector_features['sector_change_change_5d'] = data['sector_change'].pct_change(5).fillna(0)
            else:
                sector_features['sector_change'] = 0.0
                sector_features['sector_change_trend_5d'] = 0.0
                sector_features['sector_change_trend_10d'] = 0.0
                sector_features['sector_change_change'] = 0.0
                sector_features['sector_change_change_5d'] = 0.0
            
            # 板块资金流入量
            if 'sector_money_flow' in data.columns:
                sector_features['sector_money_flow'] = data['sector_money_flow']
                # 板块资金流入5日趋势
                sector_features['sector_money_flow_trend_5d'] = data['sector_money_flow'].rolling(5).mean()
                # 板块资金流入变化率
                sector_features['sector_money_flow_change'] = data['sector_money_flow'].pct_change().fillna(0)
            else:
                sector_features['sector_money_flow'] = 0.0
                sector_features['sector_money_flow_trend_5d'] = 0.0
                sector_features['sector_money_flow_change'] = 0.0
            
            # 板块涨跌幅排名
            if 'sector_rank' in data.columns:
                sector_features['sector_rank'] = data['sector_rank']
            else:
                sector_features['sector_rank'] = 50.0  # 默认中间排名
            
            # 板块内涨停家数
            if 'sector_limit_up_count' in data.columns:
                sector_features['sector_limit_up_count'] = data['sector_limit_up_count']
                # 板块内涨停家数5日趋势
                sector_features['sector_limit_up_count_trend_5d'] = data['sector_limit_up_count'].rolling(5).mean()
                # 板块内涨停家数变化率
                sector_features['sector_limit_up_count_change'] = data['sector_limit_up_count'].pct_change().fillna(0)
            else:
                sector_features['sector_limit_up_count'] = 0.0
                sector_features['sector_limit_up_count_trend_5d'] = 0.0
                sector_features['sector_limit_up_count_change'] = 0.0
            
            # 板块联动特征
            # 板块与大盘的相关性
            if 'index_change' in data.columns:
                # 计算板块与大盘的5日相关性
                sector_features['sector_index_correlation_5d'] = data['sector_change'].rolling(5).corr(data['index_change'].rolling(5))
            
            # 板块热度变化率
            if 'sector_hotness' in data.columns:
                sector_features['sector_hotness'] = data['sector_hotness']
                sector_features['sector_hotness_change'] = data['sector_hotness'].pct_change().fillna(0)
                sector_features['sector_hotness_trend_5d'] = data['sector_hotness'].rolling(5).mean()
            
            # 板块内股票平均涨跌幅
            if 'sector_avg_return' in data.columns:
                sector_features['sector_avg_return'] = data['sector_avg_return']
                sector_features['sector_avg_return_trend_5d'] = data['sector_avg_return'].rolling(5).mean()
        
        return sector_features
    
    def _extract_market_features(self, data):
        """提取市场情绪特征"""
        logger.info("提取市场情绪特征")
        
        market_features = pd.DataFrame(index=data.index) if isinstance(data, pd.DataFrame) else pd.DataFrame()
        
        if isinstance(data, pd.DataFrame):
            # 大盘指数走势
            if 'index_change' in data.columns:
                market_features['index_change'] = data['index_change']
                # 大盘指数波动率
                if 'index_volatility' in data.columns:
                    market_features['index_volatility'] = data['index_volatility']
                else:
                    market_features['index_volatility'] = 0.0
            else:
                market_features['index_change'] = 0.0
                market_features['index_volatility'] = 0.0
            
            # 涨跌家数比
            if 'up_down_ratio' in data.columns:
                market_features['up_down_ratio'] = data['up_down_ratio']
                # 涨跌家数比变化率
                if 'up_down_ratio_prev' in data.columns:
                    market_features['up_down_ratio_change'] = (data['up_down_ratio'] - data['up_down_ratio_prev']) / data['up_down_ratio_prev']
                else:
                    market_features['up_down_ratio_change'] = 0.0
            else:
                market_features['up_down_ratio'] = 1.0  # 默认涨跌平衡
                market_features['up_down_ratio_change'] = 0.0
            
            # 赚钱效应
            if 'profit_effect' in data.columns:
                market_features['profit_effect'] = data['profit_effect']
                # 赚钱效应变化率
                if 'profit_effect_prev' in data.columns:
                    market_features['profit_effect_change'] = (data['profit_effect'] - data['profit_effect_prev']) / data['profit_effect_prev']
                else:
                    market_features['profit_effect_change'] = 0.0
            else:
                market_features['profit_effect'] = 0.0
                market_features['profit_effect_change'] = 0.0
            
            # 市场成交量
            if 'market_volume' in data.columns:
                market_features['market_volume'] = data['market_volume']
                # 市场成交量变化率
                if 'market_volume_prev' in data.columns:
                    market_features['market_volume_change'] = (data['market_volume'] - data['market_volume_prev']) / data['market_volume_prev']
                else:
                    market_features['market_volume_change'] = 0.0
            else:
                market_features['market_volume'] = 0.0
                market_features['market_volume_change'] = 0.0
            
            # 市场量比
            if 'market_volume_ratio' in data.columns:
                market_features['market_volume_ratio'] = data['market_volume_ratio']
            else:
                market_features['market_volume_ratio'] = 1.0  # 默认量比1
            
            # 新增市场情绪特征：大单成交比例和金额
            if 'large_order_volume' in data.columns and 'volume' in data.columns:
                # 大单成交比例
                market_features['large_order_ratio'] = data['large_order_volume'] / data['volume']
                # 大单成交金额
                if 'large_order_amount' in data.columns:
                    market_features['large_order_amount'] = data['large_order_amount']
                # 大单成交变化率
                if 'large_order_volume_prev' in data.columns:
                    market_features['large_order_ratio_change'] = (data['large_order_volume'] - data['large_order_volume_prev']) / data['large_order_volume_prev']
                else:
                    market_features['large_order_ratio_change'] = 0.0
            
            # 新增市场情绪特征：主力资金流入流出
            if 'main_capital_inflow' in data.columns:
                market_features['main_capital_inflow'] = data['main_capital_inflow']
                
                if 'volume' in data.columns and data['volume'].sum() > 0:
                    # 主力资金流入比例
                    market_features['main_capital_ratio'] = data['main_capital_inflow'] / data['volume']
                
                # 主力资金流入变化率
                if 'main_capital_inflow_prev' in data.columns:
                    market_features['main_capital_inflow_change'] = (data['main_capital_inflow'] - data['main_capital_inflow_prev']) / data['main_capital_inflow_prev']
                else:
                    market_features['main_capital_inflow_change'] = 0.0
            
            # 新增市场情绪特征：散户资金占比
            if 'retail_capital_ratio' in data.columns:
                market_features['retail_capital_ratio'] = data['retail_capital_ratio']
                # 散户资金占比变化率
                if 'retail_capital_ratio_prev' in data.columns:
                    market_features['retail_capital_ratio_change'] = (data['retail_capital_ratio'] - data['retail_capital_ratio_prev']) / data['retail_capital_ratio_prev']
                else:
                    market_features['retail_capital_ratio_change'] = 0.0
            
            # 新增市场情绪特征：机构持仓变化
            if 'institutional_holding_change' in data.columns:
                market_features['institutional_holding_change'] = data['institutional_holding_change']
            
            # 新增市场情绪特征：北向资金流向
            if 'northbound_flow' in data.columns:
                market_features['northbound_flow'] = data['northbound_flow']
                # 北向资金流入比例
                if 'market_volume' in data.columns and data['market_volume'] > 0:
                    market_features['northbound_flow_ratio'] = data['northbound_flow'] / data['market_volume']
                else:
                    market_features['northbound_flow_ratio'] = 0.0
            
            # 新增市场情绪特征：资金流向强度
            if 'capital_flow_strength' in data.columns:
                market_features['capital_flow_strength'] = data['capital_flow_strength']
            
            # 新增市场情绪特征：内外盘比
            if 'inner_volume' in data.columns and 'outer_volume' in data.columns:
                market_features['inner_outer_ratio'] = data['outer_volume'] / data['inner_volume']
                # 处理可能的除零情况
                market_features['inner_outer_ratio'] = market_features['inner_outer_ratio'].fillna(1.0)
                market_features['inner_outer_ratio'] = market_features['inner_outer_ratio'].replace([np.inf, -np.inf], 1.0)
            
            # 新增市场情绪特征：市场恐慌指数
            if 'vix' in data.columns:
                market_features['vix'] = data['vix']
            else:
                market_features['vix'] = 20.0  # 默认中性水平
            
            # 新增市场情绪特征：涨停家数
            if 'limit_up_count' in data.columns:
                market_features['limit_up_count'] = data['limit_up_count']
                # 涨停家数占比
                if 'total_stock_count' in data.columns:
                    market_features['limit_up_ratio'] = data['limit_up_count'] / data['total_stock_count']
                else:
                    market_features['limit_up_ratio'] = 0.0
            else:
                market_features['limit_up_count'] = 0.0
                market_features['limit_up_ratio'] = 0.0
            
            # 新增市场情绪特征：跌停家数
            if 'limit_down_count' in data.columns:
                market_features['limit_down_count'] = data['limit_down_count']
                # 跌停家数占比
                if 'total_stock_count' in data.columns:
                    market_features['limit_down_ratio'] = data['limit_down_count'] / data['total_stock_count']
                else:
                    market_features['limit_down_ratio'] = 0.0
            else:
                market_features['limit_down_count'] = 0.0
                market_features['limit_down_ratio'] = 0.0
            
            # 新增市场情绪特征：两市成交额
            if 'market_turnover' in data.columns:
                market_features['market_turnover'] = data['market_turnover']
                # 成交额变化率
                if 'market_turnover_prev' in data.columns:
                    market_features['market_turnover_change'] = (data['market_turnover'] - data['market_turnover_prev']) / data['market_turnover_prev']
                else:
                    market_features['market_turnover_change'] = 0.0
            else:
                market_features['market_turnover'] = 0.0
                market_features['market_turnover_change'] = 0.0
            
            # 新增市场情绪特征：融资余额变化
            if 'margin_balance_change' in data.columns:
                market_features['margin_balance_change'] = data['margin_balance_change']
            else:
                market_features['margin_balance_change'] = 0.0
            
            # 新增市场情绪特征：融券余额变化
            if 'short_balance_change' in data.columns:
                market_features['short_balance_change'] = data['short_balance_change']
            else:
                market_features['short_balance_change'] = 0.0
            
            # 新增市场情绪特征：融资融券余额比
            if 'margin_balance' in data.columns and 'short_balance' in data.columns:
                market_features['margin_short_ratio'] = data['margin_balance'] / data['short_balance']
                # 处理可能的除零情况
                market_features['margin_short_ratio'] = market_features['margin_short_ratio'].fillna(0.0)
                market_features['margin_short_ratio'] = market_features['margin_short_ratio'].replace([np.inf, -np.inf], 0.0)
            else:
                market_features['margin_short_ratio'] = 0.0
        
        return market_features
    
    def _extract_fundamental_features(self, data):
        """提取个股基本面特征"""
        logger.info("提取个股基本面特征")
        
        fundamental_features = pd.DataFrame(index=data.index) if isinstance(data, pd.DataFrame) else pd.DataFrame()
        
        if isinstance(data, pd.DataFrame):
            # 1. 市值规模相关特征
            # 流通市值
            if 'circulating_market_cap' in data.columns:
                fundamental_features['circulating_market_cap'] = data['circulating_market_cap']
            else:
                fundamental_features['circulating_market_cap'] = 5000000000.0  # 默认50亿
            
            # 总市值（如果有）
            if 'total_market_cap' in data.columns:
                fundamental_features['total_market_cap'] = data['total_market_cap']
            
            # 市值规模分类（小、中、大）
            if 'circulating_market_cap' in data.columns:
                # 将市值分为5个等级
                fundamental_features['market_cap_rank'] = pd.qcut(data['circulating_market_cap'], 5, labels=False, duplicates='drop')
            
            # 2. 股价相关特征
            # 当前股价
            if 'close' in data.columns:
                fundamental_features['current_price'] = data['close']
            elif 'open' in data.columns:
                fundamental_features['current_price'] = data['open']
            else:
                fundamental_features['current_price'] = 10.0  # 默认10元
            
            # 股价波动率（新增）
            if 'volatility_5d' in data.columns:
                fundamental_features['volatility_5d'] = data['volatility_5d']
            else:
                fundamental_features['volatility_5d'] = 0.0  # 默认0
            
            # 3. 估值指标
            # 市盈率
            if 'pe_ratio' in data.columns:
                fundamental_features['pe_ratio'] = data['pe_ratio']
            else:
                fundamental_features['pe_ratio'] = 20.0  # 默认20倍
            
            # 市净率
            if 'pb_ratio' in data.columns:
                fundamental_features['pb_ratio'] = data['pb_ratio']
            else:
                fundamental_features['pb_ratio'] = 2.0  # 默认2倍
            
            # 市销率（如果有）
            if 'ps_ratio' in data.columns:
                fundamental_features['ps_ratio'] = data['ps_ratio']
            
            # 市盈率相对行业水平（如果有）
            if 'pe_ratio' in data.columns and 'industry_pe_ratio' in data.columns:
                fundamental_features['pe_industry_ratio'] = data['pe_ratio'] / data['industry_pe_ratio']
            
            # 4. 股本结构
            # 流通股比例（如果有）
            if 'circulating_share_ratio' in data.columns:
                fundamental_features['circulating_share_ratio'] = data['circulating_share_ratio']
            elif 'circulating_cap' in data.columns and 'total_shares' in data.columns:
                fundamental_features['circulating_share_ratio'] = data['circulating_cap'] / data['total_shares']
            
            # 股东户数（如果有）
            if 'shareholder_count' in data.columns:
                fundamental_features['shareholder_count'] = data['shareholder_count']
                # 股东户数变化率
                if 'shareholder_count_prev' in data.columns:
                    fundamental_features['shareholder_count_change'] = (data['shareholder_count'] - data['shareholder_count_prev']) / data['shareholder_count_prev']
            
            # 5. 财务质量
            # 净利润增长率（如果有）
            if 'net_profit_growth' in data.columns:
                fundamental_features['net_profit_growth'] = data['net_profit_growth']
            
            # 营收增长率（如果有）
            if 'revenue_growth' in data.columns:
                fundamental_features['revenue_growth'] = data['revenue_growth']
            
            # 净资产收益率（如果有）
            if 'roe' in data.columns:
                fundamental_features['roe'] = data['roe']
            
            # 资产负债率（如果有）
            if 'debt_ratio' in data.columns:
                fundamental_features['debt_ratio'] = data['debt_ratio']
            
            # 6. 其他重要特征
            # 近期业绩公告影响
            if 'earnings_announcement' in data.columns:
                fundamental_features['earnings_announcement'] = data['earnings_announcement']
            else:
                fundamental_features['earnings_announcement'] = 0  # 默认无公告
            
            # 换手率
            if 'turnover_rate' in data.columns:
                fundamental_features['turnover_rate'] = data['turnover_rate']
            else:
                fundamental_features['turnover_rate'] = 0.0  # 默认0
            
            # 北向资金持股比例（如果有）
            if 'northbound_holding_ratio' in data.columns:
                fundamental_features['northbound_holding_ratio'] = data['northbound_holding_ratio']
            
            # 机构持股比例（如果有）
            if 'institutional_holding_ratio' in data.columns:
                fundamental_features['institutional_holding_ratio'] = data['institutional_holding_ratio']
        
        return fundamental_features
    
    def _extract_technical_features(self, data):
        """提取技术指标特征"""
        logger.info("提取技术指标特征")
        
        technical_features = pd.DataFrame(index=data.index) if isinstance(data, pd.DataFrame) else pd.DataFrame()
        
        if isinstance(data, pd.DataFrame):
            # 均线相关特征
            if 'close' in data.columns:
                # 5日均线与收盘价的关系
                if 'ma5' in data.columns:
                    technical_features['price_ma5_ratio'] = data['close'] / data['ma5']
                else:
                    technical_features['price_ma5_ratio'] = 1.0  # 默认与5日均线持平
                
                # 10日均线与收盘价的关系
                if 'ma10' in data.columns:
                    technical_features['price_ma10_ratio'] = data['close'] / data['ma10']
                else:
                    technical_features['price_ma10_ratio'] = 1.0  # 默认与10日均线持平
            else:
                technical_features['price_ma5_ratio'] = 1.0
                technical_features['price_ma10_ratio'] = 1.0
            
            # MACD相关特征
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                technical_features['macd_diff'] = data['macd'] - data['macd_signal']
            else:
                technical_features['macd_diff'] = 0.0  # 默认MACD金叉
            
            # RSI相关特征
            if 'rsi_14' in data.columns:
                technical_features['rsi_14'] = data['rsi_14']
            else:
                technical_features['rsi_14'] = 50.0  # 默认中性
            
            # 成交量相关特征
            if 'volume' in data.columns:
                # 量比
                if 'volume_ratio' in data.columns:
                    technical_features['volume_ratio'] = data['volume_ratio']
                else:
                    technical_features['volume_ratio'] = 1.0  # 默认量比1
                
                # 成交量变化率，使用fillna(0)处理NaN值，避免数据长度减少
                technical_features['volume_change_rate'] = data['volume'].pct_change().fillna(0)
            else:
                technical_features['volume_ratio'] = 1.0
                technical_features['volume_change_rate'] = 0.0
            
            # 新增技术指标：布林带（Bollinger Bands）
            if 'ma20' in data.columns and 'close' in data.columns:
                # 计算标准差
                if 'std20' in data.columns:
                    std20 = data['std20']
                else:
                    # 如果没有提供std20，尝试计算（需要足够的数据）
                    if len(data) >= 20:
                        std20 = data['close'].rolling(20).std()
                    else:
                        std20 = 0.0
                
                # 布林带上轨、中轨、下轨
                technical_features['bollinger_upper'] = data['ma20'] + 2 * std20
                technical_features['bollinger_middle'] = data['ma20']
                technical_features['bollinger_lower'] = data['ma20'] - 2 * std20
                
                # 布林带带宽和百分比
                technical_features['bollinger_bandwidth'] = (technical_features['bollinger_upper'] - technical_features['bollinger_lower']) / data['ma20']
                technical_features['bollinger_percent'] = (data['close'] - technical_features['bollinger_lower']) / (technical_features['bollinger_upper'] - technical_features['bollinger_lower'])
                
                # 处理可能的除零情况
                technical_features['bollinger_bandwidth'] = technical_features['bollinger_bandwidth'].fillna(0.0)
                technical_features['bollinger_bandwidth'] = technical_features['bollinger_bandwidth'].replace([np.inf, -np.inf], 0.0)
                technical_features['bollinger_percent'] = technical_features['bollinger_percent'].fillna(0.5)
                technical_features['bollinger_percent'] = technical_features['bollinger_percent'].replace([np.inf, -np.inf], 0.5)
            
            # 新增技术指标：KDJ指标
            if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
                # 计算RSV（未成熟随机值）
                if len(data) >= 9:
                    lowest_low = data['low'].rolling(9).min()
                    highest_high = data['high'].rolling(9).max()
                    rsv = (data['close'] - lowest_low) / (highest_high - lowest_low) * 100
                    rsv = rsv.fillna(50.0)
                    
                    # 计算K、D、J值
                    technical_features['kdj_k'] = rsv.ewm(alpha=1/3, adjust=False).mean()
                    technical_features['kdj_d'] = technical_features['kdj_k'].ewm(alpha=1/3, adjust=False).mean()
                    technical_features['kdj_j'] = 3 * technical_features['kdj_k'] - 2 * technical_features['kdj_d']
                else:
                    technical_features['kdj_k'] = 50.0
                    technical_features['kdj_d'] = 50.0
                    technical_features['kdj_j'] = 50.0
            
            # 新增技术指标：OBV（能量潮）
            if 'volume' in data.columns and 'close' in data.columns:
                if len(data) >= 1:
                    # 计算价格变化
                    price_change = data['close'].diff()
                    # 计算成交量方向
                    volume_direction = price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                    # 计算OBV
                    technical_features['obv'] = (data['volume'] * volume_direction).cumsum()
                else:
                    technical_features['obv'] = 0.0
            
            # 新增技术指标：ATR（平均真实波幅）
            if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
                if len(data) >= 14:
                    # 计算真实波幅（TR）
                    tr1 = data['high'] - data['low']
                    tr2 = abs(data['high'] - data['close'].shift(1))
                    tr3 = abs(data['low'] - data['close'].shift(1))
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    # 计算ATR
                    technical_features['atr'] = tr.rolling(14).mean()
                else:
                    technical_features['atr'] = 0.0
            
            # 新增技术指标：ADX（平均趋向指数）
            if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
                if len(data) >= 14:
                    # 计算上升趋向和下降趋向
                    plus_dm = data['high'].diff().where(data['high'].diff() > data['low'].diff().abs(), 0)
                    minus_dm = data['low'].diff().abs().where(data['low'].diff() < data['high'].diff(), 0)
                    
                    # 计算真实波幅
                    tr1 = data['high'] - data['low']
                    tr2 = abs(data['high'] - data['close'].shift(1))
                    tr3 = abs(data['low'] - data['close'].shift(1))
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    
                    # 计算+DI、-DI和DX
                    plus_di = 100 * (plus_dm.rolling(14).sum() / tr.rolling(14).sum())
                    minus_di = 100 * (minus_dm.rolling(14).sum() / tr.rolling(14).sum())
                    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                    
                    # 计算ADX
                    technical_features['adx'] = dx.rolling(14).mean()
                    technical_features['plus_di'] = plus_di
                    technical_features['minus_di'] = minus_di
                else:
                    technical_features['adx'] = 20.0
                    technical_features['plus_di'] = 20.0
                    technical_features['minus_di'] = 20.0
            
            # 新增价格形态和趋势特征
            # 1. 突破特征
            if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
                # 突破前5日最高价
                if len(data) >= 5:
                    prev_5d_high = data['high'].shift(1).rolling(5).max()
                    technical_features['break_prev_5d_high'] = (data['high'] > prev_5d_high).astype(int)
                else:
                    technical_features['break_prev_5d_high'] = 0
                
                # 突破前10日最高价
                if len(data) >= 10:
                    prev_10d_high = data['high'].shift(1).rolling(10).max()
                    technical_features['break_prev_10d_high'] = (data['high'] > prev_10d_high).astype(int)
                else:
                    technical_features['break_prev_10d_high'] = 0
                
                # 突破前20日最高价
                if len(data) >= 20:
                    prev_20d_high = data['high'].shift(1).rolling(20).max()
                    technical_features['break_prev_20d_high'] = (data['high'] > prev_20d_high).astype(int)
                else:
                    technical_features['break_prev_20d_high'] = 0
            
            # 2. 支撑压力位特征
            if 'close' in data.columns:
                # 计算前10日的支撑位（最低价均值）
                if len(data) >= 10:
                    support_10d = data['low'].shift(1).rolling(10).mean()
                    # 收盘价距离支撑位的距离
                    technical_features['distance_to_support_10d'] = (data['close'] - support_10d) / support_10d
                else:
                    technical_features['distance_to_support_10d'] = 0.0
                
                # 计算前10日的压力位（最高价均值）
                if len(data) >= 10:
                    resistance_10d = data['high'].shift(1).rolling(10).mean()
                    # 收盘价距离压力位的距离
                    technical_features['distance_to_resistance_10d'] = (data['close'] - resistance_10d) / resistance_10d
                else:
                    technical_features['distance_to_resistance_10d'] = 0.0
            
            # 3. 趋势特征
            if 'close' in data.columns:
                # 短期趋势（5日）
                if len(data) >= 5:
                    technical_features['trend_5d'] = data['close'].rolling(5).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
                else:
                    technical_features['trend_5d'] = 0.0
                
                # 中期趋势（20日）
                if len(data) >= 20:
                    technical_features['trend_20d'] = data['close'].rolling(20).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
                else:
                    technical_features['trend_20d'] = 0.0
                
                # 长期趋势（60日）
                if len(data) >= 60:
                    technical_features['trend_60d'] = data['close'].rolling(60).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
                else:
                    technical_features['trend_60d'] = 0.0
            
            # 4. 均线多头/空头排列特征
            if 'ma5' in data.columns and 'ma10' in data.columns and 'ma20' in data.columns:
                # 多头排列：短期均线上穿中期均线，中期均线上穿长期均线
                technical_features['bullish_ma_arrangement'] = ((data['ma5'] > data['ma10']) & (data['ma10'] > data['ma20'])).astype(int)
                # 空头排列：短期均线下穿中期均线，中期均线下穿长期均线
                technical_features['bearish_ma_arrangement'] = ((data['ma5'] < data['ma10']) & (data['ma10'] < data['ma20'])).astype(int)
                # 均线粘合度：计算5日、10日、20日均线的标准差，越小越粘合
                ma_values = pd.concat([data['ma5'], data['ma10'], data['ma20']], axis=1)
                technical_features['ma_cohesion'] = ma_values.std(axis=1) / ma_values.mean(axis=1)
            else:
                technical_features['bullish_ma_arrangement'] = 0
                technical_features['bearish_ma_arrangement'] = 0
                technical_features['ma_cohesion'] = 0.0
            
            # 5. 价格形态特征
            if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns and 'open' in data.columns:
                # 锤子线形态（下影线是实体的2倍以上，上影线很短）
                if len(data) >= 1:
                    body = abs(data['close'] - data['open'])
                    upper_shadow = data['high'] - data['close'].where(data['close'] > data['open'], data['open'])
                    lower_shadow = data['open'].where(data['close'] > data['open'], data['close']) - data['low']
                    # 锤子线：下影线长度 > 2倍实体长度，且上影线 < 实体长度
                    technical_features['hammer_pattern'] = ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int)
                else:
                    technical_features['hammer_pattern'] = 0
                
                # 倒锤子线形态（上影线是实体的2倍以上，下影线很短）
                if len(data) >= 1:
                    body = abs(data['close'] - data['open'])
                    upper_shadow = data['high'] - data['close'].where(data['close'] > data['open'], data['open'])
                    lower_shadow = data['open'].where(data['close'] > data['open'], data['close']) - data['low']
                    # 倒锤子线：上影线长度 > 2倍实体长度，且下影线 < 实体长度
                    technical_features['inverted_hammer_pattern'] = ((upper_shadow > 2 * body) & (lower_shadow < body)).astype(int)
                else:
                    technical_features['inverted_hammer_pattern'] = 0
                
                # 新增形态：十字星
                if len(data) >= 1:
                    body = abs(data['close'] - data['open'])
                    range_total = data['high'] - data['low']
                    # 十字星：实体很小（<1%），上下影线较长
                    technical_features['doji_pattern'] = ((body / range_total < 0.1) & (body / data['close'] < 0.01)).astype(int)
                else:
                    technical_features['doji_pattern'] = 0
                
                # 新增形态：看涨吞没
                if len(data) >= 2:
                    # 昨天的K线
                    prev_close = data['close'].shift(1)
                    prev_open = data['open'].shift(1)
                    prev_high = data['high'].shift(1)
                    prev_low = data['low'].shift(1)
                    
                    # 今天的K线
                    curr_close = data['close']
                    curr_open = data['open']
                    
                    # 看涨吞没：昨天是阴线，今天是阳线，今天的开盘价低于昨天的收盘价，今天的收盘价高于昨天的开盘价
                    technical_features['bullish_engulfing'] = ((prev_close < prev_open) & (curr_close > curr_open) & \
                                                          (curr_open < prev_close) & (curr_close > prev_open)).astype(int)
                    
                    # 看跌吞没
                    technical_features['bearish_engulfing'] = ((prev_close > prev_open) & (curr_close < curr_open) & \
                                                          (curr_open > prev_close) & (curr_close < prev_open)).astype(int)
                else:
                    technical_features['bullish_engulfing'] = 0
                    technical_features['bearish_engulfing'] = 0
            
            # 6. MACD相关增强特征
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                # MACD金叉/死叉
                technical_features['macd_golden_cross'] = ((data['macd'].shift(1) < data['macd_signal'].shift(1)) & \
                                                      (data['macd'] > data['macd_signal'])).astype(int)
                technical_features['macd_death_cross'] = ((data['macd'].shift(1) > data['macd_signal'].shift(1)) & \
                                                      (data['macd'] < data['macd_signal'])).astype(int)
                # MACD柱状图变化
                if 'macd_hist' in data.columns:
                    technical_features['macd_hist'] = data['macd_hist']
                    technical_features['macd_hist_change'] = data['macd_hist'] - data['macd_hist'].shift(1)
            else:
                technical_features['macd_golden_cross'] = 0
                technical_features['macd_death_cross'] = 0
            
            # 7. RSI相关增强特征
            if 'rsi_14' in data.columns:
                # RSI超买超卖
                technical_features['rsi_overbought'] = (data['rsi_14'] > 70).astype(int)
                technical_features['rsi_oversold'] = (data['rsi_14'] < 30).astype(int)
            else:
                technical_features['rsi_overbought'] = 0
                technical_features['rsi_oversold'] = 0
            
            # 8. KDJ相关增强特征
            if 'kdj_k' in data.columns and 'kdj_d' in data.columns and 'kdj_j' in data.columns:
                # KDJ金叉/死叉
                technical_features['kdj_golden_cross'] = ((data['kdj_k'].shift(1) < data['kdj_d'].shift(1)) & \
                                                     (data['kdj_k'] > data['kdj_d'])).astype(int)
                technical_features['kdj_death_cross'] = ((data['kdj_k'].shift(1) > data['kdj_d'].shift(1)) & \
                                                     (data['kdj_k'] < data['kdj_d'])).astype(int)
                # KDJ超买超卖
                technical_features['kdj_overbought'] = ((data['kdj_k'] > 80) & (data['kdj_d'] > 80)).astype(int)
                technical_features['kdj_oversold'] = ((data['kdj_k'] < 20) & (data['kdj_d'] < 20)).astype(int)
            else:
                technical_features['kdj_golden_cross'] = 0
                technical_features['kdj_death_cross'] = 0
                technical_features['kdj_overbought'] = 0
                technical_features['kdj_oversold'] = 0
            
            # 9. 成交量相关增强特征
            if 'volume' in data.columns:
                # 成交量创新高/低
                if len(data) >= 20:
                    technical_features['volume_new_high'] = (data['volume'] == data['volume'].rolling(20).max()).astype(int)
                    technical_features['volume_new_low'] = (data['volume'] == data['volume'].rolling(20).min()).astype(int)
                else:
                    technical_features['volume_new_high'] = 0
                    technical_features['volume_new_low'] = 0
                
                # 量价配合：价格上涨，成交量放大；价格下跌，成交量萎缩
                price_change = data['close'].pct_change()
                volume_change = data['volume'].pct_change()
                technical_features['volume_price_fit'] = ((price_change > 0) & (volume_change > 0) | \
                                                     (price_change < 0) & (volume_change < 0)).astype(int)
            else:
                technical_features['volume_new_high'] = 0
                technical_features['volume_new_low'] = 0
                technical_features['volume_price_fit'] = 0
            
            # 10. 波动相关特征
            if 'close' in data.columns:
                # 价格波动率
                if len(data) >= 5:
                    technical_features['price_volatility_5d'] = data['close'].rolling(5).std() / data['close'].rolling(5).mean()
            
            # 11. 涨停相关特征
            if 'close' in data.columns and 'prev_close' in data.columns:
                # 计算每日涨跌幅
                data['daily_return'] = (data['close'] - data['prev_close']) / data['prev_close']
                
                # 涨停阈值（9.8%）
                limit_up_threshold = 0.098
                
                # 判断当日是否涨停
                data['is_limit_up'] = (data['daily_return'] >= limit_up_threshold).astype(int)
                
                # 11.1 前N日涨停频率与强度
                # 前5日涨停次数
                technical_features['limit_up_count_5d'] = data['is_limit_up'].rolling(5).sum()
                # 前10日涨停次数
                technical_features['limit_up_count_10d'] = data['is_limit_up'].rolling(10).sum()
                # 前20日涨停次数
                technical_features['limit_up_count_20d'] = data['is_limit_up'].rolling(20).sum()
                
                # 11.2 连续涨停特征
                # 连续涨停天数
                def count_consecutive_limit_up(x):
                    count = 0
                    for i in reversed(x):
                        if i == 1:
                            count += 1
                        else:
                            break
                    return count
                
                technical_features['consecutive_limit_up'] = data['is_limit_up'].rolling(10).apply(count_consecutive_limit_up, raw=True)
                
                # 11.3 涨停基因（历史涨停次数）
                technical_features['total_limit_up_count'] = data['is_limit_up'].cumsum()
                
                # 11.4 前一天是否涨停
                technical_features['prev_day_limit_up'] = data['is_limit_up'].shift(1)
                
                # 11.5 涨停强度
                technical_features['limit_up_strength'] = data['daily_return'].where(data['is_limit_up'] == 1, 0)
                
                # 11.6 涨停后表现
                # 涨停次日涨跌幅
                technical_features['after_limit_up_return'] = data['daily_return'].shift(-1).where(data['is_limit_up'] == 1, 0)
            
            # 12. 板块涨停特征
            # 板块涨停密度（假设data中包含板块涨停股票数量）
            if 'sector_limit_up_count' in data.columns and 'sector_stock_count' in data.columns:
                technical_features['sector_limit_up_density'] = data['sector_limit_up_count'] / data['sector_stock_count']
            
            # 13. 资金流向特征
            # 主力资金流入比例（假设data中包含主力资金流入数据）
            if 'main_capital_inflow' in data.columns and 'volume' in data.columns:
                technical_features['main_capital_ratio'] = data['main_capital_inflow'] / data['volume']
                
            # 14. 龙虎榜特征
            # 龙虎榜资金流入（假设data中包含龙虎榜数据）
            if 'dragon_tiger_inflow' in data.columns:
                technical_features['dragon_tiger_inflow'] = data['dragon_tiger_inflow']
                technical_features['dragon_tiger_inflow_ratio'] = data['dragon_tiger_inflow'] / data['volume']
            
            # 15. 盘口委托队列特征
            # 买卖盘委托比（假设data中包含盘口数据）
            if 'buy_order_volume' in data.columns and 'sell_order_volume' in data.columns:
                technical_features['order_volume_ratio'] = data['buy_order_volume'] / (data['sell_order_volume'] + 1e-6)
            
            # 16. 涨停延续性特征
            # 近3日涨停连续性
            if 'is_limit_up' in data.columns:
                technical_features['limit_up_continuity_3d'] = (data['is_limit_up'] + data['is_limit_up'].shift(1) + data['is_limit_up'].shift(2)).rolling(3).sum() / 3
            
            # 17. 价格波动率扩展
            if 'close' in data.columns:
                # 10日价格波动率
                if len(data) >= 10:
                    technical_features['price_volatility_10d'] = data['close'].rolling(10).std() / data['close'].rolling(10).mean()
                else:
                    technical_features['price_volatility_10d'] = 0.0
        
        return technical_features
    
    def _merge_features(self, feature_list):
        """合并所有特征"""
        logger.info("合并所有特征")
        
        # 过滤掉空的特征DataFrame
        valid_features = [f for f in feature_list if not f.empty]
        
        if not valid_features:
            return pd.DataFrame()
        
        # 合并所有特征
        all_features = pd.concat(valid_features, axis=1)
        
        return all_features
    
    def _select_features(self, features):
        """特征选择和重要性评估"""
        logger.info("特征选择和重要性评估")
        
        if features.empty:
            return features
        
        # 分离标签列和特征列
        has_label = 'label' in features.columns
        if has_label:
            labels = features['label']
            feature_cols = features.drop('label', axis=1)
        else:
            feature_cols = features.copy()
            labels = None
        
        # 1. 方差选择：删除低方差特征
        logger.info("进行方差选择")
        from sklearn.feature_selection import VarianceThreshold
        # 动态调整方差阈值，对于小样本测试数据使用更低的阈值
        variance_threshold = 0.001 if len(feature_cols) < 100 else 0.01  # 小样本使用更低的阈值
        variance_selector = VarianceThreshold(threshold=variance_threshold)  # 保留方差大于阈值的特征
        feature_cols = pd.DataFrame(variance_selector.fit_transform(feature_cols), 
                                   index=feature_cols.index,
                                   columns=feature_cols.columns[variance_selector.get_support()])
        logger.info(f"方差选择后特征数量：{len(feature_cols.columns)}")
        
        # 如果方差选择后没有特征，使用所有特征
        if feature_cols.empty:
            logger.warning(f"方差选择后没有特征，使用所有原始特征")
            feature_cols = features.copy() if not has_label else features.drop('label', axis=1)
        
        # 如果没有特征剩下，返回空DataFrame
        if feature_cols.empty:
            return pd.DataFrame() if not has_label else pd.DataFrame({'label': labels})
        
        # 1.1 增加特征交互项
        logger.info("增加特征交互项")
        interaction_features = pd.DataFrame(index=feature_cols.index)
        
        # 选择重要的特征列进行交互
        important_feature_cols = feature_cols.columns.tolist()[:25]  # 选择前25个重要特征进行交互
        
        # 生成特征交互项
        # 1. 价格与成交量的交互
        if 'bid_price_change' in important_feature_cols and 'bid_volume_change' in important_feature_cols:
            interaction_features['price_volume_interaction'] = feature_cols['bid_price_change'] * feature_cols['bid_volume_change']
        
        # 2. 竞价强度与板块热度的交互
        if 'bid_intensity' in important_feature_cols and 'sector_change' in important_feature_cols:
            interaction_features['bid_sector_interaction'] = feature_cols['bid_intensity'] * feature_cols['sector_change']
        
        # 3. 市场情绪与个股表现的交互
        if 'up_down_ratio' in important_feature_cols and 'bid_price_change' in important_feature_cols:
            interaction_features['market_stock_interaction'] = feature_cols['up_down_ratio'] * feature_cols['bid_price_change']
        
        # 4. 技术指标与竞价特征的交互
        if 'rsi_14' in important_feature_cols and 'bid_intensity' in important_feature_cols:
            interaction_features['rsi_bid_interaction'] = feature_cols['rsi_14'] * feature_cols['bid_intensity']
        
        # 5. 均线关系与成交量的交互
        if 'price_ma5_ratio' in important_feature_cols and 'volume_ratio' in important_feature_cols:
            interaction_features['ma_volume_interaction'] = feature_cols['price_ma5_ratio'] * feature_cols['volume_ratio']
        
        # 6. 封单量与板块资金流向的交互
        if 'buy_order_size_ratio' in important_feature_cols and 'sector_money_flow' in important_feature_cols:
            interaction_features['order_flow_interaction'] = feature_cols['buy_order_size_ratio'] * feature_cols['sector_money_flow']
        
        # 7. 趋势特征与竞价特征的交互
        if 'trend_5d' in important_feature_cols and 'bid_intensity' in important_feature_cols:
            interaction_features['trend_bid_interaction'] = feature_cols['trend_5d'] * feature_cols['bid_intensity']
        
        # 8. 布林带特征与成交量的交互
        if 'bollinger_percent' in important_feature_cols and 'volume_ratio' in important_feature_cols:
            interaction_features['bollinger_volume_interaction'] = feature_cols['bollinger_percent'] * feature_cols['volume_ratio']
        
        # 9. KDJ指标与趋势特征的交互
        if 'kdj_j' in important_feature_cols and 'trend_20d' in important_feature_cols:
            interaction_features['kdj_trend_interaction'] = feature_cols['kdj_j'] * feature_cols['trend_20d']
        
        # 10. 突破特征与资金流向的交互
        if 'break_prev_10d_high' in important_feature_cols and 'main_capital_ratio' in important_feature_cols:
            interaction_features['break_flow_interaction'] = feature_cols['break_prev_10d_high'] * feature_cols['main_capital_ratio']
        
        # 11. 竞价强度与市场情绪的交互
        if 'bid_intensity' in important_feature_cols and 'profit_effect' in important_feature_cols:
            interaction_features['bid_market_sentiment_interaction'] = feature_cols['bid_intensity'] * feature_cols['profit_effect']
        
        # 12. MACD与成交量的交互
        if 'macd_diff' in important_feature_cols and 'volume_change_rate' in important_feature_cols:
            interaction_features['macd_volume_interaction'] = feature_cols['macd_diff'] * feature_cols['volume_change_rate']
        
        # 13. KDJ与RSI的交互
        if 'kdj_j' in important_feature_cols and 'rsi_14' in important_feature_cols:
            interaction_features['kdj_rsi_interaction'] = feature_cols['kdj_j'] * feature_cols['rsi_14']
        
        # 14. 流通市值与市盈率的交互
        if 'circulating_market_cap' in important_feature_cols and 'pe_ratio' in important_feature_cols:
            interaction_features['market_cap_pe_interaction'] = feature_cols['circulating_market_cap'] * feature_cols['pe_ratio']
        
        # 15. 板块涨停家数与个股竞价强度的交互
        if 'sector_limit_up_count' in important_feature_cols and 'bid_intensity' in important_feature_cols:
            interaction_features['sector_limit_up_bid_interaction'] = feature_cols['sector_limit_up_count'] * feature_cols['bid_intensity']
        
        # 合并原始特征和交互特征
        if not interaction_features.empty:
            feature_cols = pd.concat([feature_cols, interaction_features], axis=1)
            logger.info(f"添加交互特征后特征数量：{len(feature_cols.columns)}")
        
        # 2. 标准化特征
        logger.info("标准化特征")
        standardized_features = standardize_data(feature_cols)
        
        # 3. 如果有标签，进行基于模型的特征选择
        if has_label:
            # 3.1 互信息法选择
            logger.info("进行互信息法特征选择")
            from sklearn.feature_selection import mutual_info_classif
            
            # 处理NaN值，确保输入到mutual_info_classif的数据没有NaN
            standardized_features = standardized_features.fillna(0)
            
            mutual_info = mutual_info_classif(standardized_features, labels, random_state=42)
            
            # 只保留互信息前50%的特征
            mi_threshold = np.percentile(mutual_info, 50)
            selected_cols_mi = standardized_features.columns[mutual_info >= mi_threshold]
            standardized_features = standardized_features[selected_cols_mi]
            logger.info(f"互信息选择后特征数量：{len(standardized_features.columns)}")
            
            if standardized_features.empty:
                return pd.DataFrame({'label': labels})
            
            # 3.2 卡方检验选择（仅适用于非负特征）
            logger.info("进行卡方检验特征选择")
            from sklearn.feature_selection import SelectKBest, chi2
            # 确保特征非负（卡方检验要求）
            standardized_features = standardized_features.clip(lower=0)
            
            # 使用卡方检验选择特征
            chi2_selector = SelectKBest(chi2, k='all')
            chi2_selector.fit(standardized_features, labels)
            chi2_scores = chi2_selector.scores_
            
            # 只保留卡方检验前50%的特征
            chi2_threshold = np.percentile(chi2_scores, 50)
            selected_cols_chi2 = standardized_features.columns[chi2_scores >= chi2_threshold]
            
            # 如果卡方检验后没有特征，使用所有特征
            if len(selected_cols_chi2) == 0:
                logger.warning(f"卡方检验后没有特征，使用所有特征")
                standardized_features = standardized_features  # 保持不变
            else:
                standardized_features = standardized_features[selected_cols_chi2]
            
            logger.info(f"卡方检验选择后特征数量：{len(standardized_features.columns)}")
            
            if standardized_features.empty:
                return pd.DataFrame({'label': labels})
            
            # 3.3 递归特征消除（RFE）：仅当特征数量 >= 2时执行
            logger.info("进行递归特征消除")
            if len(standardized_features.columns) >= 2:
                from sklearn.feature_selection import RFE
                from sklearn.ensemble import RandomForestClassifier
                
                # 使用随机森林作为基模型
                rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                # 保留50%的特征
                n_features_to_select = max(1, int(len(standardized_features.columns) * 0.5))
                rfe_selector = RFE(estimator=rf_model, n_features_to_select=n_features_to_select, step=1)
                rfe_selector.fit(standardized_features, labels)
                
                selected_cols_rfe = standardized_features.columns[rfe_selector.support_]
                standardized_features = standardized_features[selected_cols_rfe]
                logger.info(f"递归特征消除后特征数量：{len(standardized_features.columns)}")
            else:
                logger.info(f"特征数量不足，跳过递归特征消除")
            
            if standardized_features.empty:
                return pd.DataFrame({'label': labels})
            
            # 3.4 L1正则化（Lasso）特征选择
            logger.info("进行L1正则化（Lasso）特征选择")
            from sklearn.linear_model import Lasso
            
            # 使用Lasso进行特征选择
            lasso = Lasso(alpha=0.1, random_state=42)
            lasso.fit(standardized_features, labels)
            
            # 获取Lasso选择的特征（系数不为0的特征）
            lasso_selected_mask = lasso.coef_ != 0
            if any(lasso_selected_mask):
                selected_cols_lasso = standardized_features.columns[lasso_selected_mask]
                standardized_features = standardized_features[selected_cols_lasso]
                logger.info(f"L1正则化选择后特征数量：{len(standardized_features.columns)}")
            else:
                logger.warning("L1正则化后没有特征被选择，使用所有当前特征")
            
            if standardized_features.empty:
                return pd.DataFrame({'label': labels})
            
            # 3.5 弹性网络（Elastic Net）特征选择
            logger.info("进行弹性网络（Elastic Net）特征选择")
            from sklearn.linear_model import ElasticNet
            
            # 使用Elastic Net进行特征选择
            elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            elastic_net.fit(standardized_features, labels)
            
            # 获取Elastic Net选择的特征（系数不为0的特征）
            elastic_net_selected_mask = elastic_net.coef_ != 0
            if any(elastic_net_selected_mask):
                selected_cols_elastic_net = standardized_features.columns[elastic_net_selected_mask]
                standardized_features = standardized_features[selected_cols_elastic_net]
                logger.info(f"弹性网络选择后特征数量：{len(standardized_features.columns)}")
            else:
                logger.warning("弹性网络后没有特征被选择，使用所有当前特征")
            
            if standardized_features.empty:
                return pd.DataFrame({'label': labels})
            
            # 3.6 特征降维：PCA
            logger.info("进行PCA降维")
            from sklearn.decomposition import PCA
            # 保留95%的方差
            pca = PCA(n_components=0.95, random_state=42)
            pca_features = pca.fit_transform(standardized_features)
            
            # 转换为DataFrame
            pca_columns = [f'pca_{i}' for i in range(pca_features.shape[1])]
            standardized_features = pd.DataFrame(pca_features, 
                                               index=standardized_features.index, 
                                               columns=pca_columns)
            logger.info(f"PCA降维后特征数量：{len(standardized_features.columns)}")
            
            # 3.7 特征降维：LDA（线性判别分析）
            logger.info("进行LDA降维")
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            
            # LDA的n_components不能超过类别数-1
            n_classes = len(np.unique(labels))
            n_lda_components = min(len(standardized_features.columns), n_classes - 1)
            
            if n_lda_components > 0:
                try:
                    # 尝试使用random_state参数（仅支持sklearn 1.0+）
                    lda = LinearDiscriminantAnalysis(n_components=n_lda_components, random_state=42)
                    lda_features = lda.fit_transform(standardized_features, labels)
                except TypeError:
                    # 如果不支持random_state参数，不使用该参数
                    logger.warning("LDA不支持random_state参数，使用默认初始化")
                    lda = LinearDiscriminantAnalysis(n_components=n_lda_components)
                    lda_features = lda.fit_transform(standardized_features, labels)
                
                # 转换为DataFrame
                lda_columns = [f'lda_{i}' for i in range(lda_features.shape[1])]
                standardized_features = pd.DataFrame(lda_features, 
                                                   index=standardized_features.index, 
                                                   columns=lda_columns)
                logger.info(f"LDA降维后特征数量：{len(standardized_features.columns)}")
            
            # 4. 进行特征重要性评估
            importance_df = self._evaluate_feature_importance(standardized_features, labels)
            
            # 5. 基于特征重要性进行权重调整
            logger.info("基于特征重要性进行权重调整")
            
            # 获取特征重要性
            feature_importance_dict = importance_df.set_index('feature')['avg_importance'].to_dict()
            
            # 优化权重调整机制
            # 1. 计算特征重要性的归一化权重
            total_importance = sum(feature_importance_dict.values())
            normalized_importance_dict = {}
            max_importance = max(feature_importance_dict.values())
            min_importance = min(feature_importance_dict.values())
            
            # 2. 对特征重要性进行归一化处理
            for feature, importance in feature_importance_dict.items():
                # 使用归一化后的重要性，确保权重在合理范围内
                if max_importance != min_importance:
                    normalized_importance = (importance - min_importance) / (max_importance - min_importance)
                else:
                    normalized_importance = 1.0
                normalized_importance_dict[feature] = normalized_importance
            
            # 3. 使用非线性转换增强重要特征的权重影响
            # 例如，使用指数函数或平方根函数来增强重要特征的权重
            enhanced_importance_dict = {}
            for feature, importance in normalized_importance_dict.items():
                # 使用指数函数增强重要特征的权重影响
                # 指数参数可以根据实际情况调整，这里使用2.0
                enhanced_importance = importance ** 2.0
                enhanced_importance_dict[feature] = enhanced_importance
            
            # 4. 根据特征重要性对特征进行加权
            weighted_features = pd.DataFrame()
            for col in standardized_features.columns:
                if col in enhanced_importance_dict:
                    # 获取增强后的特征重要性
                    importance = enhanced_importance_dict[col]
                    
                    # 使用增强后的重要性作为权重系数，对特征进行加权
                    weighted_features[col] = standardized_features[col] * importance
                    
                    logger.info(f"特征 {col} 的原始重要性：{feature_importance_dict[col]:.4f}，归一化后：{normalized_importance_dict[col]:.4f}，增强后：{importance:.4f}")
                else:
                    # 如果特征不在重要性字典中，使用原始特征
                    weighted_features[col] = standardized_features[col]
            
            # 合并标签列
            weighted_features['label'] = labels
            
            # 更新标准化特征为加权特征
            standardized_features = weighted_features
        
        return standardized_features
    
    def _evaluate_feature_importance(self, features, labels):
        """评估特征重要性"""
        logger.info("评估特征重要性")
        
        try:
            # 1. 随机森林特征重要性
            logger.info("使用随机森林评估特征重要性")
            from sklearn.ensemble import RandomForestClassifier
            
            # 初始化模型
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
            # 训练模型
            rf_model.fit(features, labels)
            
            # 获取特征重要性
            rf_importance = rf_model.feature_importances_
            
            # 2. XGBoost特征重要性
            logger.info("使用XGBoost评估特征重要性")
            try:
                from xgboost import XGBClassifier
                
                # 初始化模型
                xgb_model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                
                # 训练模型
                xgb_model.fit(features, labels)
                
                # 获取特征重要性
                xgb_importance = xgb_model.feature_importances_
            except ImportError:
                logger.warning("XGBoost库未安装，跳过XGBoost特征重要性评估")
                xgb_importance = None
            except Exception as e:
                logger.error(f"XGBoost特征重要性评估失败：{e}")
                xgb_importance = None
            
            # 3. LightGBM特征重要性
            logger.info("使用LightGBM评估特征重要性")
            try:
                from lightgbm import LGBMClassifier
                
                # 初始化模型
                lgbm_model = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                
                # 训练模型
                lgbm_model.fit(features, labels)
                
                # 获取特征重要性
                lgbm_importance = lgbm_model.feature_importances_ / lgbm_model.feature_importances_.sum()  # 归一化
            except ImportError:
                logger.warning("LightGBM库未安装，跳过LightGBM特征重要性评估")
                lgbm_importance = None
            except Exception as e:
                logger.error(f"LightGBM特征重要性评估失败：{e}")
                lgbm_importance = None
            
            # 4. 基于Permutation的特征重要性
            logger.info("使用Permutation方法评估特征重要性")
            from sklearn.inspection import permutation_importance
            
            perm_importance = permutation_importance(rf_model, features, labels, 
                                                   n_repeats=10, random_state=42, n_jobs=-1)
            perm_importance_mean = perm_importance.importances_mean
            
            # 5. SHAP值解释（如果安装了SHAP库）
            logger.info("使用SHAP值解释特征重要性")
            shap_values = None
            shap_importance = None
            try:
                import shap
                # 使用TreeExplainer解释树模型，优先使用XGBoost模型
                if xgb_importance is not None:
                    explainer = shap.TreeExplainer(xgb_model)
                    shap_values = explainer.shap_values(features)
                else:
                    explainer = shap.TreeExplainer(rf_model)
                    shap_values = explainer.shap_values(features)
                
                # 计算SHAP重要性（绝对值的均值）
                if isinstance(shap_values, list):
                    # 对于多分类问题，取第1类（涨停）的SHAP值
                    shap_importance = np.abs(shap_values[1]).mean(axis=0)
                else:
                    shap_importance = np.abs(shap_values).mean(axis=0)
                
                logger.info("SHAP值解释完成")
            except ImportError:
                logger.warning("SHAP库未安装，跳过SHAP值解释")
            except Exception as e:
                logger.error(f"SHAP值解释失败：{e}")
            
            # 创建综合特征重要性DataFrame
            importance_data = {
                'feature': features.columns,
                'rf_importance': rf_importance,
                'perm_importance': perm_importance_mean
            }
            
            if xgb_importance is not None:
                importance_data['xgb_importance'] = xgb_importance
            
            if lgbm_importance is not None:
                importance_data['lgbm_importance'] = lgbm_importance
            
            if shap_importance is not None:
                importance_data['shap_importance'] = shap_importance
            
            importance_df = pd.DataFrame(importance_data)
            
            # 计算平均重要性
            importance_cols = [col for col in importance_df.columns if 'importance' in col]
            importance_df['avg_importance'] = importance_df[importance_cols].mean(axis=1)
            
            # 按平均重要性排序
            importance_df = importance_df.sort_values('avg_importance', ascending=False)
            
            # 保存特征重要性结果
            self._save_feature_importance(importance_df)
            
            logger.info("特征重要性评估完成")
            logger.info(f"重要性前10的特征：\n{importance_df[['feature', 'avg_importance']].head(10)}")
            
            # 保存SHAP值可视化（如果可用）
            if shap_values is not None:
                self._save_shap_visualization(shap_values, features)
            
            # 返回特征重要性DataFrame
            return importance_df
            
        except Exception as e:
            logger.error(f"特征重要性评估失败：{e}")
            # 发生错误时返回空的DataFrame
            return pd.DataFrame()
    
    def _save_shap_visualization(self, shap_values, features):
        """保存SHAP值可视化结果"""
        logger.info("保存SHAP值可视化结果")
        
        try:
            import shap
            import matplotlib.pyplot as plt
            
            # 创建保存目录
            import os
            from datetime import datetime
            shap_dir = os.path.join(config.FEATURE_DATA_PATH, 'shap')
            os.makedirs(shap_dir, exist_ok=True)
            
            # 1. 保存SHAP摘要图
            plt.figure(figsize=(10, 6))
            if isinstance(shap_values, list):
                # 多分类问题，取第1类（涨停）的SHAP值
                shap.summary_plot(shap_values[1], features, show=False)
            else:
                shap.summary_plot(shap_values, features, show=False)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_plot_path = os.path.join(shap_dir, f"shap_summary_{timestamp}.png")
            plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SHAP摘要图保存成功：{summary_plot_path}")
            
            # 2. 保存SHAP依赖图（前3个重要特征）
            for i, feature in enumerate(features.columns[:3]):
                plt.figure(figsize=(10, 6))
                if isinstance(shap_values, list):
                    shap.dependence_plot(feature, shap_values[1], features, show=False)
                else:
                    shap.dependence_plot(feature, shap_values, features, show=False)
                
                dependence_plot_path = os.path.join(shap_dir, f"shap_dependence_{feature}_{timestamp}.png")
                plt.savefig(dependence_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"SHAP依赖图保存成功：{dependence_plot_path}")
                
        except Exception as e:
            logger.error(f"保存SHAP可视化结果失败：{e}")
    
    def _save_feature_importance(self, importance_df):
        """保存特征重要性结果"""
        logger.info("保存特征重要性结果")
        
        import os
        from datetime import datetime
        
        # 创建保存目录
        importance_dir = os.path.join(config.FEATURE_DATA_PATH, 'importance')
        os.makedirs(importance_dir, exist_ok=True)
        
        # 保存数据
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(importance_dir, f"feature_importance_{timestamp}.csv")
        
        try:
            importance_df.to_csv(file_path, index=False)
            logger.info(f"特征重要性结果保存成功：{file_path}")
        except Exception as e:
            logger.error(f"保存特征重要性结果失败：{e}")
    
    def _save_feature_data(self, features):
        """保存特征数据"""
        logger.info(f"保存特征数据到本地，路径：{config.FEATURE_DATA_PATH}")
        
        # 创建保存目录
        import os
        os.makedirs(config.FEATURE_DATA_PATH, exist_ok=True)
        
        # 保存数据
        file_path = os.path.join(config.FEATURE_DATA_PATH, f"features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        try:
            pd.to_pickle(features, file_path)
            logger.info(f"特征数据保存成功：{file_path}")
        except Exception as e:
            logger.error(f"保存特征数据失败：{e}")
