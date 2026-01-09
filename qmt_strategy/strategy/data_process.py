#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理模块
"""

import pandas as pd
import numpy as np
from qmt_strategy.utils.logger import setup_logger
import qmt_strategy.config as config

logger = setup_logger()

class DataProcessor:
    """数据预处理类，用于清洗、转换和整合数据"""
    
    def __init__(self):
        """初始化数据预处理器"""
        logger.info("初始化数据预处理器")
    
    def _filter_stock_pool(self, data):
        """股票池筛选，排除ST股票和北交所股票"""
        logger.info("开始股票池筛选")
        
        if isinstance(data, dict):
            # 字典格式，键为股票代码
            filtered_data = {}
            for stock_code, stock_data in data.items():
                # 排除北交所股票（代码以43或83开头）
                if stock_code.startswith('43') or stock_code.startswith('83'):
                    logger.info(f"排除北交所股票：{stock_code}")
                    continue
                
                # 排除ST股票（名称包含ST）
                if isinstance(stock_data, dict) and 'name' in stock_data:
                    if 'ST' in stock_data['name']:
                        logger.info(f"排除ST股票：{stock_code} - {stock_data['name']}")
                        continue
                
                filtered_data[stock_code] = stock_data
            
            logger.info(f"股票池筛选完成，保留{len(filtered_data)}只股票")
            return filtered_data
        elif isinstance(data, pd.DataFrame):
            # DataFrame格式
            # 排除北交所股票（代码以43或83开头）
            data = data[~data['stock_code'].str.startswith('43') & ~data['stock_code'].str.startswith('83')]
            
            # 如果有股票名称列，排除ST股票
            if 'name' in data.columns:
                data = data[~data['name'].str.contains('ST')]
            
            logger.info(f"股票池筛选完成，保留{len(data)}行数据")
            return data
        else:
            logger.warning("不支持的数据格式，跳过股票池筛选")
            return data
    
    def process(self, data):
        """数据预处理主函数"""
        logger.info("开始数据预处理")
        
        if data is None:
            logger.warning("输入数据为空，跳过预处理")
            return None
        
        try:
            # 股票池筛选，排除ST股票和北交所股票
            filtered_data = self._filter_stock_pool(data)
            
            # 如果数据是字典格式（股票代码为键，股票数据为值），则转换为DataFrame
            if isinstance(filtered_data, dict):
                logger.info("将字典格式数据转换为DataFrame")
                filtered_data = self._convert_dict_to_dataframe(filtered_data)
            
            # 数据清洗
            cleaned_data = self._clean_data(filtered_data)
            
            # 数据转换
            transformed_data = self._transform_data(cleaned_data)
            
            # 数据整合
            integrated_data = self._integrate_data(transformed_data)
            
            # 保存预处理后的数据
            self._save_processed_data(integrated_data)
            
            logger.info("数据预处理完成")
            return integrated_data
            
        except Exception as e:
            logger.error(f"数据预处理失败：{e}")
            return None
    
    def _clean_data(self, data):
        """数据清洗"""
        logger.info("执行数据清洗")
        
        cleaned_data = data.copy()
        
        # 处理缺失值
        logger.info("处理缺失值")
        if isinstance(cleaned_data, pd.DataFrame):
            # 删除全部为空的行和列
            cleaned_data = cleaned_data.dropna(how='all')
            cleaned_data = cleaned_data.dropna(axis=1, how='all')
            
            # 填充数值列的缺失值
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                # 对价格相关列使用前向填充，保留趋势
                if col in ['open', 'high', 'low', 'close', 'prev_close', 'bid_price_915', 'bid_price_920', 'bid_price_925']:
                    cleaned_data[col] = cleaned_data[col].ffill()
                    cleaned_data[col] = cleaned_data[col].bfill()
                # 对成交量、成交额等使用0填充
                elif col in ['volume', 'amount', 'bid_volume_915', 'bid_volume_920', 'bid_volume_925', 'buy_order_size_925']:
                    cleaned_data[col] = cleaned_data[col].fillna(0)
                # 对其他数值列使用均值填充
                else:
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mean())
            
            # 处理非数值列的缺失值
            non_numeric_cols = cleaned_data.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode().iloc[0] if not cleaned_data[col].mode().empty else '')
        
        # 处理异常值
        logger.info("处理异常值")
        if isinstance(cleaned_data, pd.DataFrame):
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if cleaned_data[col].isna().all():
                    continue  # 跳过全部为NaN的列
                    
                # 价格相关列特殊处理
                if col in ['open', 'high', 'low', 'close', 'prev_close', 'bid_price_915', 'bid_price_920', 'bid_price_925']:
                    # 使用IQR方法检测异常值，但保留一定的价格波动空间
                    Q1 = cleaned_data[col].quantile(0.01)  # 更宽松的分位数
                    Q3 = cleaned_data[col].quantile(0.99)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR  # 更宽松的异常值范围
                    upper_bound = Q3 + 3 * IQR
                
                # 成交量、成交额相关列特殊处理
                elif col in ['volume', 'amount', 'bid_volume_915', 'bid_volume_920', 'bid_volume_925', 'buy_order_size_925']:
                    # 成交量不可能为负，设置下限为0
                    lower_bound = 0
                    # 使用99.9%分位数作为上限
                    upper_bound = cleaned_data[col].quantile(0.999)
                
                # 其他数值列使用标准IQR方法
                else:
                    Q1 = cleaned_data[col].quantile(0.25)
                    Q3 = cleaned_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                
                # 将异常值替换为上下界
                cleaned_data[col] = cleaned_data[col].clip(lower=lower_bound, upper=upper_bound)
                
                # 确保价格的合理性：最高价 >= 收盘价 >= 开盘价 >= 最低价
                if {'open', 'high', 'low', 'close'}.issubset(cleaned_data.columns):
                    # 确保最高价不低于其他价格
                    cleaned_data['high'] = cleaned_data[['open', 'high', 'low', 'close']].max(axis=1)
                    # 确保最低价不高于其他价格
                    cleaned_data['low'] = cleaned_data[['open', 'high', 'low', 'close']].min(axis=1)
                    # 确保收盘价在最高价和最低价之间
                    cleaned_data['close'] = cleaned_data['close'].clip(lower=cleaned_data['low'], upper=cleaned_data['high'])
                    # 确保开盘价在最高价和最低价之间
                    cleaned_data['open'] = cleaned_data['open'].clip(lower=cleaned_data['low'], upper=cleaned_data['high'])
        
        return cleaned_data
    
    def _transform_data(self, data):
        """数据转换"""
        logger.info("执行数据转换")
        
        transformed_data = data.copy()
        
        # 转换数据类型
        logger.info("转换数据类型")
        if isinstance(transformed_data, pd.DataFrame):
            # 将日期列转换为datetime类型
            for col in transformed_data.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    transformed_data[col] = pd.to_datetime(transformed_data[col])
            
            # 确保数值列的数据类型正确
            numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
            transformed_data[numeric_cols] = transformed_data[numeric_cols].astype(np.float64)
        
        # 计算衍生指标
        logger.info("计算衍生指标")
        if isinstance(transformed_data, pd.DataFrame):
            # 计算涨跌幅
            if 'close' in transformed_data.columns:
                transformed_data['change'] = transformed_data['close'].pct_change()
            
            # 计算换手率（如果有流通股本数据）
            if 'volume' in transformed_data.columns and 'circulating_cap' in transformed_data.columns:
                transformed_data['turnover_rate'] = transformed_data['volume'] / transformed_data['circulating_cap'] * 100
        
        return transformed_data
    
    def _integrate_data(self, data):
        """数据整合"""
        logger.info("执行数据整合")
        
        # 这里根据实际数据结构进行整合
        # 例如，将不同来源的数据合并为一个统一的DataFrame
        integrated_data = data
        
        return integrated_data
    
    def _convert_dict_to_dataframe(self, data_dict):
        """
        将字典格式的数据转换为DataFrame格式
        
        Args:
            data_dict: 字典格式的数据，{stock_code: {field: values}}
            
        Returns:
            pd.DataFrame: 转换后的DataFrame
        """
        logger.info(f"转换字典格式数据，包含{len(data_dict)}只股票")
        
        all_stock_data = []
        
        for stock_code, stock_data in data_dict.items():
            # 如果stock_data是字典格式（field: values），则转换为DataFrame
            if isinstance(stock_data, dict):
                # 转换为DataFrame
                df = pd.DataFrame(stock_data)
                # 添加股票代码列
                df['stock_code'] = stock_code
                # 根据真实涨停规则生成label列（1表示涨停，0表示非涨停）
                # 计算涨跌幅
                df['prev_close'] = df['close'].shift(1)  # 前一日收盘价
                df['change_rate'] = (df['close'] - df['prev_close']) / df['prev_close'] * 100
                
                # 根据涨停规则生成label
                # 普通股票涨停幅度为10%，ST股票为5%，科创板为20%
                # 这里简化处理，默认使用10%的涨停幅度
                # 实际应用中应根据股票类型调整涨停幅度
                df['label'] = 0
                # 考虑涨停幅度的正负（涨停为正，跌停为负）
                df.loc[df['change_rate'] >= 9.95, 'label'] = 1  # 允许一定误差，9.95%以上视为涨停
                
                # 处理首日上市股票（无prev_close）
                df.loc[df['prev_close'].isna(), 'label'] = 0
                # 添加到列表中
                all_stock_data.append(df)
        
        if not all_stock_data:
            logger.warning("没有可转换的数据")
            return pd.DataFrame()
        
        # 合并所有股票数据
        combined_data = pd.concat(all_stock_data, ignore_index=True)
        
        # 添加一些模拟的特征列，用于特征工程
        # 模拟竞价数据
        combined_data['bid_price_915'] = combined_data['open'] * np.random.normal(1, 0.01, len(combined_data))
        combined_data['bid_price_920'] = combined_data['open'] * np.random.normal(1, 0.01, len(combined_data))
        combined_data['bid_price_925'] = combined_data['open']
        combined_data['bid_volume_915'] = combined_data['volume'] * np.random.uniform(0.5, 1.0, len(combined_data))
        combined_data['bid_volume_920'] = combined_data['volume'] * np.random.uniform(0.7, 1.2, len(combined_data))
        combined_data['bid_volume_925'] = combined_data['volume'] * np.random.uniform(0.8, 1.5, len(combined_data))
        combined_data['buy_order_size_925'] = combined_data['volume'] * np.random.uniform(0.1, 0.5, len(combined_data))
        
        # 模拟板块数据
        combined_data['sector_change'] = np.random.normal(0, 0.02, len(combined_data))
        combined_data['sector_money_flow'] = np.random.uniform(-100000000, 100000000, len(combined_data))
        combined_data['sector_rank'] = np.random.randint(1, 100, len(combined_data))
        combined_data['sector_limit_up_count'] = np.random.randint(0, 20, len(combined_data))
        
        # 模拟市场情绪数据
        combined_data['index_change'] = np.random.normal(0, 0.01, len(combined_data))
        combined_data['up_down_ratio'] = np.random.uniform(0.5, 2.0, len(combined_data))
        combined_data['profit_effect'] = np.random.normal(0, 0.1, len(combined_data))
        combined_data['market_volume'] = np.random.uniform(1000000000, 10000000000, len(combined_data))
        combined_data['market_volume_ratio'] = np.random.uniform(0.5, 2.0, len(combined_data))
        
        # 模拟基本面数据
        combined_data['circulating_market_cap'] = np.random.uniform(1000000000, 10000000000, len(combined_data))
        combined_data['pe_ratio'] = np.random.uniform(10, 50, len(combined_data))
        combined_data['pb_ratio'] = np.random.uniform(1, 5, len(combined_data))
        combined_data['earnings_announcement'] = np.random.choice([0, 1], size=len(combined_data), p=[0.95, 0.05])
        combined_data['turnover_rate'] = combined_data['volume'] / combined_data['circulating_market_cap'] * 100
        combined_data['volatility_5d'] = np.random.normal(0, 0.05, len(combined_data))
        
        # 模拟技术指标
        combined_data['ma5'] = combined_data['close'].rolling(window=5).mean()
        combined_data['ma10'] = combined_data['close'].rolling(window=10).mean()
        combined_data['macd'] = np.random.normal(0, 0.1, len(combined_data))
        combined_data['macd_signal'] = np.random.normal(0, 0.1, len(combined_data))
        combined_data['rsi_14'] = np.random.uniform(30, 70, len(combined_data))
        combined_data['volume_ratio'] = np.random.uniform(0.5, 2.0, len(combined_data))
        
        logger.info(f"数据转换完成，共{len(combined_data)}行数据")
        return combined_data
    
    def _save_processed_data(self, data):
        """保存预处理后的数据"""
        logger.info(f"保存预处理后的数据到本地，路径：{config.PROCESSED_DATA_PATH}")
        
        # 创建保存目录
        import os
        os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
        
        # 保存数据
        file_path = os.path.join(config.PROCESSED_DATA_PATH, f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        try:
            pd.to_pickle(data, file_path)
            logger.info(f"预处理后的数据保存成功：{file_path}")
        except Exception as e:
            logger.error(f"保存预处理后的数据失败：{e}")
    
    def process_bid_data(self, bid_data):
        """处理集合竞价数据"""
        logger.info("处理集合竞价数据")
        
        if bid_data is None:
            logger.warning("集合竞价数据为空，跳过处理")
            return None
        
        try:
            # 清洗集合竞价数据
            cleaned_bid_data = self._clean_data(bid_data)
            
            # 转换集合竞价数据
            transformed_bid_data = self._transform_bid_data(cleaned_bid_data)
            
            logger.info("集合竞价数据处理完成")
            return transformed_bid_data
            
        except Exception as e:
            logger.error(f"处理集合竞价数据失败：{e}")
            return None
    
    def _transform_bid_data(self, bid_data):
        """转换集合竞价数据"""
        logger.info("转换集合竞价数据")
        
        transformed_bid_data = bid_data.copy()
        
        # 计算竞价强度（9:20后不可撤单阶段的涨幅）
        if isinstance(transformed_bid_data, pd.DataFrame):
            # 假设数据中包含9:20和9:25的竞价价格
            if 'bid_price_920' in transformed_bid_data.columns and 'bid_price_925' in transformed_bid_data.columns:
                transformed_bid_data['bid_intensity'] = (transformed_bid_data['bid_price_925'] - transformed_bid_data['bid_price_920']) / transformed_bid_data['bid_price_920']
            
            # 计算委托单量比
            if 'bid_volume_925' in transformed_bid_data.columns and 'avg_volume_5d' in transformed_bid_data.columns:
                transformed_bid_data['bid_volume_ratio'] = transformed_bid_data['bid_volume_925'] / transformed_bid_data['avg_volume_5d']
        
        return transformed_bid_data
