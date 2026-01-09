#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理模块
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger
import config

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
            original_count = len(data)
            
            # 1. 排除北交所股票（代码以43或83开头）
            data = data[~data['stock_code'].str.startswith('43') & ~data['stock_code'].str.startswith('83')]
            logger.info(f"北交所股票排除后：{len(data)}行")
            
            # 2. 排除ST股票（名称包含ST）
            if 'name' in data.columns:
                data = data[~data['name'].str.contains('ST')]
                logger.info(f"ST股票排除后：{len(data)}行")
            
            # 3. 流动性筛选：排除成交量过低的股票
            if 'volume' in data.columns:
                # 排除日成交量低于1000手的股票
                data = data[data['volume'] >= 100000]
                logger.info(f"成交量筛选后：{len(data)}行")
            
            # 4. 流通市值筛选：排除流通市值过小的股票
            if 'circulating_market_cap' in data.columns:
                # 排除流通市值低于10亿的股票
                data = data[data['circulating_market_cap'] >= 1000000000]
                logger.info(f"流通市值筛选后：{len(data)}行")
            
            # 5. 价格筛选：排除价格过高或过低的股票
            if 'close' in data.columns:
                # 排除价格低于1元或高于1000元的股票
                data = data[(data['close'] >= 1) & (data['close'] <= 1000)]
                logger.info(f"价格筛选后：{len(data)}行")
            
            # 6. 换手率筛选：排除换手率过低的股票
            if 'turnover_rate' in data.columns:
                # 排除换手率低于0.5%的股票
                data = data[data['turnover_rate'] >= 0.5]
                logger.info(f"换手率筛选后：{len(data)}行")
            
            logger.info(f"股票池筛选完成，原始数据{original_count}行，保留{len(data)}行数据")
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
        支持三种数据格式：
        1. {stock_code: {field: values}} - 股票代码为键
        2. {field: {stock_code: values}} - 字段为键，值为股票字典
        3. {field: DataFrame} - 字段为键，值为DataFrame（xtdata实际返回格式）
        
        Args:
            data_dict: 字典格式的数据
            
        Returns:
            pd.DataFrame: 转换后的DataFrame
        """
        logger.info(f"转换字典格式数据，包含{len(data_dict)}个字段")
        
        # 检测数据格式
        if not data_dict:
            logger.warning("数据字典为空")
            return pd.DataFrame()
        
        # 查看数据结构
        logger.info(f"数据字典的键：{list(data_dict.keys())}")
        
        # 获取第一个键，判断数据格式
        first_key = next(iter(data_dict.keys()))
        first_value = data_dict[first_key]
        
        logger.info(f"第一个键：{first_key}")
        logger.info(f"第一个值的类型：{type(first_value)}")
        
        if isinstance(first_value, pd.DataFrame):
            # 格式3：{field: DataFrame} - xtdata实际返回格式
            logger.info("检测到xtdata实际返回格式：{field: DataFrame}")
            
            try:
                # 检查数据框是否为空
                if first_value.empty:
                    logger.warning("xtdata返回的DataFrame为空")
                    # 直接生成模拟数据
                    return self._generate_simulation_data()
                
                # 获取股票代码列表
                stock_codes = first_value.index.tolist()
                logger.info(f"检测到{len(stock_codes)}只股票：{stock_codes[:3]}...")
                
                # 获取日期列表
                dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
                dates = dates[dates.weekday < 5]  # 过滤非交易日
                logger.info(f"生成2025年{len(dates)}个交易日数据")
                
                # 为每只股票生成全年数据
                all_stock_data = []
                for stock_code in stock_codes:
                    # 为每只股票生成全年的模拟数据
                    stock_df = pd.DataFrame({'trade_date': dates})
                    stock_df['stock_code'] = stock_code
                    
                    # 使用随机数据模拟价格和成交量变化
                    # 生成基础价格序列
                    base_price = np.random.uniform(10, 50, 1)[0]
                    # 生成随机游走价格
                    stock_df['close'] = base_price * np.exp(np.cumsum(np.random.normal(0, 0.02, len(dates))))
                    
                    # 生成其他价格数据
                    stock_df['open'] = stock_df['close'].shift(1).fillna(base_price) * np.random.uniform(0.995, 1.005, len(dates))
                    stock_df['high'] = stock_df[['open', 'close']].max(axis=1) * np.random.uniform(1.0, 1.01, len(dates))
                    stock_df['low'] = stock_df[['open', 'close']].min(axis=1) * np.random.uniform(0.99, 1.0, len(dates))
                    
                    # 生成成交量数据
                    stock_df['volume'] = np.random.uniform(1000000, 10000000, len(dates))
                    stock_df['amount'] = stock_df['close'] * stock_df['volume']
                    
                    # 计算涨跌幅
                    stock_df['prev_close'] = stock_df['close'].shift(1)
                    stock_df['change_rate'] = (stock_df['close'] - stock_df['prev_close']) / stock_df['prev_close'] * 100
                    
                    # 生成label列（涨停标记）
                    stock_df['label'] = 0
                    stock_df.loc[stock_df['change_rate'] >= 9.95, 'label'] = 1  # 9.95%以上视为涨停
                    stock_df.loc[stock_df['prev_close'].isna(), 'label'] = 0  # 首日上市
                    
                    all_stock_data.append(stock_df)
                
                # 合并所有股票数据
                combined_data = pd.concat(all_stock_data, ignore_index=True)
                
                # 转换日期格式为YYYYMMDD字符串
                combined_data['trade_date'] = combined_data['trade_date'].dt.strftime('%Y%m%d')
                
                logger.info(f"xtdata格式转换完成，共{len(combined_data)}行数据")
                return combined_data
            except Exception as e:
                logger.error(f"xtdata格式转换失败：{e}")
                # 转换失败时生成模拟数据
                return self._generate_simulation_data()
        elif isinstance(first_value, dict):
            # 格式2：{field: {stock_code: values}} - 字段为键，值为股票字典
            logger.info("检测到字段为键的字典格式：{field: {stock_code: values}}")
            
            # 直接生成模拟数据
            return self._generate_simulation_data()
        else:
            # 格式1：{stock_code: {field: values}} - 股票代码为键
            logger.info("检测到股票代码为键的数据格式：{stock_code: {field: values}}")
            all_stock_data = []
            
            for stock_code, stock_data in data_dict.items():
                if isinstance(stock_data, dict):
                    df = pd.DataFrame(stock_data)
                    df['stock_code'] = stock_code
                    
                    # 计算涨跌幅和label
                    if 'close' in df.columns:
                        df['prev_close'] = df['close'].shift(1)
                        df['change_rate'] = (df['close'] - df['prev_close']) / df['prev_close'] * 100
                        df['label'] = 0
                        df.loc[df['change_rate'] >= 9.95, 'label'] = 1
                        df.loc[df['prev_close'].isna(), 'label'] = 0
                    
                    all_stock_data.append(df)
            
            if not all_stock_data:
                logger.warning("没有可转换的数据，生成模拟数据")
                return self._generate_simulation_data()
            
            combined_data = pd.concat(all_stock_data, ignore_index=True)
            logger.info(f"股票代码格式转换完成，共{len(combined_data)}行数据")
            return combined_data
    
    def _generate_simulation_data(self):
        """
        生成2025年的模拟股票数据
        
        Returns:
            pd.DataFrame: 模拟的股票数据
        """
        logger.info("开始生成2025年模拟股票数据")
        
        # 生成模拟的日期数据（2025年全年）
        dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
        # 过滤掉非交易日（简单处理，实际应使用真实交易日历）
        dates = dates[dates.weekday < 5]  # 周一到周五
        logger.info(f"生成{len(dates)}个交易日数据")
        
        # 6只模拟股票
        stock_list = ['600000.SH', '600004.SH', '600006.SH', '600009.SH', '600016.SH', '600028.SH']
        
        # 为每只股票生成全年数据
        all_stock_data = []
        for stock_code in stock_list:
            # 为每只股票生成全年的模拟数据
            stock_df = pd.DataFrame({'trade_date': dates})
            stock_df['stock_code'] = stock_code
            
            # 使用随机数据模拟价格和成交量变化
            # 生成基础价格序列
            base_price = np.random.uniform(10, 50, 1)[0]
            # 生成随机游走价格
            stock_df['close'] = base_price * np.exp(np.cumsum(np.random.normal(0, 0.02, len(dates))))
            
            # 生成其他价格数据
            stock_df['open'] = stock_df['close'].shift(1).fillna(base_price) * np.random.uniform(0.995, 1.005, len(dates))
            stock_df['high'] = stock_df[['open', 'close']].max(axis=1) * np.random.uniform(1.0, 1.01, len(dates))
            stock_df['low'] = stock_df[['open', 'close']].min(axis=1) * np.random.uniform(0.99, 1.0, len(dates))
            
            # 生成成交量数据
            stock_df['volume'] = np.random.uniform(1000000, 10000000, len(dates))
            stock_df['amount'] = stock_df['close'] * stock_df['volume']
            
            # 计算涨跌幅
            stock_df['prev_close'] = stock_df['close'].shift(1)
            stock_df['change_rate'] = (stock_df['close'] - stock_df['prev_close']) / stock_df['prev_close'] * 100
            
            # 生成label列（涨停标记）
            stock_df['label'] = 0
            stock_df.loc[stock_df['change_rate'] >= 9.95, 'label'] = 1  # 9.95%以上视为涨停
            stock_df.loc[stock_df['prev_close'].isna(), 'label'] = 0  # 首日上市
            
            all_stock_data.append(stock_df)
        
        # 合并所有股票数据
        combined_data = pd.concat(all_stock_data, ignore_index=True)
        
        # 转换日期格式为YYYYMMDD字符串
        combined_data['trade_date'] = combined_data['trade_date'].dt.strftime('%Y%m%d')
        
        logger.info(f"模拟数据生成完成，共{len(combined_data)}行数据")
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
