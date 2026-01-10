#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
决策建议生成模块
整合各模块分析结果，形成系统化的投资决策建议
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from utils.logger import setup_logger

logger = setup_logger()


class DecisionGenerator:
    """决策建议生成器，用于整合各模块分析结果，形成系统化的投资决策建议"""
    
    def __init__(self):
        """初始化决策建议生成器
        
        决策建议生成器负责整合各模块分析结果，生成系统化的投资决策建议，包括：
        - 个股买入优先级
        - 建议买入价格区间
        - 仓位配置比例
        - 止盈止损点位
        - 具体操作指引
        """
        logger.info("初始化决策建议生成器")
        
        # 明确定义决策权重配置
        self.decision_weights = {
            'prediction_score': 0.40,  # 模型预测分数权重
            'stock_score': 0.35,  # 个股综合评分权重
            'risk_score': 0.25  # 风险评分权重（风险越低，分数越高）
        }
        
        # 明确定义仓位配置参数
        self.position_config = {
            'max_single_position': 0.20,  # 单票最大仓位比例：20%
            'max_total_position': 0.80,  # 总仓位最大比例：80%
            'base_position': 0.05,  # 基础仓位比例：5%
            'position_adjustment_factor': 0.15  # 仓位调整因子：15%
        }
        
        # 明确定义止盈止损参数
        self.profit_loss_config = {
            'base_profit_target': 0.15,  # 基础止盈目标：15%
            'base_stop_loss': 0.06,  # 基础止损比例：6%
            'trailing_stop_ratio': 0.08,  # 基础跟踪止损比例：8%
            'profit_target_adjustment': 0.05,  # 止盈目标调整幅度：5%
            'stop_loss_adjustment': 0.02  # 止损比例调整幅度：2%
        }
    
    def generate_decisions(self, evaluated_stocks: pd.DataFrame) -> pd.DataFrame:
        """生成投资决策建议
        
        Args:
            evaluated_stocks: 经过各模块评估的股票数据
            
        Returns:
            带有决策建议的股票数据
        """
        logger.info("开始生成投资决策建议")
        
        if evaluated_stocks.empty:
            logger.error("输入股票数据为空，无法生成决策建议")
            return pd.DataFrame()
        
        # 1. 计算综合决策得分
        evaluated_stocks = self._calculate_decision_score(evaluated_stocks)
        
        # 2. 生成买入优先级
        evaluated_stocks = self._generate_buy_priority(evaluated_stocks)
        
        # 3. 计算建议买入价格区间
        evaluated_stocks = self._calculate_buy_price_range(evaluated_stocks)
        
        # 4. 计算仓位配置比例
        evaluated_stocks = self._calculate_position_size(evaluated_stocks)
        
        # 5. 计算止盈止损点位
        evaluated_stocks = self._calculate_profit_loss_levels(evaluated_stocks)
        
        # 6. 计算动态止盈止损调整
        evaluated_stocks = self._calculate_dynamic_profit_loss_adjustment(evaluated_stocks)
        
        # 7. 生成完整的决策建议
        decisions = self._generate_final_decisions(evaluated_stocks)
        
        logger.info("投资决策建议生成完成")
        return decisions
    
    def _get_dynamic_decision_weights(self, evaluated_stocks: pd.DataFrame) -> dict:
        """获取动态决策权重
        
        Args:
            evaluated_stocks: 经过各模块评估的股票数据
            
        Returns:
            动态决策权重字典，根据市场环境和股票特性动态调整
        """
        logger.info("计算动态决策权重")
        
        # 计算市场环境指标
        # 1. 平均风险得分
        avg_risk_score = (1 - evaluated_stocks['total_risk']).mean()
        
        # 2. 平均预测得分
        avg_prediction_score = evaluated_stocks['prediction_score'].mean()
        
        # 3. 预测得分标准差（反映预测一致性）
        pred_score_std = evaluated_stocks['prediction_score'].std()
        
        # 4. 平均封板概率
        avg_limit_up_prob = evaluated_stocks['limit_up_prob'].mean() if 'limit_up_prob' in evaluated_stocks.columns else 0.5
        
        # 5. 平均风险等级分布
        risk_level_counts = evaluated_stocks['risk_level'].value_counts(normalize=True)
        high_risk_ratio = risk_level_counts.get('高风险', 0)
        low_risk_ratio = risk_level_counts.get('低风险', 0)
        
        # 初始化基础权重
        weights = {
            'prediction_score': 0.40,
            'stock_score': 0.35,
            'risk_score': 0.25
        }
        
        # 根据市场整体风险调整权重
        logger.info(f"市场平均风险得分：{avg_risk_score:.2f}")
        logger.info(f"高风险股票比例：{high_risk_ratio:.2%}")
        logger.info(f"低风险股票比例：{low_risk_ratio:.2%}")
        
        # 如果整体风险较高，增加风险得分权重
        if avg_risk_score < 0.55 or high_risk_ratio > 0.4:
            weights['risk_score'] += 0.15
            weights['prediction_score'] -= 0.10
            weights['stock_score'] -= 0.05
            logger.info("市场风险较高，增加风险得分权重")
        elif avg_risk_score > 0.85 or low_risk_ratio > 0.6:
            # 整体风险较低，减少风险得分权重，增加其他权重
            weights['risk_score'] -= 0.10
            weights['prediction_score'] += 0.05
            weights['stock_score'] += 0.05
            logger.info("市场风险较低，减少风险得分权重，增加其他权重")
        
        # 根据预测质量调整权重
        logger.info(f"平均预测得分：{avg_prediction_score:.2f}")
        logger.info(f"预测得分标准差：{pred_score_std:.2f}")
        
        # 如果预测得分较高且一致性好，增加预测得分权重
        if avg_prediction_score > 0.75 and pred_score_std < 0.15:
            weights['prediction_score'] += 0.10
            weights['stock_score'] -= 0.10
            logger.info("预测质量较高且一致性好，增加预测得分权重")
        # 如果预测得分较低且一致性差，减少预测得分权重
        elif avg_prediction_score < 0.55 or pred_score_std > 0.3:
            weights['prediction_score'] -= 0.05
            weights['stock_score'] += 0.03
            weights['risk_score'] += 0.02
            logger.info("预测质量较低或一致性差，减少预测得分权重")
        
        # 根据平均封板概率调整权重
        logger.info(f"平均封板概率：{avg_limit_up_prob:.2f}")
        
        if avg_limit_up_prob > 0.7:
            # 整体封板概率较高，增加预测得分权重
            weights['prediction_score'] += 0.05
            weights['stock_score'] -= 0.05
            logger.info("整体封板概率较高，增加预测得分权重")
        elif avg_limit_up_prob < 0.4:
            # 整体封板概率较低，减少预测得分权重
            weights['prediction_score'] -= 0.05
            weights['risk_score'] += 0.05
            logger.info("整体封板概率较低，减少预测得分权重")
        
        # 确保所有权重为正值
        for key in weights:
            weights[key] = max(0.05, weights[key])
        
        # 确保权重之和为1
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        logger.info(f"动态决策权重：{weights}")
        return weights
    
    def _calculate_decision_score(self, evaluated_stocks: pd.DataFrame) -> pd.DataFrame:
        """计算综合决策得分
        
        Args:
            evaluated_stocks: 经过各模块评估的股票数据
            
        Returns:
            带有综合决策得分的股票数据
        """
        logger.info("计算综合决策得分")
        
        # 1. 计算风险分数（风险越低，分数越高）
        evaluated_stocks['risk_score'] = 1 - evaluated_stocks['total_risk']
        
        # 2. 归一化各评分指标
        def normalize_column(df, column):
            min_val = df[column].min()
            max_val = df[column].max()
            if max_val - min_val == 0:
                return df[column] * 0 + 0.5
            return (df[column] - min_val) / (max_val - min_val)
        
        # 归一化预测得分
        evaluated_stocks['normalized_prediction_score'] = normalize_column(evaluated_stocks, 'prediction_score')
        
        # 处理股票评分 - 如果stock_score不存在，使用prediction_score作为替代
        if 'stock_score' in evaluated_stocks.columns:
            evaluated_stocks['normalized_stock_score'] = normalize_column(evaluated_stocks, 'stock_score')
        else:
            logger.warning("stock_score列不存在，使用prediction_score作为替代")
            evaluated_stocks['normalized_stock_score'] = evaluated_stocks['normalized_prediction_score']
        
        # 归一化风险评分
        evaluated_stocks['normalized_risk_score'] = normalize_column(evaluated_stocks, 'risk_score')
        
        # 3. 获取动态决策权重
        dynamic_weights = self._get_dynamic_decision_weights(evaluated_stocks)
        
        # 4. 计算综合决策得分
        evaluated_stocks['decision_score'] = (
            evaluated_stocks['normalized_prediction_score'] * dynamic_weights['prediction_score'] +
            evaluated_stocks['normalized_stock_score'] * dynamic_weights['stock_score'] +
            evaluated_stocks['normalized_risk_score'] * dynamic_weights['risk_score']
        )
        
        logger.info(f"综合决策得分计算完成，平均得分：{evaluated_stocks['decision_score'].mean():.2f}")
        return evaluated_stocks
    
    def _generate_buy_priority(self, evaluated_stocks: pd.DataFrame) -> pd.DataFrame:
        """生成买入优先级
        
        Args:
            evaluated_stocks: 带有综合决策得分的股票数据
            
        Returns:
            带有买入优先级的股票数据
        """
        logger.info("生成买入优先级")
        
        # 按综合决策得分降序排序
        evaluated_stocks = evaluated_stocks.sort_values(by='decision_score', ascending=False)
        
        # 添加买入优先级
        evaluated_stocks['buy_priority'] = range(1, len(evaluated_stocks) + 1)
        
        logger.info("买入优先级生成完成")
        return evaluated_stocks
    
    def _calculate_buy_price_range(self, evaluated_stocks: pd.DataFrame) -> pd.DataFrame:
        """计算建议买入价格区间
        
        Args:
            evaluated_stocks: 带有买入优先级的股票数据
            
        Returns:
            带有买入价格区间的股票数据
        """
        logger.info("计算建议买入价格区间")
        
        # 1. 基于竞价价格计算买入价格区间
        # 假设使用竞价价格作为基准
        if 'price_bid_price' in evaluated_stocks.columns:
            base_price = evaluated_stocks['price_bid_price']
        elif 'open' in evaluated_stocks.columns:
            base_price = evaluated_stocks['open']
        else:
            logger.warning("没有找到合适的基准价格，使用默认价格")
            base_price = pd.Series([100] * len(evaluated_stocks), index=evaluated_stocks.index)
        
        # 2. 计算买入价格区间
        # 买入价格区间 = [基准价格 * (1 - 0.5%), 基准价格 * (1 + 1%)]
        evaluated_stocks['buy_price_lower'] = base_price * 0.995
        evaluated_stocks['buy_price_upper'] = base_price * 1.01
        
        # 3. 四舍五入到两位小数
        evaluated_stocks['buy_price_lower'] = evaluated_stocks['buy_price_lower'].round(2)
        evaluated_stocks['buy_price_upper'] = evaluated_stocks['buy_price_upper'].round(2)
        
        logger.info("建议买入价格区间计算完成")
        return evaluated_stocks
    
    def _calculate_position_size(self, evaluated_stocks: pd.DataFrame) -> pd.DataFrame:
        """计算仓位配置比例
        
        Args:
            evaluated_stocks: 带有买入价格区间的股票数据
            
        Returns:
            带有仓位配置比例的股票数据，包括：
            - 仓位配置比例
            - 建议买入金额
            - 建议买入数量
            - 实际买入金额和仓位
        """
        logger.info("计算仓位配置比例")
        
        # 1. 基于多维度因素调整仓位
        def calculate_position(row):
            # 基础仓位
            position = self.position_config['base_position']
            
            # 根据决策得分调整仓位（得分越高，仓位越高）
            decision_score_factor = row['decision_score'] * self.position_config['position_adjustment_factor']
            position += decision_score_factor
            
            # 根据风险等级调整仓位（风险越低，仓位越高）
            risk_level_adjustment = {
                '低风险': 1.3,  # 低风险股票，仓位放大30%
                '中风险': 1.0,  # 中风险股票，仓位保持不变
                '高风险': 0.7   # 高风险股票，仓位缩小30%
            }.get(row['risk_level'], 1.0)
            position *= risk_level_adjustment
            
            # 根据封板概率调整仓位（封板概率越高，仓位越高）
            if 'limit_up_prob' in row:
                limit_up_prob = row['limit_up_prob']
                if limit_up_prob >= 0.9:
                    position *= 1.3  # 封板概率≥90%，仓位放大30%
                elif limit_up_prob >= 0.8:
                    position *= 1.2  # 封板概率≥80%，仓位放大20%
                elif limit_up_prob >= 0.7:
                    position *= 1.1  # 封板概率≥70%，仓位放大10%
                elif limit_up_prob < 0.4:
                    position *= 0.8  # 封板概率<40%，仓位缩小20%
                elif limit_up_prob < 0.2:
                    position *= 0.6  # 封板概率<20%，仓位缩小40%
            
            # 根据预测置信度调整仓位（置信度越高，仓位越高）
            if 'prediction_confidence' in row:
                confidence = row['prediction_confidence']
                if confidence >= 0.9:
                    position *= 1.2  # 置信度≥90%，仓位放大20%
                elif confidence >= 0.8:
                    position *= 1.1  # 置信度≥80%，仓位放大10%
                elif confidence < 0.6:
                    position *= 0.9  # 置信度<60%，仓位缩小10%
            
            # 根据资金流入强度调整仓位（资金流入越强，仓位越高）
            if 'fund_flow_net_flow' in row:
                net_flow = row['fund_flow_net_flow']
                # 资金净流入每增加1000万，仓位增加5%
                net_flow_factor = min(net_flow / 10000000 * 0.05, 0.3)  # 最大增加30%
                position *= (1 + net_flow_factor)
            
            # 根据成交量比调整仓位（成交量比越大，仓位越高）
            if 'volume_bid_volume_ratio' in row:
                volume_ratio = row['volume_bid_volume_ratio']
                if volume_ratio >= 5:
                    position *= 1.2  # 成交量比≥5，仓位放大20%
                elif volume_ratio >= 3:
                    position *= 1.1  # 成交量比≥3，仓位放大10%
                elif volume_ratio < 1.5:
                    position *= 0.9  # 成交量比<1.5，仓位缩小10%
            
            # 根据买入优先级调整仓位（优先级越高，仓位越高）
            priority_adjustment = 1.0 - (row['buy_priority'] - 1) * 0.05
            position *= priority_adjustment
            
            # 确保仓位在合理范围内
            position = max(0.01, min(position, self.position_config['max_single_position']))
            
            return position
        
        # 应用仓位计算逻辑
        evaluated_stocks['position_ratio'] = evaluated_stocks.apply(calculate_position, axis=1)
        
        # 2. 计算建议买入金额（基于初始资金）
        # 从配置文件中获取初始资金
        from config import INITIAL_CAPITAL
        initial_capital = INITIAL_CAPITAL
        logger.info(f"使用初始资金：{initial_capital}元")
        
        # 3. 确保总仓位不超过最大总仓位
        total_position = evaluated_stocks['position_ratio'].sum()
        max_total_position = self.position_config['max_total_position']
        
        if total_position > max_total_position:
            logger.info(f"总仓位{total_position:.2%}超过最大总仓位{max_total_position:.2%}，进行调整")
            # 按比例调整仓位
            adjustment_factor = max_total_position / total_position
            evaluated_stocks['position_ratio'] *= adjustment_factor
            logger.info(f"调整后总仓位：{evaluated_stocks['position_ratio'].sum():.2%}")
        
        evaluated_stocks['suggested_buy_amount'] = initial_capital * evaluated_stocks['position_ratio']
        
        # 4. 计算建议买入价格和数量
        # 建议买入价格：取买入价格区间的中间值
        evaluated_stocks['suggested_buy_price'] = (evaluated_stocks['buy_price_lower'] + evaluated_stocks['buy_price_upper']) / 2
        # 建议买入数量：根据建议买入金额和建议买入价格计算，按100股为单位
        evaluated_stocks['suggested_buy_quantity'] = (evaluated_stocks['suggested_buy_amount'] / evaluated_stocks['suggested_buy_price']).astype(int) // 100 * 100
        # 实际买入金额：根据建议买入数量和建议买入价格计算
        evaluated_stocks['actual_buy_amount'] = evaluated_stocks['suggested_buy_quantity'] * evaluated_stocks['suggested_buy_price']
        # 实际仓位比例：根据实际买入金额和初始资金计算
        evaluated_stocks['actual_position_ratio'] = evaluated_stocks['actual_buy_amount'] / initial_capital
        
        logger.info("仓位配置比例计算完成")
        return evaluated_stocks
    
    def _calculate_profit_loss_levels(self, evaluated_stocks: pd.DataFrame) -> pd.DataFrame:
        """计算止盈止损点位
        
        Args:
            evaluated_stocks: 带有仓位配置比例的股票数据
            
        Returns:
            带有止盈止损点位的股票数据，包括：
            - 止盈目标点位
            - 止损点位
            - 跟踪止损比例
            - 动态调整因子
        """
        logger.info("计算止盈止损点位")
        
        # 1. 基于多维度因素调整止盈止损参数
        def calculate_profit_loss(row):
            # 使用建议买入价格
            suggested_buy_price = row['suggested_buy_price']
            
            # 基础止盈目标和止损比例
            base_profit_target = self.profit_loss_config['base_profit_target']
            base_stop_loss = self.profit_loss_config['base_stop_loss']
            base_trailing_stop = self.profit_loss_config['trailing_stop_ratio']
            
            # 获取关键指标
            limit_up_prob = row['limit_up_prob']
            decision_score = row['decision_score']
            risk_level = row['risk_level']
            prediction_confidence = row['prediction_confidence'] if 'prediction_confidence' in row else 0.7
            
            # 基于封板概率调整止盈目标
            profit_target_adjustment = 0
            if limit_up_prob >= 0.95:
                profit_target_adjustment = self.profit_loss_config['profit_target_adjustment'] * 3  # 封板概率极高，大幅提高止盈目标
            elif limit_up_prob >= 0.9:
                profit_target_adjustment = self.profit_loss_config['profit_target_adjustment'] * 2.5  # 封板概率很高，大幅提高止盈目标
            elif limit_up_prob >= 0.85:
                profit_target_adjustment = self.profit_loss_config['profit_target_adjustment'] * 2  # 封板概率高，大幅提高止盈目标
            elif limit_up_prob >= 0.8:
                profit_target_adjustment = self.profit_loss_config['profit_target_adjustment'] * 1.5  # 封板概率较高，提高止盈目标
            elif limit_up_prob >= 0.7:
                profit_target_adjustment = self.profit_loss_config['profit_target_adjustment']  # 封板概率中等，提高止盈目标
            elif limit_up_prob < 0.4:
                profit_target_adjustment = -self.profit_loss_config['profit_target_adjustment'] * 0.5  # 封板概率低，降低止盈目标
            elif limit_up_prob < 0.2:
                profit_target_adjustment = -self.profit_loss_config['profit_target_adjustment']  # 封板概率极低，大幅降低止盈目标
            
            # 基于决策得分调整止盈目标
            if decision_score >= 0.95:
                profit_target_adjustment += self.profit_loss_config['profit_target_adjustment'] * 0.8  # 决策得分极高，大幅提高止盈目标
            elif decision_score >= 0.9:
                profit_target_adjustment += self.profit_loss_config['profit_target_adjustment'] * 0.5  # 决策得分高，小幅提高止盈目标
            elif decision_score >= 0.8:
                profit_target_adjustment += self.profit_loss_config['profit_target_adjustment'] * 0.3  # 决策得分较高，小幅提高止盈目标
            elif decision_score < 0.5:
                profit_target_adjustment -= self.profit_loss_config['profit_target_adjustment'] * 0.5  # 决策得分低，小幅降低止盈目标
            elif decision_score < 0.3:
                profit_target_adjustment -= self.profit_loss_config['profit_target_adjustment'] * 0.8  # 决策得分极低，大幅降低止盈目标
            
            # 基于预测置信度调整止盈目标
            if prediction_confidence >= 0.95:
                profit_target_adjustment += self.profit_loss_config['profit_target_adjustment'] * 0.5  # 置信度极高，提高止盈目标
            elif prediction_confidence < 0.6:
                profit_target_adjustment -= self.profit_loss_config['profit_target_adjustment'] * 0.5  # 置信度低，降低止盈目标
            
            # 基于风险等级调整止盈目标
            if risk_level == '高风险':
                profit_target_adjustment -= self.profit_loss_config['profit_target_adjustment'] * 0.5  # 高风险，降低止盈目标
            elif risk_level == '低风险':
                profit_target_adjustment += self.profit_loss_config['profit_target_adjustment'] * 0.3  # 低风险，提高止盈目标
            
            # 基于资金流入强度调整止盈目标
            if 'fund_flow_net_flow' in row:
                net_flow = row['fund_flow_net_flow']
                if net_flow > 50000000:  # 资金净流入超过5000万
                    profit_target_adjustment += self.profit_loss_config['profit_target_adjustment'] * 0.5  # 提高止盈目标
                elif net_flow < -10000000:  # 资金净流出超过1000万
                    profit_target_adjustment -= self.profit_loss_config['profit_target_adjustment'] * 0.3  # 降低止盈目标
            
            # 计算最终止盈目标
            profit_target = base_profit_target + profit_target_adjustment
            profit_target = max(0.03, min(0.5, profit_target))  # 限制止盈目标在3%-50%范围内
            
            # 基于风险等级调整止损比例
            stop_loss_adjustment = 0
            if risk_level == '高风险':
                stop_loss_adjustment = -self.profit_loss_config['stop_loss_adjustment'] * 1.5  # 高风险，大幅收紧止损
            elif risk_level == '低风险':
                stop_loss_adjustment = self.profit_loss_config['stop_loss_adjustment'] * 1.5  # 低风险，大幅放宽止损
            
            # 基于预测置信度调整止损比例
            if prediction_confidence >= 0.9:
                stop_loss_adjustment += self.profit_loss_config['stop_loss_adjustment']  # 置信度高，放宽止损
            elif prediction_confidence < 0.6:
                stop_loss_adjustment -= self.profit_loss_config['stop_loss_adjustment']  # 置信度低，收紧止损
            
            # 基于波动率风险调整止损比例
            if 'volatility_risk' in row:
                volatility_risk = row['volatility_risk']
                if volatility_risk >= 0.8:
                    stop_loss_adjustment -= self.profit_loss_config['stop_loss_adjustment']  # 高波动率，收紧止损
                elif volatility_risk < 0.3:
                    stop_loss_adjustment += self.profit_loss_config['stop_loss_adjustment']  # 低波动率，放宽止损
            
            # 基于流动性风险调整止损比例
            if 'liquidity_risk' in row:
                liquidity_risk = row['liquidity_risk']
                if liquidity_risk >= 0.8:
                    stop_loss_adjustment -= self.profit_loss_config['stop_loss_adjustment']  # 高流动性风险，收紧止损
            
            # 基于成交量比调整止损比例
            if 'volume_bid_volume_ratio' in row:
                volume_ratio = row['volume_bid_volume_ratio']
                if volume_ratio >= 5:
                    stop_loss_adjustment += self.profit_loss_config['stop_loss_adjustment']  # 成交量比极高，放宽止损
                elif volume_ratio < 1.5:
                    stop_loss_adjustment -= self.profit_loss_config['stop_loss_adjustment']  # 成交量比低，收紧止损
            
            # 计算最终止损比例
            stop_loss = base_stop_loss + stop_loss_adjustment
            stop_loss = max(0.02, min(0.15, stop_loss))  # 限制止损比例在2%-15%范围内
            
            # 基于封板概率调整跟踪止损比例
            trailing_stop_ratio = base_trailing_stop
            if limit_up_prob >= 0.9:
                trailing_stop_ratio += self.profit_loss_config['stop_loss_adjustment'] * 1.5  # 封板概率极高，大幅放宽跟踪止损
            elif limit_up_prob >= 0.8:
                trailing_stop_ratio += self.profit_loss_config['stop_loss_adjustment']  # 封板概率高，放宽跟踪止损
            elif limit_up_prob >= 0.7:
                trailing_stop_ratio += self.profit_loss_config['stop_loss_adjustment'] * 0.5  # 封板概率中等，小幅放宽跟踪止损
            elif limit_up_prob < 0.4:
                trailing_stop_ratio -= self.profit_loss_config['stop_loss_adjustment']  # 封板概率低，收紧跟踪止损
            elif limit_up_prob < 0.2:
                trailing_stop_ratio -= self.profit_loss_config['stop_loss_adjustment'] * 1.5  # 封板概率极低，大幅收紧跟踪止损
            
            # 基于预测置信度调整跟踪止损比例
            if prediction_confidence >= 0.9:
                trailing_stop_ratio += self.profit_loss_config['stop_loss_adjustment'] * 0.5  # 置信度高，放宽跟踪止损
            elif prediction_confidence < 0.6:
                trailing_stop_ratio -= self.profit_loss_config['stop_loss_adjustment'] * 0.5  # 置信度低，收紧跟踪止损
            
            trailing_stop_ratio = max(0.03, min(0.2, trailing_stop_ratio))  # 限制跟踪止损比例在3%-20%范围内
            
            # 计算止盈点位
            take_profit_price = suggested_buy_price * (1 + profit_target)
            
            # 计算止损点位
            stop_loss_price = suggested_buy_price * (1 - stop_loss)
            
            # 计算次日溢价空间（如果有预测结果）
            next_day_premium_price = None
            next_day_premium_target_price = None
            if 'next_day_premium' in row:
                next_day_premium = row['next_day_premium']
                next_day_premium_price = suggested_buy_price * (1 + next_day_premium)
                # 计算次日溢价目标点位（结合封板概率和置信度）
                confidence_factor = prediction_confidence * 0.8 + 0.2  # 置信度因子，范围0.2-1.0
                next_day_premium_target_price = suggested_buy_price * (1 + next_day_premium * limit_up_prob * confidence_factor)
            
            return {
                'take_profit_price': take_profit_price,
                'stop_loss_price': stop_loss_price,
                'trailing_stop_ratio': trailing_stop_ratio,
                'profit_target': profit_target,
                'stop_loss_ratio': stop_loss,
                'next_day_premium_price': next_day_premium_price,
                'next_day_premium_target_price': next_day_premium_target_price
            }
        
        # 应用计算
        profit_loss_results = evaluated_stocks.apply(calculate_profit_loss, axis=1, result_type='expand')
        evaluated_stocks = pd.concat([evaluated_stocks, profit_loss_results], axis=1)
        
        # 四舍五入到两位小数
        price_columns = ['take_profit_price', 'stop_loss_price', 'next_day_premium_price', 'next_day_premium_target_price']
        evaluated_stocks[price_columns] = evaluated_stocks[price_columns].round(2)
        
        logger.info("止盈止损点位计算完成")
        return evaluated_stocks
    
    def _calculate_dynamic_profit_loss_adjustment(self, evaluated_stocks: pd.DataFrame) -> pd.DataFrame:
        """计算动态止盈止损调整
        
        Args:
            evaluated_stocks: 带有止盈止损点位的股票数据
            
        Returns:
            带有动态止盈止损调整的股票数据
        """
        logger.info("计算动态止盈止损调整")
        
        def calculate_dynamic_adjustment(row):
            # 基于预测置信度调整止盈止损
            confidence_adjustment = {
                'take_profit_adjustment': 0,
                'stop_loss_adjustment': 0,
                'trailing_stop_adjustment': 0
            }
            
            confidence = row['prediction_confidence']
            
            # 置信度高时，提高止盈目标，放宽止损
            if confidence >= 0.9:
                confidence_adjustment['take_profit_adjustment'] = self.profit_loss_config['profit_target_adjustment'] * 0.5
                confidence_adjustment['stop_loss_adjustment'] = self.profit_loss_config['stop_loss_adjustment']
                confidence_adjustment['trailing_stop_adjustment'] = self.profit_loss_config['stop_loss_adjustment'] * 0.5
            # 置信度低时，降低止盈目标，收紧止损
            elif confidence < 0.7:
                confidence_adjustment['take_profit_adjustment'] = -self.profit_loss_config['profit_target_adjustment'] * 0.5
                confidence_adjustment['stop_loss_adjustment'] = -self.profit_loss_config['stop_loss_adjustment']
                confidence_adjustment['trailing_stop_adjustment'] = -self.profit_loss_config['stop_loss_adjustment'] * 0.5
            
            # 应用调整
            adjusted_take_profit = row['take_profit_price'] * (1 + confidence_adjustment['take_profit_adjustment'])
            adjusted_stop_loss = row['stop_loss_price'] * (1 - confidence_adjustment['stop_loss_adjustment'])
            adjusted_trailing_stop = row['trailing_stop_ratio'] + confidence_adjustment['trailing_stop_adjustment']
            
            # 确保调整后的止盈止损合理
            adjusted_trailing_stop = max(0.03, min(0.2, adjusted_trailing_stop))  # 限制跟踪止损比例在3%-20%范围内
            
            return {
                'adjusted_take_profit': adjusted_take_profit,
                'adjusted_stop_loss': adjusted_stop_loss,
                'adjusted_trailing_stop': adjusted_trailing_stop,
                'confidence_adjustment': confidence
            }
        
        # 应用计算
        dynamic_adjustment_results = evaluated_stocks.apply(calculate_dynamic_adjustment, axis=1, result_type='expand')
        evaluated_stocks = pd.concat([evaluated_stocks, dynamic_adjustment_results], axis=1)
        
        # 四舍五入到两位小数
        price_columns = ['adjusted_take_profit', 'adjusted_stop_loss']
        evaluated_stocks[price_columns] = evaluated_stocks[price_columns].round(2)
        
        logger.info("动态止盈止损调整计算完成")
        return evaluated_stocks
    
    def _generate_final_decisions(self, evaluated_stocks: pd.DataFrame) -> pd.DataFrame:
        """生成完整的决策建议
        
        Args:
            evaluated_stocks: 带有所有评估结果的股票数据
            
        Returns:
            完整的决策建议数据，包含：
            - 基本信息
            - 决策得分和优先级
            - 买入建议
            - 止盈止损建议
            - 预测结果
            - 风险评估
            - 决策时间
        """
        logger.info("生成完整的决策建议")
        
        # 明确定义决策建议包含的列
        decision_columns = [
            # 基本信息
            'stock_code', 'sector', 'industry',
            
            # 决策得分和优先级
            'decision_score', 'buy_priority',
            
            # 买入建议
            'buy_price_lower', 'buy_price_upper', 'suggested_buy_price',
            'position_ratio', 'actual_position_ratio',
            'suggested_buy_amount', 'actual_buy_amount',
            'suggested_buy_quantity',
            
            # 止盈止损建议
            'take_profit_price', 'stop_loss_price', 'next_day_premium_price',
            'trailing_stop_ratio',
            'profit_target', 'stop_loss_ratio',
            
            # 预测结果
            'prediction_score', 'limit_up_prob', 'next_day_premium',
            'rise_prob_30min', 'rise_prob_30min_lower', 'rise_prob_30min_upper',
            'limit_up_prob_lower', 'limit_up_prob_upper',
            'next_day_premium_lower', 'next_day_premium_upper',
            'prediction_confidence',
            
            # 风险评估
            'total_risk', 'risk_level', 'risk_alert'
        ]
        
        # 确保所有列都存在
        existing_columns = [col for col in decision_columns if col in evaluated_stocks.columns]
        logger.info(f"最终决策建议包含{len(existing_columns)}列")
        
        # 生成决策建议
        decisions = evaluated_stocks[existing_columns].copy()
        
        # 添加决策时间
        decisions['decision_time'] = pd.Timestamp.now()
        
        # 添加具体操作指引
        decisions = self._add_operation_guidance(decisions)
        
        logger.info("完整的决策建议生成完成")
        return decisions
    
    def _add_operation_guidance(self, decisions: pd.DataFrame) -> pd.DataFrame:
        """添加具体操作指引
        
        Args:
            decisions: 决策建议DataFrame
            
        Returns:
            带有操作指引的决策建议DataFrame
        """
        logger.info("添加具体操作指引")
        
        def generate_guidance(row):
            """生成单只股票的操作指引"""
            guidance = {
                '买入策略': f"以{row['suggested_buy_price']:.2f}元左右价格买入{row['suggested_buy_quantity']}股，实际买入金额约{row['actual_buy_amount']:.2f}元",
                '止盈策略': f"目标止盈价格{row['take_profit_price']:.2f}元，预期收益率{row['profit_target']:.2%}",
                '止损策略': f"止损价格{row['stop_loss_price']:.2f}元，止损比例{row['stop_loss_ratio']:.2%}；跟踪止损比例{row['trailing_stop_ratio']:.2%}",
                '仓位建议': f"建议仓位比例{row['actual_position_ratio']:.2%}，总仓位控制在80%以内",
                '风险提示': "建议设置止损，严格控制风险" if row['risk_alert'] else "风险在可控范围内",
                '决策依据': f"决策得分{row['decision_score']:.2f}，封板概率{row['limit_up_prob']:.2f}，风险等级{row['risk_level']}"
            }
            return pd.Series(guidance)
        
        # 应用操作指引生成
        guidance_results = decisions.apply(generate_guidance, axis=1)
        decisions = pd.concat([decisions, guidance_results], axis=1)
        
        return decisions
    
    def filter_top_decisions(self, decisions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """筛选TOP决策建议
        
        Args:
            decisions: 完整的决策建议数据
            top_n: 要筛选的TOP数量
            
        Returns:
            TOP决策建议数据
        """
        logger.info(f"筛选TOP {top_n}决策建议")
        
        if decisions.empty:
            return pd.DataFrame()
        
        # 1. 首先过滤掉风险预警的股票
        filtered_decisions = decisions[decisions['risk_alert'] == False]
        
        # 如果过滤后为空，则使用原始数据
        if filtered_decisions.empty:
            logger.warning("所有股票都触发了风险预警，使用原始数据")
            filtered_decisions = decisions.copy()
        
        # 2. 过滤掉高风险股票
        filtered_decisions = filtered_decisions[filtered_decisions['risk_level'] != '高风险']
        
        # 如果过滤后为空，则使用风险预警过滤后的数据
        if filtered_decisions.empty:
            logger.warning("没有低风险或中风险股票，使用风险预警过滤后的数据")
            filtered_decisions = decisions[decisions['risk_alert'] == False].copy()
        
        # 3. 选择TOP N决策
        top_decisions = filtered_decisions.head(top_n).copy()
        
        logger.info(f"筛选完成，共得到{len(top_decisions)}个TOP决策建议")
        return top_decisions
    
    def generate_decision_summary(self, decisions: pd.DataFrame) -> Dict[str, any]:
        """生成决策建议摘要
        
        Args:
            decisions: 完整的决策建议数据
            
        Returns:
            决策建议摘要字典
        """
        logger.info("生成决策建议摘要")
        
        if decisions.empty:
            return {}
        
        # 1. 计算决策建议统计信息
        summary = {
            'total_decisions': len(decisions),
            'top_5_stocks': decisions.head(5)['stock_code'].tolist(),
            'avg_decision_score': decisions['decision_score'].mean(),
            'avg_position_ratio': decisions['position_ratio'].mean(),
            'avg_profit_target': decisions['profit_target'].mean(),
            'avg_stop_loss_ratio': decisions['stop_loss_ratio'].mean(),
            'sector_distribution': decisions['sector'].value_counts().to_dict(),
            'risk_level_distribution': decisions['risk_level'].value_counts().to_dict(),
            'total_suggested_position': decisions['position_ratio'].sum(),
            'max_single_position': decisions['position_ratio'].max(),
            'decision_time': decisions['decision_time'].iloc[0]
        }
        
        logger.info(f"决策建议摘要：{summary}")
        return summary
    
    def export_decisions(self, decisions: pd.DataFrame, file_path: str = None) -> bool:
        """导出决策建议到文件
        
        Args:
            decisions: 完整的决策建议数据
            file_path: 导出文件路径，如果为None则使用默认路径
            
        Returns:
            是否导出成功
        """
        logger.info("导出决策建议到文件")
        
        if decisions.empty:
            logger.error("决策建议为空，无法导出")
            return False
        
        # 设置默认文件路径
        if file_path is None:
            import os
            from datetime import datetime
            
            # 创建决策建议目录
            os.makedirs('decisions', exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join('decisions', f'decisions_{timestamp}.csv')
        
        try:
            # 导出到CSV文件
            decisions.to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"决策建议导出成功，文件路径：{file_path}")
            return True
        except Exception as e:
            logger.error(f"决策建议导出失败：{e}")
            return False
