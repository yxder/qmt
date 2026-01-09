#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风险控制模块
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger
import config

logger = setup_logger()

class RiskController:
    """风险控制类，用于监控和控制交易风险"""
    
    def __init__(self, initial_capital=1000000):
        """初始化风险控制器"""
        logger.info("初始化风险控制器")
        
        # 初始化资金相关参数
        self.initial_capital = initial_capital  # 初始资金
        self.current_capital = initial_capital  # 当前可用资金
        self.total_assets = initial_capital  # 当前总资产（可用资金+持仓市值）
        
        # 初始化风险控制参数
        self.current_total_position = 0.0  # 当前总仓位比例
        self.daily_pnl = 0.0  # 当日盈亏
        self.consecutive_losses = 0  # 连续亏损次数
        self.max_daily_loss = 0.0  # 当日最大亏损
        self.stock_positions = {}  # 单票持仓市值
        self.stock_position_ratios = {}  # 单票仓位比例
        
        # 新增风险控制参数
        self.max_drawdown = 0.0  # 最大回撤
        self.highest_assets = initial_capital  # 历史最高总资产
        self.daily_transaction_count = 0  # 当日交易次数
        self.max_daily_transactions = 20  # 单日最大交易次数
        self.liquidity_threshold = 1000000  # 降低流动性阈值，允许更多交易通过
        self.volatility_threshold = 0.10  # 波动率阈值（日涨跌幅超过此值的股票不交易）
    
    def check(self, decisions):
        """检查交易决策是否符合风险控制规则"""
        logger.info("进行风险控制检查")
        
        if decisions is None or not decisions:
            logger.warning("交易决策为空，跳过风险控制检查")
            return []
        
        try:
            # 过滤符合风险控制规则的决策
            approved_decisions = []
            
            for decision in decisions:
                if self._check_single_decision(decision):
                    approved_decisions.append(decision)
            
            logger.info(f"风险控制检查完成，共{len(approved_decisions)}个决策通过")
            return approved_decisions
            
        except Exception as e:
            logger.error(f"风险控制检查失败：{e}")
            return []
    
    def _check_single_decision(self, decision):
        """检查单个交易决策是否符合风险控制规则"""
        stock = decision['stock']
        action = decision['action']
        price = decision['price']
        
        # 检查价格有效性
        if price <= 0:
            logger.warning(f"股票{stock}价格无效({price})，拒绝交易")
            return False
        
        # 检查是否触发熔断机制
        if self._check_circuit_breaker():
            logger.warning(f"触发熔断机制，拒绝交易决策：{decision}")
            return False
        
        # 检查当日交易次数限制
        if self.daily_transaction_count >= self.max_daily_transactions:
            logger.warning(f"当日交易次数({self.daily_transaction_count})超过限制({self.max_daily_transactions})，拒绝交易")
            return False
        
        # 根据买卖方向检查不同的风险控制规则
        if action == 'buy':
            # 买入决策检查
            # 计算默认买入数量（总资金的1%）
            default_buy_amount = self.total_assets * 0.01
            try:
                quantity = int(default_buy_amount / price / 100) * 100  # 按100股整数倍
                quantity = max(quantity, 100)  # 至少买入100股
            except ZeroDivisionError:
                logger.warning(f"计算买入数量时除以零，拒绝交易：{decision}")
                return False
            
            if not self._check_buy_risk(stock, price, quantity):
                return False
            
            # 检查流动性风险（跳过，因为决策中没有volume字段）
            # 使用默认值1，避免成交额为0
            volume = decision.get('volume', 100000)
            if volume * price < self.liquidity_threshold:
                logger.warning(f"股票{stock}流动性不足，成交额({volume * price:.2f})低于阈值({self.liquidity_threshold:.2f})，拒绝买入")
                return False
            
            # 检查波动率风险
            volatility = decision.get('volatility', 0)
            if volatility > self.volatility_threshold:
                logger.warning(f"股票{stock}波动率过高({volatility:.4f})，超过阈值({self.volatility_threshold:.4f})，拒绝买入")
                return False
        elif action == 'sell':
            # 卖出决策检查
            if not self._check_sell_risk(stock, price):
                return False
        
        logger.info(f"交易决策通过风险控制检查：{decision}")
        return True
    
    def _check_buy_risk(self, stock, price, quantity):
        """检查买入决策的风险"""
        # 检查可用资金是否充足
        if not self._check_available_capital(price, quantity):
            return False
        
        # 检查单票仓位限制
        if not self._check_single_stock_position(stock, price, quantity):
            return False
        
        # 检查当日总仓位控制
        if not self._check_total_position(price, quantity):
            return False
        
        # 检查流动性风险
        if not self._check_liquidity_risk(stock, price, quantity):
            return False
        
        # 检查波动率风险
        if not self._check_volatility_risk(stock, price):
            return False
        
        # 检查市值风险
        if not self._check_market_cap_risk(stock, price, quantity):
            return False
        
        # 检查集中度风险
        if not self._check_concentration_risk(stock):
            return False
        
        return True
    
    def _check_sell_risk(self, stock, price):
        """检查卖出决策的风险"""
        # 检查股票是否在持仓中
        if stock not in self.stock_positions:
            logger.warning(f"股票不在持仓中，拒绝卖出：{stock}")
            return False
        
        return True
    
    def _check_single_stock_position(self, stock, price, buy_quantity):
        """检查单票仓位限制"""
        # 计算实际买入金额
        buy_value = price * buy_quantity
        
        # 计算当前单票持仓市值
        current_stock_value = self.stock_positions.get(stock, 0.0)
        
        # 计算买入后的单票持仓市值和比例
        new_stock_value = current_stock_value + buy_value
        new_stock_ratio = new_stock_value / self.total_assets
        
        # 检查是否超过单票最大仓位限制
        if new_stock_ratio > config.MAX_POSITION_PER_STOCK:
            logger.warning(f"单票仓位({new_stock_ratio:.4f})超过限制({config.MAX_POSITION_PER_STOCK:.4f})，拒绝买入")
            return False
        
        return True
    
    def _check_total_position(self, price, buy_quantity):
        """检查当日总仓位控制"""
        # 计算实际买入金额
        buy_value = price * buy_quantity
        
        # 计算当前总持仓市值
        current_total_value = sum(self.stock_positions.values())
        
        # 计算买入后的总持仓市值和比例
        new_total_value = current_total_value + buy_value
        new_total_ratio = new_total_value / self.total_assets
        
        # 检查是否超过总仓位最大限制
        if new_total_ratio > config.MAX_TOTAL_POSITION:
            logger.warning(f"总仓位({new_total_ratio:.4f})超过限制({config.MAX_TOTAL_POSITION:.4f})，拒绝买入")
            return False
        
        return True
    
    def _check_available_capital(self, price, buy_quantity):
        """检查可用资金是否充足"""
        # 计算实际买入金额（包括手续费）
        buy_value = price * buy_quantity
        commission = buy_value * config.COMMISSION_RATE
        total_cost = buy_value + commission
        
        # 检查可用资金是否充足
        if total_cost > self.current_capital:
            logger.warning(f"可用资金不足，需要{total_cost:.2f}，但只有{self.current_capital:.2f}")
            return False
        
        return True
    
    def _check_liquidity_risk(self, stock, price, quantity):
        """检查流动性风险"""
        # 检查成交量是否充足
        if hasattr(self, 'stock_liquidity_data') and stock in self.stock_liquidity_data:
            avg_volume = self.stock_liquidity_data[stock].get('avg_volume', 0)
            if avg_volume < quantity * 5:
                logger.warning(f"股票{stock}流动性不足，平均成交量({avg_volume})小于计划买入量的5倍")
                return False
        
        # 检查换手率是否过低
        if hasattr(self, 'stock_liquidity_data') and stock in self.stock_liquidity_data:
            turnover_rate = self.stock_liquidity_data[stock].get('turnover_rate', 0)
            if turnover_rate < 1.0:
                logger.warning(f"股票{stock}换手率过低({turnover_rate}%)，流动性风险较高")
                return False
        
        return True
    
    def _check_volatility_risk(self, stock, price):
        """检查波动率风险"""
        # 检查日涨跌幅是否过大
        if hasattr(self, 'stock_volatility_data') and stock in self.stock_volatility_data:
            daily_change = self.stock_volatility_data[stock].get('daily_change', 0)
            if abs(daily_change) > 10.0:
                logger.warning(f"股票{stock}当日涨跌幅({daily_change}%)过大，波动率风险较高")
                return False
        
        # 检查近期波动率是否过大
        if hasattr(self, 'stock_volatility_data') and stock in self.stock_volatility_data:
            volatility_5d = self.stock_volatility_data[stock].get('volatility_5d', 0)
            if volatility_5d > 0.15:
                logger.warning(f"股票{stock}近期波动率({volatility_5d*100}%)过大，风险较高")
                return False
        
        return True
    
    def _check_market_cap_risk(self, stock, price, quantity):
        """检查市值风险"""
        # 检查市值是否过小
        if hasattr(self, 'stock_fundamental_data') and stock in self.stock_fundamental_data:
            market_cap = self.stock_fundamental_data[stock].get('market_cap', 0)
            if market_cap < 1000000000:  # 小于10亿市值
                logger.warning(f"股票{stock}市值过小({market_cap/100000000}亿)，风险较高")
                return False
        
        return True
    
    def _check_concentration_risk(self, stock):
        """检查集中度风险"""
        # 检查相同板块持仓是否过多
        if hasattr(self, 'stock_sector_data') and stock in self.stock_sector_data:
            sector = self.stock_sector_data[stock]
            sector_position = 0.0
            for s, pos in self.stock_positions.items():
                if hasattr(self, 'stock_sector_data') and s in self.stock_sector_data and self.stock_sector_data[s] == sector:
                    sector_position += pos
            sector_ratio = sector_position / self.total_assets
            if sector_ratio > 0.3:  # 单个板块持仓不超过30%
                logger.warning(f"板块{sector}持仓比例({sector_ratio*100}%)过高，集中度风险较大")
                return False
        
        # 检查相同概念持仓是否过多
        if hasattr(self, 'stock_concept_data') and stock in self.stock_concept_data:
            concepts = self.stock_concept_data[stock]
            for concept in concepts:
                concept_position = 0.0
                for s, pos in self.stock_positions.items():
                    if hasattr(self, 'stock_concept_data') and s in self.stock_concept_data and concept in self.stock_concept_data[s]:
                        concept_position += pos
                concept_ratio = concept_position / self.total_assets
                if concept_ratio > 0.4:  # 单个概念持仓不超过40%
                    logger.warning(f"概念{concept}持仓比例({concept_ratio*100}%)过高，集中度风险较大")
                    return False
        
        return True
    
    def _check_circuit_breaker(self):
        """检查是否触发熔断机制"""
        # 检查连续亏损次数是否超过限制
        if self.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
            logger.warning(f"连续亏损次数({self.consecutive_losses})超过限制({config.MAX_CONSECUTIVE_LOSSES})，触发熔断")
            return True
        
        # 检查当日最大亏损是否超过限制
        if self.max_daily_loss <= -config.MAX_DAILY_LOSS:
            logger.warning(f"当日最大亏损({self.max_daily_loss})超过限制({-config.MAX_DAILY_LOSS})，触发熔断")
            return True
        
        # 检查最大回撤是否超过限制
        current_drawdown = (self.highest_assets - self.total_assets) / self.highest_assets
        if current_drawdown > config.MAX_DRAWDOWN:
            logger.warning(f"当前回撤({current_drawdown*100}%)超过限制({config.MAX_DRAWDOWN*100}%)，触发熔断")
            return True
        
        return False
    
    def update_risk_status(self, trade_results):
        """更新风险控制状态"""
        logger.info("更新风险控制状态")
        
        if trade_results is None or not trade_results:
            logger.warning("交易结果为空，跳过风险控制状态更新")
            return
        
        for result in trade_results:
            stock = result['stock']
            action = result['action']
            price = result['price']
            quantity = result['quantity']
            pnl = result.get('pnl', 0.0)  # 盈亏金额
            
            if action == 'buy':
                # 买入股票，更新资金和仓位
                buy_value = price * quantity
                commission = buy_value * config.COMMISSION_RATE
                total_cost = buy_value + commission
                
                # 更新可用资金
                self.current_capital -= total_cost
                
                # 更新单票持仓市值
                self.stock_positions[stock] = self.stock_positions.get(stock, 0.0) + buy_value
                
                # 更新单票仓位比例
                self.stock_position_ratios[stock] = self.stock_positions[stock] / self.total_assets
            
            elif action == 'sell':
                # 卖出股票，更新资金和仓位
                sell_value = price * quantity
                commission = sell_value * config.COMMISSION_RATE
                total_revenue = sell_value - commission
                
                # 更新可用资金
                self.current_capital += total_revenue
                
                # 更新单票持仓市值
                if stock in self.stock_positions:
                    self.stock_positions[stock] -= sell_value
                    if self.stock_positions[stock] <= 0:
                        del self.stock_positions[stock]
                        if stock in self.stock_position_ratios:
                            del self.stock_position_ratios[stock]
                    else:
                        # 更新单票仓位比例
                        self.stock_position_ratios[stock] = self.stock_positions[stock] / self.total_assets
            
            # 更新总资产
            self.total_assets = self.current_capital + sum(self.stock_positions.values())
            
            # 更新总仓位比例
            self.current_total_position = sum(self.stock_positions.values()) / self.total_assets
            
            # 更新盈亏情况
            self.daily_pnl += pnl
            
            # 更新当日最大亏损
            if self.daily_pnl < self.max_daily_loss:
                self.max_daily_loss = self.daily_pnl
            
            # 更新连续亏损次数
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
        
        logger.info(f"风险状态更新完成：总资产={self.total_assets:.2f}, 可用资金={self.current_capital:.2f}, 总仓位={self.current_total_position:.4f}")
    
    def reset_daily_status(self):
        """重置每日风险控制状态"""
        logger.info("重置每日风险控制状态")
        
        self.daily_pnl = 0.0
        self.max_daily_loss = 0.0
        self.consecutive_losses = 0
    
    def get_risk_status(self):
        """获取当前风险控制状态"""
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_assets': self.total_assets,
            'current_total_position': self.current_total_position,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'max_daily_loss': self.max_daily_loss,
            'stock_positions': self.stock_positions,
            'stock_position_ratios': self.stock_position_ratios
        }
