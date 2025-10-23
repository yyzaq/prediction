# trading_ai_with_visualization_fixed.py
import os
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class VisualConfig:
    """可视化配置"""
    def __init__(self):
        # 数据路径
        self.FEAT_PATH = "features/15min/eth_15min_features.csv"
        
        # 训练参数
        self.DD_PENALTY = 0.02
        self.FEE_RATE = 0.0003
        self.TRADE_SIZE = 0.3
        self.PROFIT_REWARD = 1.5
        
        self.PPO_TIMESTEPS = 20_000  # 进一步减少步数以便快速演示
        self.N_STEPS = 512
        self.BATCH_SIZE = 128
        self.N_EPOCHS = 3
        self.PPO_LR = 8e-4
        self.GAMMA = 0.97
        self.CLIP_RANGE = 0.25
        self.ENT_COEF = 0.15
        
        # 环境参数
        self.INIT_CASH = 10000.0
        self.OBS_WIN = 30  # 减小观察窗口
        self.EPISODE_MINUTES = 360 * 2 * 24 * 60  # 2天数据
        
        # 技术指标
        self.SMA_SHORT = 10
        self.SMA_LONG = 50
        self.MFI_WIN = 14
        self.ATVMF_WIN = 20
        self.MFI_LOW = 20.0
        self.MFI_HIGH = 80.0
        
        # 可视化参数
        self.SHOW_LIVE_CHART = True
        self.UPDATE_INTERVAL = 1.0  # 增加更新间隔减少闪烁
        self.MAX_DISPLAY_STEPS = 200
        
        # 系统
        self.USE_GPU = False  # 在笔记本上使用CPU
        self.SEED = 42
        self.LOG_DIR = "./visual_logs"

cfg = VisualConfig()

class TradingVisualizer:
    """交易可视化器"""
    
    def __init__(self):
        self.fig = None
        self.ax1 = None  # 价格图表
        self.ax2 = None  # 资金曲线
        self.ax3 = None  # 技术指标
        
    def setup_charts(self):
        """设置图表布局"""
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        plt.tight_layout(pad=3.0)
        plt.ion()  # 开启交互模式
        print("📊 图表初始化完成")
        
    def update_charts(self, prices, equity_curve, current_idx, action_info, indicators):
        """更新所有图表"""
        # 清除所有图表
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # 1. 价格图表
        if len(prices) > 0 and current_idx < len(prices):
            self.ax1.plot(range(len(prices[:current_idx+1])), prices[:current_idx+1], 
                         'b-', linewidth=1, label='价格')
            self.ax1.axvline(x=current_idx, color='gray', linestyle='--', alpha=0.5)
            
            # 标记当前动作
            action_type = action_info.get('action_type', 'HOLD')
            color_map = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}
            marker_map = {'BUY': '^', 'SELL': 'v', 'HOLD': 'o'}
            
            self.ax1.scatter(current_idx, prices[current_idx], 
                           c=color_map.get(action_type, 'gray'), 
                           marker=marker_map.get(action_type, 'o'),
                           s=100, zorder=5, 
                           label=f'{action_type} (A:{action_info.get("action", 0)})')
            
            self.ax1.set_title(f'实时价格 - 当前: {prices[current_idx]:.2f} | 动作: {action_type}', 
                             fontsize=12, fontweight='bold')
            self.ax1.set_ylabel('价格')
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
        
        # 2. 资金曲线
        if len(equity_curve) > 0:
            self.ax2.plot(range(len(equity_curve)), equity_curve, 'g-', linewidth=2, label='资金曲线')
            self.ax2.axhline(y=cfg.INIT_CASH, color='r', linestyle='--', label='初始资金')
            
            current_equity = equity_curve[-1] if equity_curve else cfg.INIT_CASH
            self.ax2.scatter(len(equity_curve)-1, current_equity, c='red', s=50, zorder=5)
            
            profit_pct = (current_equity - cfg.INIT_CASH) / cfg.INIT_CASH * 100
            self.ax2.set_title(f'资金曲线 - 当前: {current_equity:.2f} ({profit_pct:+.2f}%)', 
                             fontsize=12, fontweight='bold')
            self.ax2.set_ylabel('资金')
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
        
        # 3. 技术指标
        if indicators:
            indicator_names = ['交叉状态', 'MFI', '成交量强度']
            values = [
                indicators.get('cross', 0),
                indicators.get('mfi', 50),
                indicators.get('volume_strength', 0)
            ]
            
            colors = ['blue', 'purple', 'brown']
            bars = self.ax3.bar(indicator_names, values, color=colors, alpha=0.7)
            
            # 在柱状图上显示数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                self.ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{value:.1f}', ha='center', va='bottom')
            
            # 添加参考线
            self.ax3.axhline(y=cfg.MFI_HIGH, color='red', linestyle='--', alpha=0.5, label='MFI超买')
            self.ax3.axhline(y=cfg.MFI_LOW, color='green', linestyle='--', alpha=0.5, label='MFI超卖')
            self.ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            self.ax3.set_title('技术指标', fontsize=12, fontweight='bold')
            self.ax3.set_ylabel('指标值')
            self.ax3.legend()
            self.ax3.grid(True, alpha=0.3)
        
        plt.draw()
        plt.pause(0.01)

class VisualTradingEnv(gym.Env):
    """修复版可视化交易环境"""
    
    def __init__(self, data_df, enable_visualization=True):
        super().__init__()
        self.df = data_df.reset_index(drop=True)
        self.close = self.df["close"].values.astype(float)
        self.enable_visualization = enable_visualization
        
        # 技术指标
        self.cross_state = self.df["cross_state"].values.astype(float)
        self.mfi = self.df["mfi"].values.astype(float)
        self.atvmf = self.df["atvmf"].values.astype(float)
        self.atvmf_ma = self.df["atvmf_ma"].values.astype(float)
        
        # 标准化
        atvmf_mean = np.nanmean(self.atvmf)
        atvmf_std = np.nanstd(self.atvmf) + 1e-9
        self.atvmf_z = (self.atvmf - atvmf_mean) / atvmf_std
        
        # 空间定义
        self.observation_space = spaces.Box(
            low=-10, high=10, 
            shape=(cfg.OBS_WIN * 4 + 1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        
        # 可视化
        if self.enable_visualization:
            self.visualizer = TradingVisualizer()
            try:
                self.visualizer.setup_charts()
            except Exception as e:
                print(f"⚠️ 图表初始化失败: {e}")
                self.enable_visualization = False
        
        self._setup_episodes()
        self.reset()
    
    def _setup_episodes(self):
        """设置训练片段"""
        total_len = len(self.df)
        episode_len = min(cfg.EPISODE_MINUTES, total_len - cfg.OBS_WIN - 10)
        self.episodes = []
        
        start = cfg.OBS_WIN
        while start + episode_len < total_len:
            self.episodes.append((start, start + episode_len))
            start += episode_len
        
        if not self.episodes:
            self.episodes.append((cfg.OBS_WIN, total_len - 5))
    
    def _get_obs(self, idx):
        """获取观察值"""
        start = max(0, idx - cfg.OBS_WIN + 1)
        
        features = []
        for i in range(start, idx + 1):
            if i < len(self.df):
                ret = self.df["ret1"].iloc[i] if "ret1" in self.df.columns else 0
                cross = self.cross_state[i] if i < len(self.cross_state) else 0
                mfi_val = self.mfi[i] if i < len(self.mfi) and not np.isnan(self.mfi[i]) else 50.0
                atvmf_val = self.atvmf_z[i] if i < len(self.atvmf_z) and not np.isnan(self.atvmf_z[i]) else 0.0
                
                features.append([ret, cross, mfi_val, atvmf_val])
            else:
                features.append([0, 0, 50.0, 0])
        
        while len(features) < cfg.OBS_WIN:
            features.insert(0, [0, 0, 50.0, 0])
        
        features = np.array(features, dtype=np.float32)
        obs = features.flatten()
        obs = np.concatenate([obs, [self.position]], dtype=np.float32)
        
        return obs
    
    def _get_action_info(self, raw_action, effective_action, current_price, next_price):
        """修复：确保action是整数"""
        # 确保action是标量整数
        if hasattr(raw_action, '__len__'):
            raw_action = int(raw_action[0]) if len(raw_action) > 0 else 0
        else:
            raw_action = int(raw_action)
            
        if hasattr(effective_action, '__len__'):
            effective_action = int(effective_action[0]) if len(effective_action) > 0 else 0
        else:
            effective_action = int(effective_action)
        
        action_types = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_type = action_types.get(effective_action, "UNKNOWN")
        
        return {
            'raw_action': raw_action,
            'effective_action': effective_action,
            'action_type': action_type,
            'current_price': current_price,
            'next_price': next_price
        }
    
    def _get_trading_info(self, action_info, reward):
        """获取交易信息用于显示"""
        current_time = self.df["timestamp"].iloc[self.current_idx] if self.current_idx < len(self.df) else "N/A"
        
        # 计算信号强度
        cross = self.cross_state[self.current_idx] if self.current_idx < len(self.cross_state) else 0
        mfi_val = self.mfi[self.current_idx] if self.current_idx < len(self.mfi) else 50.0
        volume_strength = self.atvmf_z[self.current_idx] if self.current_idx < len(self.atvmf_z) else 0.0
        
        signal_strength = "弱"
        if (action_info['action_type'] == "BUY" and cross == 1 and mfi_val < cfg.MFI_LOW) or \
           (action_info['action_type'] == "SELL" and cross == -1 and mfi_val > cfg.MFI_HIGH):
            signal_strength = "强"
        
        return {
            'timestamp': str(current_time),
            'current_price': action_info['current_price'],
            'action': action_info['raw_action'],
            'effective_action': action_info['effective_action'],
            'action_type': action_info['action_type'],
            'position': self.position,
            'equity': self.asset,
            'total_pnl': self.asset - cfg.INIT_CASH,
            'pnl_pct': (self.asset - cfg.INIT_CASH) / cfg.INIT_CASH * 100,
            'signal_strength': signal_strength,
            'mfi': mfi_val,
            'volume_strength': volume_strength,
            'reward': reward
        }
    
    def _display_trading_table(self, trading_info):
        """显示交易信息表格"""
        print("\n" + "="*70)
        print("📊 实时交易信息")
        print("="*70)
        
        table_data = [
            ["时间", trading_info.get('timestamp', 'N/A')],
            ["当前价格", f"{trading_info.get('current_price', 0):.2f}"],
            ["AI决策", f"{trading_info.get('action', 0)} -> {trading_info.get('effective_action', 0)} ({trading_info.get('action_type', 'N/A')})"],
            ["当前仓位", f"{trading_info.get('position', 0)}"],
            ["当前资金", f"{trading_info.get('equity', cfg.INIT_CASH):.2f}"],
            ["累计盈亏", f"{trading_info.get('total_pnl', 0):.2f} ({trading_info.get('pnl_pct', 0):+.2f}%)"],
            ["技术信号", trading_info.get('signal_strength', '等待信号')],
            ["MFI指标", f"{trading_info.get('mfi', 0):.1f}"],
            ["奖励", f"{trading_info.get('reward', 0):.3f}"]
        ]
        
        for row in table_data:
            print(f"{row[0]:<12} {row[1]:<40}")
        
        print("="*70)
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        ep_idx = np.random.randint(0, len(self.episodes))
        self.start_idx, self.end_idx = self.episodes[ep_idx]
        self.current_idx = self.start_idx
        
        # 重置状态
        self.position = 0
        self.cash = cfg.INIT_CASH
        self.asset = cfg.INIT_CASH
        self.equity_curve = [cfg.INIT_CASH]
        self.max_equity = cfg.INIT_CASH
        
        if self.enable_visualization:
            print("🔄 环境重置 - 开始新的交易周期")
        
        return self._get_obs(self.current_idx), {}
    
    def step(self, action):
        """执行一步"""
        # 修复：确保action是标量
        if hasattr(action, '__len__'):
            action = int(action[0]) if len(action) > 0 else 0
        else:
            action = int(action)
            
        if self.current_idx >= len(self.df) - 1:
            obs = self._get_obs(self.current_idx)
            return obs, 0, True, False, {}
        
        current_price = self.close[self.current_idx]
        next_price = self.close[self.current_idx + 1] if self.current_idx + 1 < len(self.close) else current_price
        
        # 双确认过滤
        cross = self.cross_state[self.current_idx] if self.current_idx < len(self.cross_state) else 0
        mfi_val = self.mfi[self.current_idx] if self.current_idx < len(self.mfi) else 50.0
        atvmf_val = self.atvmf[self.current_idx] if self.current_idx < len(self.atvmf) else 0.0
        atvmf_ma_val = self.atvmf_ma[self.current_idx] if self.current_idx < len(self.atvmf_ma) else 0.0
        
        allow_long = (cross == 1) and (mfi_val < cfg.MFI_LOW) and (atvmf_val > atvmf_ma_val)
        allow_short = (cross == -1) and (mfi_val > cfg.MFI_HIGH) and (atvmf_val < atvmf_ma_val)
        
        # 应用过滤
        effective_action = action
        action_reason = "AI决策"
        if self.position == 0:
            if action == 1 and not allow_long:
                effective_action = 0
                action_reason = "过滤: 不满足买入条件"
            elif action == 2 and not allow_short:
                effective_action = 0
                action_reason = "过滤: 不满足卖出条件"
        
        # 执行交易
        old_position = self.position
        new_position = old_position
        
        if effective_action == 1:
            new_position = 1
        elif effective_action == 2:
            new_position = -1
        
        # 计算费用和收益
        position_change = abs(new_position - old_position)
        fee = current_price * cfg.TRADE_SIZE * cfg.FEE_RATE * position_change
        
        price_change_pnl = (next_price - current_price) * new_position * cfg.TRADE_SIZE
        total_pnl = price_change_pnl - fee
        
        # 更新状态
        self.position = new_position
        self.asset += total_pnl
        self.equity_curve.append(self.asset)
        self.max_equity = max(self.max_equity, self.asset)
        
        # 奖励计算
        base_reward = total_pnl / cfg.INIT_CASH * 100
        
        if total_pnl > 0:
            base_reward *= cfg.PROFIT_REWARD
        
        drawdown = (self.asset - self.max_equity) / self.max_equity
        drawdown_penalty = drawdown * cfg.DD_PENALTY * 100
        
        final_reward = base_reward + drawdown_penalty
        final_reward = np.clip(final_reward, -5, 5)
        
        # 可视化
        if self.enable_visualization and cfg.SHOW_LIVE_CHART:
            action_info = self._get_action_info(action, effective_action, current_price, next_price)
            trading_info = self._get_trading_info(action_info, final_reward)
            
            # 更新图表
            try:
                indicators = {
                    'cross': cross,
                    'mfi': mfi_val,
                    'volume_strength': atvmf_val
                }
                self.visualizer.update_charts(
                    self.close, 
                    self.equity_curve, 
                    self.current_idx, 
                    action_info,
                    indicators
                )
            except Exception as e:
                print(f"⚠️ 图表更新失败: {e}")
            
            # 显示表格
            self._display_trading_table(trading_info)
            
            # 交易动作提示
            if effective_action != 0:  # 只有实际交易时才特别提示
                action_icons = {1: "🟢 买入", 2: "🔴 卖出", 0: "⚪ 持有"}
                print(f"\n🎯 {action_icons[effective_action]} 信号!")
                print(f"   原因: {action_reason}")
                print(f"   价格: {current_price:.2f} → {next_price:.2f}")
                print(f"   收益: {total_pnl:+.2f} | 奖励: {final_reward:+.3f}")
            
            time.sleep(cfg.UPDATE_INTERVAL)
        
        self.current_idx += 1
        done = self.current_idx >= self.end_idx or self.current_idx >= len(self.df) - 2
        
        obs = self._get_obs(self.current_idx)
        return obs, float(final_reward), done, False, {}

def create_sample_data():
    """创建样本数据"""
    print("📝 创建样本数据...")
    dates = pd.date_range("2024-01-01", periods=1000, freq='15T')
    np.random.seed(42)
    
    # 创建有趋势的价格数据
    trend = np.cumsum(np.random.randn(1000) * 0.05)
    noise = np.random.randn(1000) * 0.3
    prices = 100 + trend + noise
    
    # 计算移动平均线
    sma_s = pd.Series(prices).rolling(10).mean().fillna(method='bfill')
    sma_l = pd.Series(prices).rolling(50).mean().fillna(method='bfill')
    
    # 生成交叉信号
    cross_state = np.zeros(1000)
    for i in range(1, len(prices)):
        if sma_s.iloc[i] > sma_l.iloc[i] and sma_s.iloc[i-1] <= sma_l.iloc[i-1]:
            cross_state[i] = 1  # 金叉
        elif sma_s.iloc[i] < sma_l.iloc[i] and sma_s.iloc[i-1] >= sma_l.iloc[i-1]:
            cross_state[i] = -1  # 死叉
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(1000) * 0.1,
        'high': prices + np.abs(np.random.randn(1000) * 0.2),
        'low': prices - np.abs(np.random.randn(1000) * 0.2),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 1000),
        'sma_s': sma_s,
        'sma_l': sma_l,
        'mfi': np.random.uniform(10, 90, 1000),
        'atvmf': np.random.uniform(1000, 10000, 1000),
        'atvmf_ma': np.random.uniform(1000, 10000, 1000),
        'cross_state': cross_state,
        'ret1': np.random.randn(1000) * 0.01
    })
    
    return df.fillna(method='bfill').fillna(method='ffill')

def run_visual_training():
    """运行可视化训练"""
    print("🎬 启动可视化交易训练...")
    
    # 创建目录
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    
    # 加载或创建数据
    try:
        if os.path.exists(cfg.FEAT_PATH):
            df = pd.read_csv(cfg.FEAT_PATH)
            print(f"✅ 加载数据: {len(df)} 行")
        else:
            df = create_sample_data()
            print("📊 使用生成的样本数据")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        df = create_sample_data()
        print("📊 使用生成的样本数据")
    
    # 分割数据
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"📈 训练集: {len(train_df)} 行, 测试集: {len(test_df)} 行")
    
    # 创建环境 - 训练时关闭可视化
    train_env = DummyVecEnv([lambda: VisualTradingEnv(train_df, enable_visualization=False)])
    
    # 创建模型
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=cfg.PPO_LR,
        n_steps=cfg.N_STEPS,
        batch_size=cfg.BATCH_SIZE,
        n_epochs=cfg.N_EPOCHS,
        gamma=cfg.GAMMA,
        clip_range=cfg.CLIP_RANGE,
        ent_coef=cfg.ENT_COEF,
        verbose=1,
        policy_kwargs=dict(net_arch=[128, 128]),  # 简化网络
        device="cpu"  # 在笔记本上使用CPU
    )
    
    # 训练
    print("🎯 开始训练... (训练期间无可视化)")
    start_time = time.time()
    model.learn(total_timesteps=cfg.PPO_TIMESTEPS)
    training_time = time.time() - start_time
    
    print(f"✅ 训练完成! 耗时: {training_time/60:.1f} 分钟")
    
    # 测试时开启可视化
    print("\n🎬 开始可视化测试...")
    test_env = VisualTradingEnv(test_df, enable_visualization=True)
    obs, _ = test_env.reset()
    
    print("你将看到:")
    print("📈 实时价格图表 (带交易标记)")
    print("💰 资金曲线图表") 
    print("📊 技术指标图表")
    print("📋 实时交易信息表格")
    print("🎯 交易动作提示")
    
    input("\n按 Enter 开始可视化测试...")
    
    equities = []
    max_test_steps = min(200, len(test_df) - cfg.OBS_WIN - 10)  # 限制测试步数
    
    for i in range(max_test_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = test_env.step(action)
        equities.append(test_env.asset)
        if done:
            break
    
    # 最终结果
    final_equity = equities[-1] if equities else cfg.INIT_CASH
    profit_pct = (final_equity - cfg.INIT_CASH) / cfg.INIT_CASH * 100
    
    print(f"\n🎊 最终结果:")
    print(f"  初始资金: {cfg.INIT_CASH:.0f}")
    print(f"  最终资金: {final_equity:.0f}")
    print(f"  盈亏: {profit_pct:+.2f}%")
    print(f"  测试步数: {len(equities)}")
    
    # 保存模型
    model_path = os.path.join(cfg.LOG_DIR, "visual_model")
    model.save(model_path)
    print(f"💾 模型保存: {model_path}.zip")
    
    # 保持图表显示
    if cfg.SHOW_LIVE_CHART:
        print("📊 测试完成，关闭图表窗口继续...")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    run_visual_training()