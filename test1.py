"""
优化版的PPO股价预测 - 简化学习难度
"""
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from math import sqrt
import torch
import warnings
warnings.filterwarnings('ignore')

# ==================== 优化配置参数 ====================
CONFIG = {
    "FEAT_PATH": "/Users/user/Desktop/prediction/eth_15min_features.csv",
    "TIME_COL": "timestamp",

    # 技术指标参数
    "SMA_SHORT": 10,
    "SMA_LONG": 50,
    "MFI_WIN": 14,
    "ATVMF_WIN": 20,
    "MFI_LOW": 20.0,
    "MFI_HIGH": 80.0,

    # 数据分割
    "TRAIN_START": "2022-10-01",
    "TRAIN_END":   "2024-10-01",
    "TEST_START":  "2024-10-01", 
    "TEST_END":    "2025-10-01",

    # 环境参数 - 简化版本
    "EPISODE_MINUTES": 1000,              # 缩短episode长度，让学习更快
    "INIT_CASH": 10000.0,
    "MAX_POSITION_SIZE": 1.0,
    "FEE_RATE": 0.001,
    "DD_PENALTY": 0.01,                   # 减小回撤惩罚
    "OBS_WIN": 20,                        # 减小观察窗口
    
    # 离散仓位节点设置 - 简化到5个节点
    "POSITION_NODES": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "POSITION_CHANGE_PENALTY": 0.0001,    # 大幅减小换仓惩罚

    # PPO算法参数 - 优化训练
    "PPO_TIMESTEPS": 500000,              # 减少总步数，先测试学习能力
    "PPO_LR": 1e-4,                       # 降低学习率
    "GAMMA": 0.95,                        # 降低折扣因子，更关注近期奖励
    "N_STEPS": 512,                       # 减少步数
    "BATCH_SIZE": 64,                     # 减小批大小
    "N_EPOCHS": 5,                        # 减少训练轮数
    "CLIP_RANGE": 0.1,                    # 减小剪切范围，稳定训练
    "ENT_COEF": 0.1,                      # 增加熵系数，鼓励探索
    "SEED": 42,
}

REQUIRED_COLS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "sma_s", "sma_l", "mfi", "atvmf", "atvmf_ma", "cross_state", "ret1"
]

LOG_DIR = "./runs"
LOG_TRADES_ONLY = False

# ==================== 基础工具函数 ====================
def load_features(cfg) -> pd.DataFrame:
    """加载特征数据"""
    path = cfg["FEAT_PATH"]
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    return df

def split_by_date(df: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按日期分割数据"""
    t = cfg["TIME_COL"]
    train_start = pd.Timestamp(cfg["TRAIN_START"], tz="UTC")
    train_end   = pd.Timestamp(cfg["TRAIN_END"],   tz="UTC") + pd.Timedelta(days=1)
    test_start  = pd.Timestamp(cfg["TEST_START"],  tz="UTC")
    test_end    = pd.Timestamp(cfg["TEST_END"],    tz="UTC") + pd.Timedelta(days=1)

    tr_raw = df[(df[t] >= train_start) & (df[t] < train_end)].copy()
    te_raw = df[(df[t] >= test_start)  & (df[t] < test_end)].copy()

    warm = max(cfg["SMA_LONG"], cfg["MFI_WIN"], cfg["ATVMF_WIN"])
    if len(tr_raw) <= warm:
        raise ValueError(f"训练数据不足，需要至少{warm}行")
    if len(te_raw) <= warm:
        raise ValueError(f"测试数据不足，需要至少{warm}行")

    tr = tr_raw.iloc[warm:].reset_index(drop=True)
    te = te_raw.iloc[warm:].reset_index(drop=True)
    
    print(f"[数据统计] 训练集: {len(tr)} 行, 测试集: {len(te)} 行")
    return tr, te

# ==================== 评估指标函数 ====================
def max_drawdown(equity: pd.Series) -> float:
    peaks = equity.cummax()
    dd = equity / peaks - 1.0
    return dd.min() if len(dd) else 0.0

def annualized_return(equity: pd.Series, freq_per_year=365*24*4) -> float:
    if len(equity) < 2: 
        return 0.0
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    periods = len(equity)
    return (1.0 + total_ret) ** (freq_per_year / periods) - 1.0

def sharpe_ratio(returns: pd.Series, freq_per_year=365*24*4, rf=0.0) -> float:
    if returns.std() == 0 or returns.isna().all(): 
        return 0.0
    mean, std = returns.mean(), returns.std()
    return (mean - rf) / std * sqrt(freq_per_year)

# ==================== 优化后的强化学习环境 ====================
@dataclass
class EpisodeClock:
    start_idx: int
    end_idx: int

class OptimizedTradingEnv(gym.Env):
    """优化后的交易环境 - 大幅简化学习难度"""
    
    metadata = {"render_modes": []}
    
    def __init__(self, data: pd.DataFrame, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.df = data.reset_index(drop=True)
        self.close = self.df["close"].values.astype(float)
        self.obs_win = cfg["OBS_WIN"]

        # 简化特征处理
        self.ret1 = self.df["ret1"].values.astype(float)
        self.cross_state = self.df["cross_state"].values.astype(float)
        self.mfi = np.nan_to_num(self.df["mfi"].values.astype(float), nan=50.0)

        # 简化动作空间：5个离散仓位节点
        self.position_nodes = cfg["POSITION_NODES"]
        self.action_space = spaces.Discrete(len(self.position_nodes))
        
        # 简化观察空间：只保留最重要的特征
        self.observation_space = spaces.Box(
            low=-10, high=10,  # 限制范围，帮助学习
            shape=(self.obs_win * 3 + 2,), dtype=np.float32  # 3个特征 + 仓位和价格位置
        )

        self._make_episode_slices()
        self.reset()

    def _make_episode_slices(self):
        """创建更短的episode，加速学习"""
        ep_len = self.cfg["EPISODE_MINUTES"]
        total = len(self.df)
        self.episodes = []
        
        # 创建重叠的episode，提供更多学习机会
        i = 0
        step_size = ep_len // 2  # 50%重叠
        while i + ep_len + self.obs_win < total:
            self.episodes.append(EpisodeClock(start_idx=i, end_idx=i+ep_len))
            i += step_size
            
        if not self.episodes:
            self.episodes.append(EpisodeClock(0, min(total-1, 1000)))
        
        print(f"[Env] 创建了 {len(self.episodes)} 个训练episodes (长度: {ep_len})")

    def _get_obs(self, i: int):
        """简化的观察状态"""
        s = max(0, i - self.obs_win + 1)
        
        # 只使用3个核心特征
        mat = np.stack([
            self.ret1[s:i+1],
            self.cross_state[s:i+1], 
            self.mfi[s:i+1] / 100.0,  # 归一化到[0,1]
        ], axis=1)
        
        # 填充不足部分
        if mat.shape[0] < self.obs_win:
            pad = np.zeros((self.obs_win - mat.shape[0], mat.shape[1]))
            mat = np.vstack([pad, mat])
        
        # 展平特征
        obs = mat.flatten().astype(np.float32)
        
        # 添加额外信息：当前仓位和价格相对位置
        price_position = (self.close[i] - np.min(self.close[s:i+1])) / (np.max(self.close[s:i+1]) - np.min(self.close[s:i+1]) + 1e-9)
        additional_info = np.array([self.position, price_position], dtype=np.float32)
        
        obs = np.concatenate([obs, additional_info])
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        ep = self.np_random.integers(0, len(self.episodes))
        self.ep_slice = self.episodes[ep]
        self.idx = self.ep_slice.start_idx + self.obs_win
        
        # 重置状态
        self.position = 0.0
        self.cash = self.cfg["INIT_CASH"]
        self.asset = self.cfg["INIT_CASH"]
        self.equity_curve = [self.asset]
        self.returns = [0.0]
        self.max_equity = self.asset
        self.trade_count = 0
        self._step_logs = []
        
        return self._get_obs(self.idx), {}

    def _calculate_reward(self, pnl: float, fee: float, position_change: float, equity: float) -> float:
        """简化的奖励函数"""
        # 基础奖励：收益 - 费用
        base_reward = pnl - fee
        
        # 小幅惩罚频繁交易
        trade_penalty = abs(position_change) * self.cfg["POSITION_CHANGE_PENALTY"]
        
        # 组合奖励
        reward = base_reward - trade_penalty
        
        # 大幅缩放奖励，帮助学习
        return reward * 100.0  # 放大奖励信号

    def step(self, action: int):
        i = self.idx
        if i + 1 >= len(self.close):
            done = True
            obs = self._get_obs(i)
            return obs, 0.0, True, False, {}
            
        price_now = self.close[i]
        price_next = self.close[i+1]
        
        # 动作执行
        action_idx = int(action)
        target_position = self.position_nodes[action_idx]
        pos_before = self.position
        
        # 宽松的技术指标过滤（只作为建议，不强制）
        cross = self.cross_state[i]
        mfi_t = self.mfi[i]
        
        # 技术信号作为额外奖励，而不是强制限制
        technical_bonus = 0.0
        if target_position > 0 and cross > 0 and mfi_t < 30:
            technical_bonus = 0.001  # 小幅奖励符合技术信号的行动
        elif target_position < 0 and cross < 0 and mfi_t > 70:
            technical_bonus = 0.001
        
        # 计算仓位变化和费用
        position_change = target_position - pos_before
        abs_change = abs(position_change)
        
        if abs_change > 0:
            trade_value = abs_change * price_now * self.cfg["MAX_POSITION_SIZE"]
            fee = trade_value * self.cfg["FEE_RATE"]
            self.position = target_position
            self.trade_count += 1
        else:
            fee = 0.0
        
        # 计算收益
        pnl_step = (price_next - price_now) * self.position * self.cfg["MAX_POSITION_SIZE"]
        
        # 使用简化的奖励函数
        reward = self._calculate_reward(pnl_step, fee, position_change, self.asset)
        reward += technical_bonus  # 添加技术信号奖励
        
        # 更新资产
        self.asset += pnl_step - fee
        self.max_equity = max(self.max_equity, self.asset)
        
        # 记录
        self.equity_curve.append(self.asset)
        if len(self.equity_curve) >= 2:
            ret = (self.equity_curve[-1] - self.equity_curve[-2]) / max(self.equity_curve[-2], 1e-9)
            self.returns.append(ret)
        
        # 日志记录
        self._step_logs.append({
            "step": i,
            "action": action_idx,
            "position": float(self.position),
            "price": float(price_now),
            "reward": float(reward),
            "equity": float(self.asset),
            "trade_count": self.trade_count
        })

        self.idx += 1
        done = self.idx >= self.ep_slice.end_idx - 1 or self.idx >= len(self.close) - 2
        
        return self._get_obs(self.idx), float(reward), done, False, {}

    def get_action_log_df(self, trades_only=False):
        df = pd.DataFrame(self._step_logs)
        return df

# ==================== 改进的训练回调 ====================
class ImprovedTrainingCallback(BaseCallback):
    """改进的训练回调，监控学习进度"""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super(ImprovedTrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                print(f"步骤 {self.n_calls}: 平均奖励 = {mean_reward:.3f}")
        return True

# ==================== 评估函数 ====================
def evaluate_model(env: OptimizedTradingEnv, model: PPO, name: str):
    """评估模型性能"""
    obs, _ = env.reset()
    done = False
    steps = 0
    max_steps = 5000  # 限制评估步数
    
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        steps += 1

    equity = pd.Series(env.equity_curve)
    returns = pd.Series(env.returns).fillna(0.0)
    
    print(f"\n=== {name} 评估结果 ===")
    print(f"最终权益: {equity.iloc[-1]:.2f} (初始: {env.cfg['INIT_CASH']})")
    print(f"总收益: {(equity.iloc[-1] / env.cfg['INIT_CASH'] - 1) * 100:.2f}%")
    print(f"交易次数: {env.trade_count}")
    print(f"夏普比率: {sharpe_ratio(returns):.3f}")
    print(f"最大回撤: {max_drawdown(equity) * 100:.2f}%")
    
    # 仓位统计
    action_df = env.get_action_log_df()
    if not action_df.empty:
        position_nodes = env.position_nodes
        print(f"\n仓位分布:")
        for pos in position_nodes:
            count = len(action_df[action_df['position'] == pos])
            percentage = count / len(action_df) * 100
            print(f"  {pos:5}: {count:4}次 ({percentage:5.1f}%)")

    # 简单可视化
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(equity.values)
    plt.title("权益曲线")
    plt.ylabel("权益")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    if not action_df.empty:
        plt.plot(action_df['position'].values)
        plt.title("仓位变化")
        plt.ylabel("仓位")
        plt.ylim(-1.1, 1.1)
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    if not action_df.empty:
        position_counts = action_df['position'].value_counts().sort_index()
        plt.bar([str(p) for p in position_counts.index], position_counts.values)
        plt.title("仓位分布")
        plt.xlabel("仓位")
    
    plt.subplot(2, 2, 4)
    prices = env.close[env.ep_slice.start_idx:env.ep_slice.end_idx]
    plt.plot(prices)
    plt.title("价格走势")
    plt.ylabel("价格")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("优化版PPO交易模型 - 简化学习难度")
    print(f"仓位节点: {CONFIG['POSITION_NODES']}")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(CONFIG["SEED"])
    torch.manual_seed(CONFIG["SEED"])
    
    # 加载数据
    print("📊 加载数据...")
    df = load_features(CONFIG)
    train_df, test_df = split_by_date(df, CONFIG)
    
    print(f"训练集: {len(train_df)} 行")
    print(f"测试集: {len(test_df)} 行")
    
    # 创建环境
    print("🔧 创建训练环境...")
    train_env = OptimizedTradingEnv(train_df, CONFIG)
    train_env = DummyVecEnv([lambda: train_env])
    
    # 创建PPO模型
    print("🤖 创建PPO模型...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=CONFIG["PPO_LR"],
        gamma=CONFIG["GAMMA"],
        n_steps=CONFIG["N_STEPS"],
        batch_size=CONFIG["BATCH_SIZE"],
        n_epochs=CONFIG["N_EPOCHS"],
        clip_range=CONFIG["CLIP_RANGE"],
        ent_coef=CONFIG["ENT_COEF"],
        verbose=1,
        seed=CONFIG["SEED"],
        policy_kwargs=dict(
            net_arch=[64, 64],  # 简化网络结构
            activation_fn=torch.nn.ReLU,
        )
    )
    
    # 训练模型
    print("🚀 开始训练...")
    callback = ImprovedTrainingCallback(check_freq=2000)
    model.learn(
        total_timesteps=CONFIG["PPO_TIMESTEPS"],
        callback=callback,
        progress_bar=True
    )
    
    # 评估
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    
    test_env = OptimizedTradingEnv(test_df, CONFIG)
    evaluate_model(test_env, model, "优化PPO模型")
    
    # 保存模型
    os.makedirs(LOG_DIR, exist_ok=True)
    model_path = os.path.join(LOG_DIR, "ppo_optimized.zip")
    model.save(model_path)
    print(f"💾 模型已保存至: {model_path}")
    
    print("\n✅ 训练完成!")

if __name__ == "__main__":
    main()