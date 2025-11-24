"""
半定量版本的PPO股价预测 - 9个离散仓位节点
简化版本：去掉课程学习，使用标准训练流程
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

# ==================== 配置参数 ====================
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

    # 环境参数
    "EPISODE_MINUTES": 360 * 24 * 60,  # 每个episode的长度
    "INIT_CASH": 10000.0,              # 初始资金
    "MAX_POSITION_SIZE": 1.0,          # 最大仓位规模
    "FEE_RATE": 0.001,                 # 交易手续费率
    "DD_PENALTY": 0.1,                 # 回撤惩罚系数
    "OBS_WIN": 60,                     # 观察窗口大小
    
    # 离散仓位节点设置 - 9个节点
    "POSITION_NODES": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
    "POSITION_CHANGE_PENALTY": 0.001,  # 仓位变化惩罚

    # PPO算法参数
    "PPO_TIMESTEPS": 50000,           # 总训练步数
    "PPO_LR": 3e-4,                    # 学习率
    "GAMMA": 0.99,                     # 折扣因子
    "N_STEPS": 2048,                   # 每次更新的步数
    "BATCH_SIZE": 256,                 # 批大小
    "N_EPOCHS": 10,                    # 训练轮数
    "CLIP_RANGE": 0.2,                 # 剪切范围
    "ENT_COEF": 0.05,                  # 熵系数
    "SEED": 42,                        # 随机种子
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
    
    # 检查必要列是否存在
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")
    
    # 处理时间列
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

    print("[DEBUG] 全量数据:", df[t].min(), "->", df[t].max(), "| 行数=", len(df))
    print("[DEBUG] 训练集:", train_start, "->", train_end, "| 行数=", len(tr_raw))
    print("[DEBUG] 测试集:", test_start, "->", test_end, "| 行数=", len(te_raw))

    # 确保数据足够计算技术指标
    warm = max(cfg["SMA_LONG"], cfg["MFI_WIN"], cfg["ATVMF_WIN"])
    if len(tr_raw) <= warm:
        raise ValueError(f"训练数据不足，需要至少{warm}行")
    if len(te_raw) <= warm:
        raise ValueError(f"测试数据不足，需要至少{warm}行")

    tr = tr_raw.iloc[warm:].reset_index(drop=True)
    te = te_raw.iloc[warm:].reset_index(drop=True)
    return tr, te

# ==================== 评估指标函数 ====================
def max_drawdown(equity: pd.Series) -> float:
    """计算最大回撤"""
    peaks = equity.cummax()
    dd = equity / peaks - 1.0
    return dd.min() if len(dd) else 0.0

def annualized_return(equity: pd.Series, freq_per_year=365*24*4) -> float:
    """计算年化收益率"""
    if len(equity) < 2: 
        return 0.0
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    periods = len(equity)
    return (1.0 + total_ret) ** (freq_per_year / periods) - 1.0

def sharpe_ratio(returns: pd.Series, freq_per_year=365*24*4, rf=0.0) -> float:
    """计算夏普比率"""
    if returns.std() == 0 or returns.isna().all(): 
        return 0.0
    mean, std = returns.mean(), returns.std()
    return (mean - rf) / std * sqrt(freq_per_year)

def sortino_ratio(returns: pd.Series, freq_per_year=365*24*4, rf=0.0) -> float:
    """计算索提诺比率"""
    downside = returns[returns < 0]
    if downside.std() == 0 or returns.isna().all(): 
        return 0.0
    return (returns.mean() - rf) / downside.std() * sqrt(freq_per_year)

# ==================== 强化学习环境 ====================
@dataclass
class EpisodeClock:
    """Episode时钟，定义每个训练片段的起止位置"""
    start_idx: int
    end_idx: int

class SemiDiscreteTradingEnv(gym.Env):
    """半定量交易环境 - 使用9个离散仓位节点"""
    
    metadata = {"render_modes": []}
    
    def __init__(self, data: pd.DataFrame, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.df = data.reset_index(drop=True)
        self.tcol = cfg["TIME_COL"]
        self.close = self.df["close"].values.astype(float)
        self.obs_win = cfg["OBS_WIN"]

        # 特征数据预处理
        atvmf = self.df["atvmf"].values.astype(float)
        self.atvmf_z = (atvmf - np.nanmean(atvmf)) / (np.nanstd(atvmf) + 1e-9)
        self.ret1 = self.df["ret1"].values.astype(float)
        self.cross_state = self.df["cross_state"].values.astype(float)
        self.mfi = self.df["mfi"].values.astype(float)
        self.atvmf = atvmf
        self.atvmf_ma = self.df["atvmf_ma"].values.astype(float)

        # 动作空间：9个离散仓位节点
        self.position_nodes = cfg["POSITION_NODES"]
        self.action_space = spaces.Discrete(len(self.position_nodes))
        
        # 观察空间：技术指标 + 当前仓位
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_win * 4 + 1,), dtype=np.float32  # 4个特征 × 60时间步 + 1个仓位
        )

        self._make_episode_slices()
        self.reset()

    def _make_episode_slices(self):
        """创建episode切片，将数据分成多个训练片段"""
        ep_len = self.cfg["EPISODE_MINUTES"]
        total = len(self.df)
        self.episodes = []
        i = 0
        while i + ep_len + self.obs_win + 2 < total:
            self.episodes.append(EpisodeClock(start_idx=i, end_idx=i+ep_len))
            i += ep_len
        if not self.episodes:
            self.episodes.append(EpisodeClock(0, total-1))
        
        print(f"[Env] 创建了 {len(self.episodes)} 个训练episodes")

    def _get_obs(self, i: int):
        """获取观察状态"""
        s = max(0, i - self.obs_win + 1)
        
        # 构建特征矩阵：收益率、交叉状态、MFI、成交量
        mat = np.stack([
            self.ret1[s:i+1],
            self.cross_state[s:i+1], 
            np.nan_to_num(self.mfi[s:i+1], nan=50.0),  # 处理NaN值
            self.atvmf_z[s:i+1]
        ], axis=1)
        
        # 如果数据不足，用0填充
        if mat.shape[0] < self.obs_win:
            pad = np.zeros((self.obs_win - mat.shape[0], mat.shape[1]))
            mat = np.vstack([pad, mat])
        
        # 展平特征并添加当前仓位
        obs = mat.flatten().astype(np.float32)
        obs = np.concatenate([obs, np.array([self.position], dtype=np.float32)], axis=0)
        return obs

    def reset(self, *, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        
        # 随机选择一个episode开始
        ep = self.np_random.integers(0, len(self.episodes))
        self.ep_slice = self.episodes[ep]
        self.idx = self.ep_slice.start_idx + self.obs_win  # 跳过观察窗口
        
        # 重置交易状态
        self.position = 0.0
        self.cash = self.cfg["INIT_CASH"]
        self.asset = self.cfg["INIT_CASH"]
        self.equity_curve = [self.asset]
        self.returns = [0.0]
        self.max_equity = self.asset
        self.episode_id = getattr(self, 'episode_id', -1) + 1
        self._step_logs = []
        
        print(f"[Env] 重置到episode {self.episode_id}, 索引范围: {self.ep_slice.start_idx}-{self.ep_slice.end_idx}")
        return self._get_obs(self.idx), {}

    def _log_step(self, i, action_idx, target_position, position_change, 
                  price_now, price_next, fee, pnl_step, reward):
        """记录每一步的交易日志"""
        self._step_logs.append({
            "episode_id": self.episode_id,
            "i": int(i),
            "timestamp": self.df[self.tcol].iloc[i],
            "action_index": int(action_idx),
            "target_position": float(target_position),
            "position_before": float(self.position),
            "position_after": float(self.position),
            "position_change": float(position_change),
            "price_now": float(price_now),
            "price_next": float(price_next),
            "fee": float(fee),
            "pnl_step": float(pnl_step),
            "reward": float(reward),
            "equity": float(self.asset),
            "cross_state": float(self.cross_state[i]),
            "mfi": float(self.mfi[i]) if not np.isnan(self.mfi[i]) else np.nan,
            "atvmf_z": float(self.atvmf_z[i]) if not np.isnan(self.atvmf_z[i]) else np.nan,
        })

    def get_action_log_df(self, trades_only=False):
        """获取动作日志DataFrame"""
        df = pd.DataFrame(self._step_logs)
        if df.empty:
            return df
        if trades_only:
            # 只有仓位变化时才算交易
            df = df[df["position_change"] != 0].reset_index(drop=True)
        return df

    def step(self, action: int):
        """执行一步环境更新"""
        i = self.idx
        price_now = self.close[i]
        price_next = self.close[i+1] if i+1 < len(self.close) else price_now
        
        # 将动作索引转换为目标仓位
        action_idx = int(action)
        target_position = self.position_nodes[action_idx]
        pos_before = float(self.position)
        
        # 技术指标过滤条件
        cross = self.cross_state[i]
        mfi_t = self.mfi[i]
        av = self.atvmf[i]
        av_ma = self.atvmf_ma[i]
        
        # 检查技术指标是否有效
        technical_valid = not (np.isnan(mfi_t) or np.isnan(av) or np.isnan(av_ma))
        
        if technical_valid:
            # 做多条件：金叉 + MFI超卖 + 成交量高于均线
            allow_long = (cross == 1) and (mfi_t < self.cfg["MFI_LOW"]) and (av > av_ma)
            # 做空条件：死叉 + MFI超买 + 成交量低于均线
            allow_short = (cross == -1) and (mfi_t > self.cfg["MFI_HIGH"]) and (av < av_ma)
        else:
            allow_long = allow_short = False
        
        # 应用技术过滤
        if not allow_long and target_position > 0:
            target_position = 0.0  # 不允许做多时设为空仓
        elif not allow_short and target_position < 0:
            target_position = 0.0  # 不允许做空时设为空仓
        
        # 计算仓位变化和交易费用
        position_change = target_position - pos_before
        abs_change = abs(position_change)
        
        if abs_change > 0:  # 只有仓位变化时才收费
            trade_value = abs_change * price_now * self.cfg["MAX_POSITION_SIZE"]
            fee = trade_value * self.cfg["FEE_RATE"]
            self.position = target_position  # 更新仓位
        else:
            fee = 0.0
        
        # 计算收益
        pnl_step = (price_next - price_now) * self.position * self.cfg["MAX_POSITION_SIZE"]
        rew = pnl_step - fee  # 基础奖励 = 收益 - 费用
        
        # 仓位变化惩罚（减少频繁交易）
        position_penalty = abs(position_change) * self.cfg["POSITION_CHANGE_PENALTY"]
        rew -= position_penalty
        
        # 更新资产
        self.asset += pnl_step - fee
        self.max_equity = max(self.max_equity, self.asset)
        
        # 回撤惩罚（鼓励控制风险）
        dd = (self.asset / (self.max_equity + 1e-9)) - 1.0
        rew += self.cfg["DD_PENALTY"] * dd
        
        # 记录权益曲线和收益率
        self.equity_curve.append(self.asset)
        if len(self.equity_curve) >= 2:
            ret = (self.equity_curve[-1] - self.equity_curve[-2]) / max(self.equity_curve[-2], 1e-9)
            self.returns.append(ret)
        else:
            self.returns.append(0.0)

        # 记录日志
        self._log_step(i, action_idx, target_position, position_change,
                       price_now, price_next, fee, pnl_step, rew)

        # 更新索引并检查是否结束
        self.idx += 1
        done = self.idx >= self.ep_slice.end_idx - 2
        
        return self._get_obs(self.idx), float(rew), bool(done), False, {}

# ==================== 训练回调函数 ====================
class TrainingCallback(BaseCallback):
    """训练过程回调函数，用于监控训练进度"""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # 每1000步打印一次进度
        if self.n_calls % self.check_freq == 0:
            print(f"训练进度: {self.n_calls} 步")
        return True

# ==================== 评估函数 ====================
def evaluate_model(env: SemiDiscreteTradingEnv, model: PPO, name: str):
    """评估模型性能"""
    obs, _ = env.reset(seed=CONFIG["SEED"])
    eq, rets = [], []
    
    print(f"\n开始评估模型: {name}")
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        eq.append(env.asset)
        if len(env.equity_curve) >= 2:
            rets.append(env.returns[-1])
        if done: 
            break

    # 计算性能指标
    equity = pd.Series(eq)
    returns = pd.Series(rets).fillna(0.0)
    
    print(f"\n=== {name} 评估结果 ===")
    print(f"最终权益: {equity.iloc[-1]:.2f}")
    print(f"年化收益: {annualized_return(equity):.2%}")
    print(f"夏普比率: {sharpe_ratio(returns):.3f}")
    print(f"索提诺比率: {sortino_ratio(returns):.3f}")
    print(f"最大回撤: {max_drawdown(equity):.2%}")
    
    # 输出仓位统计
    action_df = env.get_action_log_df(trades_only=False)
    if not action_df.empty:
        position_nodes = CONFIG["POSITION_NODES"]
        print(f"\n仓位节点使用统计:")
        for pos in position_nodes:
            count = len(action_df[action_df['position_after'] == pos])
            percentage = count / len(action_df) * 100
            print(f"  {pos:5}: {count:4}次 ({percentage:5.1f}%)")

    # 可视化结果
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(equity.values)
    plt.title(f"{name} - 权益曲线")
    plt.ylabel("权益")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 2)
    plt.plot(action_df['position_after'].values)
    plt.title("仓位变化")
    plt.ylabel("仓位")
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 3)
    position_counts = action_df['position_after'].value_counts().sort_index()
    plt.bar([str(p) for p in position_counts.index], position_counts.values)
    plt.title("仓位分布")
    plt.xlabel("仓位节点")
    plt.ylabel("频次")
    
    plt.subplot(4, 1, 4)
    # 显示价格走势
    prices = env.close[env.ep_slice.start_idx:env.ep_slice.end_idx]
    plt.plot(prices)
    plt.title("价格走势")
    plt.ylabel("价格")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # 导出交易日志
    os.makedirs(LOG_DIR, exist_ok=True)
    out_csv = os.path.join(LOG_DIR, f"actions_{name.replace(' ', '_')}_simple.csv")
    action_df = env.get_action_log_df(trades_only=LOG_TRADES_ONLY)
    action_df.to_csv(out_csv, index=False)
    print(f"[ACTIONS] 交易日志已保存至: {out_csv}")
    
    if not action_df.empty:
        print("\n前10条交易记录:")
        print(action_df.head(10))

# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("PPO半定量交易模型 - 简化版本")
    print(f"仓位节点: {CONFIG['POSITION_NODES']}")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(CONFIG["SEED"])
    torch.manual_seed(CONFIG["SEED"])
    
    # 加载数据
    print("📊 加载数据...")
    df = load_features(CONFIG)
    train_df, test_df = split_by_date(df, CONFIG)
    
    if len(train_df) == 0: 
        raise ValueError("[ERROR] 训练集为空")
    if len(test_df) == 0: 
        raise ValueError("[ERROR] 测试集为空")
    
    print(f"训练集: {train_df[CONFIG['TIME_COL']].iloc[0]} -> {train_df[CONFIG['TIME_COL']].iloc[-1]}")
    print(f"测试集: {test_df[CONFIG['TIME_COL']].iloc[0]} -> {test_df[CONFIG['TIME_COL']].iloc[-1]}")
    
    # 创建训练环境
    print("🔧 创建训练环境...")
    train_env = SemiDiscreteTradingEnv(train_df, CONFIG)
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
            net_arch=[128, 128],
            activation_fn=torch.nn.ReLU,
        )
    )
    
    # 训练模型
    print("🚀 开始训练...")
    callback = TrainingCallback(check_freq=1000)
    model.learn(
        total_timesteps=CONFIG["PPO_TIMESTEPS"],
        callback=callback
    )
    
    # 评估模型
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    
    # 在测试集上评估
    test_env = SemiDiscreteTradingEnv(test_df, CONFIG)
    evaluate_model(test_env, model, "PPO半定量交易模型")
    
    # 保存模型
    os.makedirs(LOG_DIR, exist_ok=True)
    model_path = os.path.join(LOG_DIR, "ppo_semi_discrete_simple.zip")
    model.save(model_path)
    print(f"💾 模型已保存至: {model_path}")
    
    print("\n✅ 程序执行完成!")

if __name__ == "__main__":
    main()