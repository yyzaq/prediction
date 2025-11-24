"""
连续动作版本的PPO股价预测
- 动作空间改为连续：[-1, 1]，表示仓位比例
- -1: 全仓做空，0: 空仓，1: 全仓做多
"""
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from math import sqrt

CONFIG = {
    "FEAT_PATH": "/Users/user/Desktop/prediction/eth_15min_features.csv",
    "TIME_COL": "timestamp",

    "SMA_SHORT": 10,
    "SMA_LONG": 50,
    "MFI_WIN": 14,
    "ATVMF_WIN": 20,
    "MFI_LOW": 20.0,
    "MFI_HIGH": 80.0,

    "TRAIN_START": "2022-10-01",
    "TRAIN_END":   "2024-10-01",
    "TEST_START":  "2024-10-01",
    "TEST_END":    "2025-10-01",

    "EPISODE_MINUTES": 360 * 24 * 60,

    "INIT_CASH": 10000.0,
    "MAX_POSITION_SIZE": 1.0,  # 最大仓位规模（单位）
    "FEE_RATE": 0.001,
    "DD_PENALTY": 0.1,
    "OBS_WIN": 60,
    "PPO_TIMESTEPS": 100_000,
    "PPO_LR": 3e-4,
    "GAMMA": 0.99,
    "N_STEPS": 2048,
    "BATCH_SIZE": 256,
    "N_EPOCHS": 10,
    "CLIP_RANGE": 0.2,
    "SEED": 42,
    "ENT_COEF": 0.05,

    # === NEW: 连续动作相关参数 ===
    "POSITION_CHANGE_PENALTY": 0.001,  # 仓位变化惩罚系数
    "MIN_POSITION_CHANGE": 0.05,       # 最小仓位变化阈值（避免频繁微调）
}

# === 导出动作日志相关 ===
LOG_DIR = "./runs"
LOG_TRADES_ONLY = False   # True=仅导出发生换仓的行；False=每一步都导出

REQUIRED_COLS = [
    "timestamp","open","high","low","close","volume",
    "sma_s","sma_l","mfi","atvmf","atvmf_ma","cross_state","ret1"
]

def load_features(cfg) -> pd.DataFrame:
    path = cfg["FEAT_PATH"]
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp","close"]).sort_values("timestamp").reset_index(drop=True)
    return df

def split_by_date(df: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t = cfg["TIME_COL"]
    train_start = pd.Timestamp(cfg["TRAIN_START"], tz="UTC")
    train_end   = pd.Timestamp(cfg["TRAIN_END"],   tz="UTC") + pd.Timedelta(days=1)
    test_start  = pd.Timestamp(cfg["TEST_START"],  tz="UTC")
    test_end    = pd.Timestamp(cfg["TEST_END"],    tz="UTC") + pd.Timedelta(days=1)

    tr_raw = df[(df[t] >= train_start) & (df[t] < train_end)].copy()
    te_raw = df[(df[t] >= test_start)  & (df[t] < test_end)].copy()

    print("[DEBUG] 全量:", df[t].min(), "->", df[t].max(), "| 行数=", len(df))
    print("[DEBUG] 训练切片:", train_start, "->", train_end, "(开区间上界) | 行数=", len(tr_raw))
    print("[DEBUG] 测试切片:",  test_start,  "->", test_end,  "(开区间上界) | 行数=", len(te_raw))

    warm = max(cfg["SMA_LONG"], cfg["MFI_WIN"], cfg["ATVMF_WIN"])
    if len(tr_raw) <= warm:
        raise ValueError(f"[ERROR] 训练在丢 warmup({warm}) 后为空；当前训练行数={len(tr_raw)}")
    if len(te_raw) <= warm:
        raise ValueError(f"[ERROR] 测试在丢 warmup({warm}) 后为空；当前测试行数={len(te_raw)}")

    tr = tr_raw.iloc[warm:].reset_index(drop=True)
    te = te_raw.iloc[warm:].reset_index(drop=True)
    return tr, te

# ---- 评估指标 ----
def max_drawdown(equity: pd.Series) -> float:
    peaks = equity.cummax(); dd = equity / peaks - 1.0
    return dd.min() if len(dd) else 0.0

def annualized_return(equity: pd.Series, freq_per_year=365*24*60) -> float:
    if len(equity) < 2: return 0.0
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    periods = len(equity)
    return (1.0 + total_ret) ** (freq_per_year / periods) - 1.0

def sharpe_ratio(returns: pd.Series, freq_per_year=365*24*60, rf=0.0) -> float:
    if returns.std() == 0 or returns.isna().all(): return 0.0
    mean, std = returns.mean(), returns.std()
    return (mean - rf) / std * sqrt(freq_per_year)

def sortino_ratio(returns: pd.Series, freq_per_year=365*24*60, rf=0.0) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0 or returns.isna().all(): return 0.0
    return (returns.mean() - rf) / downside.std() * sqrt(freq_per_year)

# ---- 强化学习环境（连续动作版本）----
@dataclass
class EpisodeClock:
    start_idx: int
    end_idx: int

class RLTradingEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, data: pd.DataFrame, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.df = data.reset_index(drop=True)
        self.tcol = cfg["TIME_COL"]
        self.close = self.df["close"].values.astype(float)
        self.obs_win = cfg["OBS_WIN"]

        atvmf = self.df["atvmf"].values.astype(float)
        self.atvmf_z = (atvmf - np.nanmean(atvmf)) / (np.nanstd(atvmf) + 1e-9)
        self.ret1 = self.df["ret1"].values.astype(float)
        self.cross_state = self.df["cross_state"].values.astype(float)
        self.mfi = self.df["mfi"].values.astype(float)
        self.atvmf = atvmf
        self.atvmf_ma = self.df["atvmf_ma"].values.astype(float)

        # === MODIFIED: 连续动作空间 ===
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.obs_win*4 + 1,), dtype=np.float32)

        self._make_episode_slices()

        # === MODIFIED: 连续仓位管理 ===
        self.position = 0.0  # 当前仓位比例 [-1, 1]
        self.target_position = 0.0  # 目标仓位
        self.cash = self.cfg["INIT_CASH"]
        self.asset = self.cfg["INIT_CASH"]
        self.equity_curve = []
        self.returns = []
        self.max_equity = self.asset
        self.idx = None
        self.ep_slice = None

        # 日志容器
        self.episode_id = -1
        self._step_logs = []

    def _make_episode_slices(self):
        ep_len = self.cfg["EPISODE_MINUTES"]
        total = len(self.df)
        self.episodes = []
        i = 0
        while i + ep_len + self.obs_win + 2 < total:
            self.episodes.append(EpisodeClock(start_idx=i, end_idx=i+ep_len))
            i += ep_len
        if not self.episodes:
            self.episodes.append(EpisodeClock(0, total-1))

    def _get_obs(self, i: int):
        s = max(0, i - self.obs_win + 1)
        mat = np.stack([
            self.ret1[s:i+1],
            self.cross_state[s:i+1],
            np.nan_to_num(self.mfi[s:i+1], nan=50.0),
            self.atvmf_z[s:i+1]
        ], axis=1)
        if mat.shape[0] < self.obs_win:
            pad = np.zeros((self.obs_win - mat.shape[0], mat.shape[1]))
            mat = np.vstack([pad, mat])
        obs = mat.flatten().astype(np.float32)
        obs = np.concatenate([obs, np.array([self.position], dtype=np.float32)], axis=0)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ep = self.np_random.integers(0, len(self.episodes))
        self.ep_slice = self.episodes[ep]
        self.idx = self.ep_slice.start_idx + self.obs_win
        self.position = 0.0
        self.target_position = 0.0
        self.cash = self.cfg["INIT_CASH"]
        self.asset = self.cfg["INIT_CASH"]
        self.equity_curve = [self.asset]
        self.returns = [0.0]
        self.max_equity = self.asset

        # 初始化日志状态
        self.episode_id += 1
        self._step_logs = []

        return self._get_obs(self.idx), {}

    def _log_step(self, i, raw_action, target_position, position_change, 
                  price_now, price_next, fee, pnl_step, reward):
        self._step_logs.append({
            "episode_id": self.episode_id,
            "i": int(i),
            "timestamp": self.df[self.tcol].iloc[i],
            "raw_action": float(raw_action),
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
        df = pd.DataFrame(self._step_logs)
        if df.empty:
            return df
        if trades_only:
            # 对于连续动作，我们定义仓位变化超过阈值才算交易
            min_change = self.cfg.get("MIN_POSITION_CHANGE", 0.05)
            df = df[df["position_change"].abs() > min_change].reset_index(drop=True)
        return df

    def step(self, action: np.ndarray):
        i = self.idx
        price_now = self.close[i]
        price_next = self.close[i+1] if i+1 < len(self.close) else price_now
        
        # === MODIFIED: 连续动作处理 ===
        raw_action = float(action[0])  # 原始动作值 [-1, 1]
        pos_before = float(self.position)
        
        # 应用双确认过滤（可选，根据需求可以保留或移除）
        cross = self.cross_state[i]
        mfi_t = self.mfi[i]
        av = self.atvmf[i]
        av_ma = self.atvmf_ma[i]
        allow_long = (cross == 1) and (mfi_t < self.cfg["MFI_LOW"]) and (av > av_ma)
        allow_short = (cross == -1) and (mfi_t > self.cfg["MFI_HIGH"]) and (av < av_ma)
        
        # 如果技术指标无效，不允许开仓
        if np.isnan(mfi_t) or np.isnan(av) or np.isnan(av_ma):
            allow_long = allow_short = False
        
        # 根据技术信号调整目标仓位
        if not allow_long and raw_action > 0:
            target_position = max(0.0, raw_action)  # 不允许做多时，至少设为0
        elif not allow_short and raw_action < 0:
            target_position = min(0.0, raw_action)  # 不允许做空时，至少设为0
        else:
            target_position = raw_action
        
        self.target_position = target_position
        
        # 计算仓位变化和交易费用
        position_change = target_position - pos_before
        abs_change = abs(position_change)
        
        # 只有当仓位变化超过阈值时才执行交易（避免频繁微调）
        if abs_change > self.cfg.get("MIN_POSITION_CHANGE", 0.05):
            # 费用基于仓位变化量计算
            trade_value = abs_change * price_now * self.cfg["MAX_POSITION_SIZE"]
            fee = trade_value * self.cfg["FEE_RATE"]
            self.position = target_position
        else:
            # 仓位变化太小，不执行交易
            fee = 0.0
            position_change = 0.0  # 记录实际变化为0
        
        # 计算收益
        pnl_step = (price_next - price_now) * self.position * self.cfg["MAX_POSITION_SIZE"]
        
        # 奖励计算
        rew = pnl_step - fee
        
        # 添加仓位变化惩罚（鼓励稳定性）
        position_penalty = abs(position_change) * self.cfg["POSITION_CHANGE_PENALTY"]
        rew -= position_penalty
        
        # 更新资产
        self.asset += pnl_step - fee
        self.max_equity = max(self.max_equity, self.asset)
        
        # 回撤惩罚
        dd = (self.asset / (self.max_equity + 1e-9)) - 1.0
        rew += self.cfg["DD_PENALTY"] * dd
        
        self.equity_curve.append(self.asset)
        if len(self.equity_curve) >= 2:
            ret = (self.equity_curve[-1] - self.equity_curve[-2]) / max(self.equity_curve[-2], 1e-9)
            self.returns.append(ret)
        else:
            self.returns.append(0.0)

        # 写日志
        self._log_step(i, raw_action, target_position, position_change,
                       price_now, price_next, fee, pnl_step, rew)

        self.idx += 1
        done = self.idx >= self.ep_slice.end_idx - 2
        return self._get_obs(self.idx), float(rew), bool(done), False, {}

class EquityLogger(BaseCallback):
    def _on_step(self) -> bool: return True

def evaluate(env: RLTradingEnv, model: PPO, name: str):
    obs, _ = env.reset(seed=CONFIG["SEED"])
    eq, rets = [], []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        eq.append(env.asset)
        if len(env.equity_curve) >= 2:
            rets.append(env.returns[-1])
        if done: break

    equity = pd.Series(eq)
    returns = pd.Series(rets).fillna(0.0)
    print(f"\n=== {name} 评估 ===")
    print(f"最终权益: {equity.iloc[-1]:.2f}")
    print(f"年化收益: {annualized_return(equity):.2%}")
    print(f"Sharpe : {sharpe_ratio(returns):.3f}")
    print(f"Sortino: {sortino_ratio(returns):.3f}")
    print(f"最大回撤: {max_drawdown(equity):.2%}")
    
    # 输出仓位统计
    action_df = env.get_action_log_df(trades_only=False)
    if not action_df.empty:
        print(f"平均仓位: {action_df['position_after'].mean():.3f}")
        print(f"仓位标准差: {action_df['position_after'].std():.3f}")
        print(f"最大做多仓位: {action_df['position_after'].max():.3f}")
        print(f"最大做空仓位: {action_df['position_after'].min():.3f}")

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(equity.values)
    plt.title(f"{name} Equity Curve")
    plt.ylabel("Equity")
    
    plt.subplot(2, 1, 2)
    plt.plot(action_df['position_after'].values)
    plt.title("Position Over Time")
    plt.xlabel("Step"); plt.ylabel("Position")
    plt.tight_layout(); plt.show()

    # 导出动作日志
    os.makedirs(LOG_DIR, exist_ok=True)
    out_csv = os.path.join(LOG_DIR, f"actions_{name.replace(' ', '_')}_continuous.csv")
    action_df = env.get_action_log_df(trades_only=LOG_TRADES_ONLY)
    action_df.to_csv(out_csv, index=False)
    print(f"[ACTIONS] saved to {out_csv}")
    if not action_df.empty:
        print(action_df.head(10))

def main():
    np.random.seed(CONFIG["SEED"])
    df = load_features(CONFIG)

    train_df, test_df = split_by_date(df, CONFIG)
    if len(train_df) == 0: raise ValueError("[ERROR] 训练集为空")
    if len(test_df) == 0: raise ValueError("[ERROR] 测试集为空")

    print(f"训练区间: {train_df[CONFIG['TIME_COL']].iloc[0]} -> {train_df[CONFIG['TIME_COL']].iloc[-1]} | 行数={len(train_df)}")
    print(f"测试区间: {test_df[CONFIG['TIME_COL']].iloc[0]} -> {test_df[CONFIG['TIME_COL']].iloc[-1]} | 行数={len(test_df)}")

    train_env = DummyVecEnv([lambda: RLTradingEnv(train_df, CONFIG)])
    test_env  = RLTradingEnv(test_df, CONFIG)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=CONFIG["PPO_LR"],
        gamma=CONFIG["GAMMA"],
        n_steps=CONFIG["N_STEPS"],
        batch_size=CONFIG["BATCH_SIZE"],
        n_epochs=CONFIG["N_EPOCHS"],
        clip_range=CONFIG["CLIP_RANGE"],
        verbose=1,
        seed=CONFIG["SEED"],
        policy_kwargs=dict(net_arch=[256, 256]),
        ent_coef=CONFIG.get("ENT_COEF", 0.02),
    )

    model.learn(total_timesteps=CONFIG["PPO_TIMESTEPS"], callback=EquityLogger())

    evaluate(test_env, model, "PPO Continuous (Test)")

if __name__ == "__main__":
    main()