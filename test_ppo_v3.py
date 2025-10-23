# ppo_from_features_oldstyle.py
# -*- coding: utf-8 -*-
"""
按旧代码语义，从已计算的分钟级特征文件训练/评估 PPO：
- 不做重采样/补齐
- 先加载特征，再按日期切分；切分后丢 warmup 行（与旧代码一致）
"""
#v2版本能够接受不同粒度的数据输入
#v3版本增加了action的输出

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
    "FEAT_PATH": "/Users/user/Desktop/股价/eth_15min_features.csv",
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
    "TRADE_SIZE": 1.0,
    "FEE_RATE": 0.001,
    "DD_PENALTY": 0.1,
    "OBS_WIN": 60,
    "PPO_TIMESTEPS": 500_000,
    "PPO_LR": 3e-4,
    "GAMMA": 0.99,
    "N_STEPS": 2048,
    "BATCH_SIZE": 256,
    "N_EPOCHS": 10,
    "CLIP_RANGE": 0.2,
    "SEED": 42,
    "ENT_COEF": 0.05,

}

# === NEW: 导出动作日志相关（目录与筛选开关） ===
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

# ---- 评估指标（同旧代码）----
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

# ---- 强化学习环境（与旧代码一致，+日志输出）----
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

        self.action_space = spaces.Discrete(3)  # 0 hold, 1 long, 2 short
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.obs_win*4 + 1,), dtype=np.float32)

        self._make_episode_slices()

        self.position = 0
        self.cash = self.cfg["INIT_CASH"]
        self.asset = self.cfg["INIT_CASH"]
        self.equity_curve = []
        self.returns = []
        self.max_equity = self.asset
        self.idx = None
        self.ep_slice = None

        # === NEW: 日志容器 ===
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
        self.position = 0
        self.cash = self.cfg["INIT_CASH"]
        self.asset = self.cfg["INIT_CASH"]
        self.equity_curve = [self.asset]
        self.returns = [0.0]
        self.max_equity = self.asset

        # === NEW: 初始化日志状态 ===
        self.episode_id += 1
        self._step_logs = []

        return self._get_obs(self.idx), {}

    # === NEW: 记录一步 ===
    def _log_step(self, i, raw_action, eff_action, allow_long, allow_short,
                  pos_before, pos_after, price_now, price_next, fee, pnl_step, reward):
        self._step_logs.append({
            "episode_id": self.episode_id,
            "i": int(i),
            "timestamp": self.df[self.tcol].iloc[i],
            "raw_action": int(raw_action),
            "eff_action": int(eff_action),
            "allow_long": bool(allow_long),
            "allow_short": bool(allow_short),
            "position_before": int(pos_before),
            "position_after": int(pos_after),
            "position_change": int(pos_after - pos_before),
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

    # === NEW: 取出日志 DataFrame ===
    def get_action_log_df(self, trades_only=False):
        df = pd.DataFrame(self._step_logs)
        if df.empty:
            return df
        if trades_only:
            df = df[df["position_change"] != 0].reset_index(drop=True)
        return df

    def step(self, action: int):
        i = self.idx
        price_now = self.close[i]
        price_next = self.close[i+1] if i+1 < len(self.close) else price_now

        raw_action = int(action)      # === NEW
        pos_before = int(self.position)  # === NEW

        # 双确认过滤（与旧代码一致）
        cross = self.cross_state[i]
        mfi_t = self.mfi[i]
        av = self.atvmf[i]
        av_ma = self.atvmf_ma[i]
        allow_long = (cross == 1) and (mfi_t < self.cfg["MFI_LOW"]) and (av > av_ma)
        allow_short = (cross == -1) and (mfi_t > self.cfg["MFI_HIGH"]) and (av < av_ma)
        if np.isnan(mfi_t) or np.isnan(av) or np.isnan(av_ma):
            allow_long = allow_short = False
        if self.position == 0:
            if action == 1 and not allow_long: action = 0
            if action == 2 and not allow_short: action = 0
        eff_action = int(action)      # === NEW

        # --- 按“腿数(legs)”一次性计费 ---
        pos_before = int(self.position)

        # 先只决定目标仓位（不要在分支里计费/改仓位）
        pos_after = pos_before
        if eff_action == 1 and pos_before != 1:
            pos_after = 1
        elif eff_action == 2 and pos_before != -1:
            pos_after = -1
        # eff_action == 0 -> pos_after 不变

        # legs: 0/1/2；反向(±1 -> ∓1) 等于两腿（先平再开）
        legs = abs(pos_after - pos_before)
        fee  = price_now * self.cfg["TRADE_SIZE"] * self.cfg["FEE_RATE"] * legs

        # 落地仓位后再用“新仓位”计算下一步收益
        self.position = pos_after
        pnl_step = (price_next - price_now) * self.position * self.cfg["TRADE_SIZE"]

        rew = pnl_step - fee

        self.asset += pnl_step - fee
        self.max_equity = max(self.max_equity, self.asset)
        dd = (self.asset / (self.max_equity + 1e-9)) - 1.0
        rew += self.cfg["DD_PENALTY"] * dd

        self.equity_curve.append(self.asset)
        self.returns.append((self.equity_curve[-1] - self.equity_curve[-2]) / max(self.equity_curve[-2], 1e-9))

        # === NEW: 写日志
        self._log_step(i, raw_action, eff_action, allow_long, allow_short,
                       pos_before, int(self.position), price_now, price_next, fee, pnl_step, rew)

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
        obs, reward, done, trunc, info = env.step(int(action))
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

    plt.figure(figsize=(10,4))
    plt.plot(equity.values)
    plt.title(f"{name} Equity Curve")
    plt.xlabel("Step"); plt.ylabel("Equity")
    plt.tight_layout(); plt.show()

    # === NEW: 导出动作日志 ===
    os.makedirs(LOG_DIR, exist_ok=True)
    out_csv = os.path.join(LOG_DIR, f"actions_{name.replace(' ', '_')}.csv")
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
    ent_coef=CONFIG.get("ENT_COEF", 0.02),  # ★ 新增：适度探索（建议 0.01~0.05）
    )

    model.learn(total_timesteps=CONFIG["PPO_TIMESTEPS"], callback=EquityLogger())

    evaluate(test_env, model, "PPO+SMA+ATVMF+MFI(Test, oldstyle)")

if __name__ == "__main__":
    main()
