"加入了定量交易"
# ppo_from_features_oldstyle.py
# -*- coding: utf-8 -*-
"""
按旧代码语义，从已计算的分钟级特征文件训练/评估 PPO：
- 不做重采样/补齐
- 先加载特征，再按日期切分；切分后丢 warmup 行（与旧代码一致）
"""
#v2版本能够接受不同粒度的数据输入
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
import tkinter as tk
from tkinter import ttk
import threading
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates

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
    "TRADE_SIZE": 1.0,  # 旧版本：固定交易量
    "TRADE_RATIO": 0.1,  # 新版本：每次交易资金比例 (0.1 = 10%)
    "MAX_POSITION_RATIO": 0.8,  # 最大仓位比例
    "FEE_RATE": 0.001,
    "DD_PENALTY": 0.1,
    "OBS_WIN": 60,
    "PPO_TIMESTEPS": 200_00,
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

# === NEW: 实时UI界面 ===
class TradingUI:
    def __init__(self, root, env):
        self.root = root
        self.env = env
        self.running = False
        self.equity_data = []
        self.time_data = []
        
        # 设置主窗口
        self.root.title("AI交易实时监控")
        self.root.geometry("1200x800")
        
        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 信息显示区域
        info_frame = ttk.LabelFrame(main_frame, text="交易状态", padding="10")
        info_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 状态标签
        self.time_label = ttk.Label(info_frame, text="当前时间: -", font=('Arial', 12))
        self.time_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.position_label = ttk.Label(info_frame, text="持仓状态: 空仓", font=('Arial', 12))
        self.position_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        self.position_size_label = ttk.Label(info_frame, text="持仓比例: 0.0%", font=('Arial', 12))
        self.position_size_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        
        self.equity_label = ttk.Label(info_frame, text="当前权益: ¥10,000.00", font=('Arial', 12, 'bold'))
        self.equity_label.grid(row=1, column=0, sticky=tk.W, padx=(0, 20))
        
        self.pnl_label = ttk.Label(info_frame, text="累计盈亏: ¥0.00", font=('Arial', 12))
        self.pnl_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 20))
        
        # 详细指标区域
        metrics_frame = ttk.LabelFrame(main_frame, text="交易指标", padding="10")
        metrics_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 创建两列指标
        self.metric_labels = {}
        metrics = [
            ("交易次数", "trades_count"),
            ("胜率", "win_rate"),
            ("平均盈亏", "avg_pnl"),
            ("最大回撤", "max_drawdown"),
            ("夏普比率", "sharpe_ratio"),
            ("索提诺比率", "sortino_ratio")
        ]
        
        for i, (name, key) in enumerate(metrics):
            label = ttk.Label(metrics_frame, text=f"{name}: -")
            label.grid(row=i//3, column=i%3, sticky=tk.W, padx=(0, 30), pady=2)
            self.metric_labels[key] = label
        
        # 图表区域
        chart_frame = ttk.LabelFrame(main_frame, text="权益曲线", padding="10")
        chart_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 创建matplotlib图表
        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 控制按钮区域
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.start_button = ttk.Button(control_frame, text="开始交易", command=self.start_trading)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="停止交易", command=self.stop_trading, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)
        
        # 交易日志区域
        log_frame = ttk.LabelFrame(main_frame, text="交易日志", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建日志文本框
        self.log_text = tk.Text(log_frame, height=8, width=100)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 配置网格权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        main_frame.rowconfigure(4, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        # 初始化图表
        self.update_chart()
        
    def log_message(self, message):
        """添加日志消息"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def update_chart(self):
        """更新权益曲线图表"""
        self.ax.clear()
        if len(self.equity_data) > 0:
            self.ax.plot(self.time_data, self.equity_data, 'b-', linewidth=2)
            self.ax.set_title('实时权益曲线', fontsize=14, fontweight='bold')
            self.ax.set_ylabel('权益 (¥)', fontsize=12)
            self.ax.set_xlabel('时间', fontsize=12)
            self.ax.grid(True, alpha=0.3)
            
            # 格式化x轴日期显示
            if len(self.time_data) > 0:
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                self.fig.autofmt_xdate()
                
        self.canvas.draw()
        
    def update_display(self, env, current_time, step_info=None):
        """更新UI显示"""
        # 更新基本信息
        self.time_label.config(text=f"当前时间: {current_time}")
        
        position_map = {0: "空仓", 1: "多头", -1: "空头"}
        position_text = position_map.get(env.position, "未知")
        self.position_label.config(text=f"持仓状态: {position_text}")
        
        # 计算并显示持仓比例
        position_ratio = abs(env.position_value / env.asset * 100) if env.asset > 0 else 0
        self.position_size_label.config(text=f"持仓比例: {position_ratio:.1f}%")
        
        current_equity = env.asset
        self.equity_label.config(text=f"当前权益: ¥{current_equity:,.2f}")
        
        initial_cash = env.cfg["INIT_CASH"]
        total_pnl = current_equity - initial_cash
        pnl_color = "green" if total_pnl >= 0 else "red"
        self.pnl_label.config(text=f"累计盈亏: ¥{total_pnl:+,.2f}", foreground=pnl_color)
        
        # 更新权益曲线数据
        self.equity_data.append(current_equity)
        self.time_data.append(current_time)
        
        # 保持最近500个数据点
        if len(self.equity_data) > 500:
            self.equity_data = self.equity_data[-500:]
            self.time_data = self.time_data[-500:]
            
        # 更新图表
        self.update_chart()
        
        # 如果有交易发生，记录日志
        if step_info and step_info.get('position_change', 0) != 0:
            action_map = {0: "持有", 1: "做多", 2: "做空"}
            raw_action = action_map.get(step_info.get('raw_action', 0), "未知")
            trade_type = "加仓" if step_info.get('position_change', 0) > 0 else "减仓"
            self.log_message(f"{trade_type}操作: {raw_action} | 仓位: {position_map.get(step_info.get('position_before', 0))} → {position_text} | 持仓比例: {position_ratio:.1f}% | 盈亏: ¥{step_info.get('pnl_step', 0):+,.2f}")
        
    def start_trading(self):
        """开始交易"""
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.log_message("AI交易系统启动...")
        
    def stop_trading(self):
        """停止交易"""
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.log_message("AI交易系统停止")

# ---- 强化学习环境（修改为支持定量交易）----
@dataclass
class EpisodeClock:
    start_idx: int
    end_idx: int

class RLTradingEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, data: pd.DataFrame, cfg: Dict[str, Any], ui_callback=None):
        super().__init__()
        self.cfg = cfg
        self.df = data.reset_index(drop=True)
        self.tcol = cfg["TIME_COL"]
        self.close = self.df["close"].values.astype(float)
        self.obs_win = cfg["OBS_WIN"]
        self.ui_callback = ui_callback  # === NEW: UI回调函数

        atvmf = self.df["atvmf"].values.astype(float)
        self.atvmf_z = (atvmf - np.nanmean(atvmf)) / (np.nanstd(atvmf) + 1e-9)
        self.ret1 = self.df["ret1"].values.astype(float)
        self.cross_state = self.df["cross_state"].values.astype(float)
        self.mfi = self.df["mfi"].values.astype(float)
        self.atvmf = atvmf
        self.atvmf_ma = self.df["atvmf_ma"].values.astype(float)

        # 修改动作空间：0=减仓, 1=持有, 2=加仓
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.obs_win*4 + 2,), dtype=np.float32)  # +2 for position and position_value

        self._make_episode_slices()

        # 仓位相关变量
        self.position = 0.0  # 仓位比例 (-1.0 到 1.0)
        self.position_value = 0.0  # 持仓市值
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
        # 添加仓位信息到观察空间
        obs = np.concatenate([obs, np.array([self.position, self.position_value / self.asset], dtype=np.float32)], axis=0)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ep = self.np_random.integers(0, len(self.episodes))
        self.ep_slice = self.episodes[ep]
        self.idx = self.ep_slice.start_idx + self.obs_win
        self.position = 0.0
        self.position_value = 0.0
        self.cash = self.cfg["INIT_CASH"]
        self.asset = self.cfg["INIT_CASH"]
        self.equity_curve = [self.asset]
        self.returns = [0.0]
        self.max_equity = self.asset

        # === NEW: 初始化日志状态 ===
        self.episode_id += 1
        self._step_logs = []

        return self._get_obs(self.idx), {}

    def _calculate_trade_size(self, current_equity):
        """计算交易量"""
        if "TRADE_RATIO" in self.cfg:
            # 按比例交易
            return current_equity * self.cfg["TRADE_RATIO"]
        else:
            # 固定交易量（向后兼容）
            return self.cfg["TRADE_SIZE"] * self.close[self.idx]

    def _execute_trade(self, target_position_change, price_now):
        """执行交易并返回费用和实际仓位变化"""
        trade_amount = abs(target_position_change) * price_now
        if trade_amount == 0:
            return 0.0, 0.0
        
        # 计算交易费用
        fee = trade_amount * self.cfg["FEE_RATE"]
        return fee, target_position_change

    # === NEW: 记录一步 ===
    def _log_step(self, i, raw_action, eff_action, allow_long, allow_short,
                  pos_before, pos_after, price_now, price_next, fee, pnl_step, reward):
        step_info = {
            "episode_id": self.episode_id,
            "i": int(i),
            "timestamp": self.df[self.tcol].iloc[i],
            "raw_action": int(raw_action),
            "eff_action": int(eff_action),
            "allow_long": bool(allow_long),
            "allow_short": bool(allow_short),
            "position_before": float(pos_before),
            "position_after": float(pos_after),
            "position_change": float(pos_after - pos_before),
            "price_now": float(price_now),
            "price_next": float(price_next),
            "fee": float(fee),
            "pnl_step": float(pnl_step),
            "reward": float(reward),
            "equity": float(self.asset),
            "position_value": float(self.position_value),
            "position_ratio": float(self.position_value / self.asset) if self.asset > 0 else 0.0,
            "cross_state": float(self.cross_state[i]),
            "mfi": float(self.mfi[i]) if not np.isnan(self.mfi[i]) else np.nan,
            "atvmf_z": float(self.atvmf_z[i]) if not np.isnan(self.atvmf_z[i]) else np.nan,
        }
        self._step_logs.append(step_info)
        
        # === NEW: 调用UI更新 ===
        if self.ui_callback:
            self.ui_callback(self, step_info)

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

        raw_action = int(action)
        pos_before = float(self.position)

        # 双确认过滤（与旧代码一致）
        cross = self.cross_state[i]
        mfi_t = self.mfi[i]
        av = self.atvmf[i]
        av_ma = self.atvmf_ma[i]
        allow_long = (cross == 1) and (mfi_t < self.cfg["MFI_LOW"]) and (av > av_ma)
        allow_short = (cross == -1) and (mfi_t > self.cfg["MFI_HIGH"]) and (av < av_ma)
        if np.isnan(mfi_t) or np.isnan(av) or np.isnan(av_ma):
            allow_long = allow_short = False

        # 计算目标仓位变化
        target_position_change = 0.0
        max_position = self.cfg.get("MAX_POSITION_RATIO", 0.8)
        
        if action == 2:  # 加仓
            if (self.position >= 0 and allow_long) or (self.position <= 0 and allow_short):
                trade_value = self._calculate_trade_size(self.asset)
                target_position_change = trade_value / price_now
                # 限制最大仓位
                if self.position >= 0:
                    max_addition = (max_position * self.asset - self.position_value) / price_now
                    target_position_change = min(target_position_change, max_addition)
                else:
                    max_addition = (max_position * self.asset + self.position_value) / price_now
                    target_position_change = max(-target_position_change, -max_addition)
                    
        elif action == 0:  # 减仓
            if self.position != 0:
                trade_value = self._calculate_trade_size(self.asset)
                target_position_change = -trade_value / price_now * np.sign(self.position)
                # 确保不会过度减仓导致反向持仓
                if abs(self.position) * price_now < trade_value:
                    target_position_change = -self.position

        # 应用过滤条件
        eff_action = raw_action
        if target_position_change > 0 and not (allow_long or (self.position < 0 and allow_short)):
            target_position_change = 0.0
            eff_action = 1  # 强制转为持有
        elif target_position_change < 0 and not (allow_short or (self.position > 0 and allow_long)):
            target_position_change = 0.0
            eff_action = 1  # 强制转为持有

        # 执行交易
        fee, actual_position_change = self._execute_trade(target_position_change, price_now)
        self.position += actual_position_change
        
        # 更新持仓市值和现金
        old_position_value = self.position_value
        self.position_value = self.position * price_now
        self.cash -= actual_position_change * price_now + fee

        # 计算盈亏
        pnl_step = (price_next - price_now) * self.position
        self.asset = self.cash + self.position * price_next

        rew = pnl_step - fee

        self.max_equity = max(self.max_equity, self.asset)
        dd = (self.asset / (self.max_equity + 1e-9)) - 1.0
        rew += self.cfg["DD_PENALTY"] * dd

        self.equity_curve.append(self.asset)
        if len(self.equity_curve) >= 2:
            ret = (self.equity_curve[-1] - self.equity_curve[-2]) / max(self.equity_curve[-2], 1e-9)
            self.returns.append(ret)
        else:
            self.returns.append(0.0)

        # === NEW: 写日志
        self._log_step(i, raw_action, eff_action, allow_long, allow_short,
                       pos_before, float(self.position), price_now, price_next, fee, pnl_step, rew)

        self.idx += 1
        done = self.idx >= self.ep_slice.end_idx - 2
        return self._get_obs(self.idx), float(rew), bool(done), False, {}

class EquityLogger(BaseCallback):
    def _on_step(self) -> bool: return True

def evaluate_with_ui(env: RLTradingEnv, model: PPO, name: str, ui: TradingUI):
    """带UI的评估函数"""
    obs, _ = env.reset(seed=CONFIG["SEED"])
    eq, rets = [], []
    
    # 等待UI开始信号
    while not ui.running:
        time.sleep(0.1)
        ui.root.update()
    
    ui.log_message(f"开始{name}评估...")
    
    step_count = 0
    while True:
        if not ui.running:
            ui.log_message("评估被用户中断")
            break
            
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))
        eq.append(env.asset)
        if len(env.equity_curve) >= 2:
            rets.append(env.returns[-1])
            
        step_count += 1
        if step_count % 10 == 0:  # 每10步更新一次UI，避免过于频繁
            time.sleep(0.05)  # 稍微减慢速度以便观察
            
        if done: 
            ui.log_message(f"{name}评估完成")
            break

    equity = pd.Series(eq)
    returns = pd.Series(rets).fillna(0.0)
    
    # 显示最终评估结果
    ui.log_message(f"=== {name} 评估结果 ===")
    ui.log_message(f"最终权益: {equity.iloc[-1]:.2f}")
    ui.log_message(f"年化收益: {annualized_return(equity):.2%}")
    ui.log_message(f"Sharpe : {sharpe_ratio(returns):.3f}")
    ui.log_message(f"Sortino: {sortino_ratio(returns):.3f}")
    ui.log_message(f"最大回撤: {max_drawdown(equity):.2%}")

    # === NEW: 导出动作日志 ===
    os.makedirs(LOG_DIR, exist_ok=True)
    out_csv = os.path.join(LOG_DIR, f"actions_{name.replace(' ', '_')}.csv")
    action_df = env.get_action_log_df(trades_only=LOG_TRADES_ONLY)
    action_df.to_csv(out_csv, index=False)
    ui.log_message(f"[ACTIONS] saved to {out_csv}")

def main():
    np.random.seed(CONFIG["SEED"])
    df = load_features(CONFIG)

    train_df, test_df = split_by_date(df, CONFIG)
    if len(train_df) == 0: raise ValueError("[ERROR] 训练集为空")
    if len(test_df) == 0: raise ValueError("[ERROR] 测试集为空")

    print(f"训练区间: {train_df[CONFIG['TIME_COL']].iloc[0]} -> {train_df[CONFIG['TIME_COL']].iloc[-1]} | 行数={len(train_df)}")
    print(f"测试区间: {test_df[CONFIG['TIME_COL']].iloc[0]} -> {test_df[CONFIG['TIME_COL']].iloc[-1]} | 行数={len(test_df)}")

    # === NEW: 创建UI ===
    root = tk.Tk()
    
    # 先创建测试环境用于UI
    test_env_for_ui = RLTradingEnv(test_df, CONFIG)
    ui = TradingUI(root, test_env_for_ui)
    
    # 重新创建训练环境，绑定UI回调
    def ui_callback(env, step_info):
        if ui.running:
            current_time = step_info["timestamp"]
            ui.update_display(env, current_time, step_info)
    
    train_env = DummyVecEnv([lambda: RLTradingEnv(train_df, CONFIG, ui_callback)])
    test_env = RLTradingEnv(test_df, CONFIG, ui_callback)

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

    # 在单独的线程中运行训练和评估
    def run_training():
        ui.log_message("开始PPO模型训练...")
        model.learn(total_timesteps=CONFIG["PPO_TIMESTEPS"], callback=EquityLogger())
        ui.log_message("模型训练完成!")
        
        # 训练完成后开始评估
        evaluate_with_ui(test_env, model, "PPO+SMA+ATVMF+MFI(Test, oldstyle)", ui)

    # 启动训练线程
    train_thread = threading.Thread(target=run_training, daemon=True)
    train_thread.start()

    # 启动UI主循环
    root.mainloop()

if __name__ == "__main__":
    main()