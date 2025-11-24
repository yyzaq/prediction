"""
半定量版本的PPO股价预测 - 9个离散仓位节点
- 动作空间改为离散仓位节点：[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
- 增加动作空间粒度，提高策略灵活性
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
import warnings
warnings.filterwarnings('ignore')

# ==================== 半定量配置 ====================
SEMI_DISCRETE_CONFIG = {
    # 离散仓位节点设置 - 9个节点
    "POSITION_NODES": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
    
    # 训练参数
    "total_training_phases": 3,
    "performance_threshold": 0.15,
    
    "phase_configs": [
        # 阶段1：宽松环境
        {
            "name": "phase1_easy",
            "description": "宽松交易环境",
            "data_filters": {"volatility_filter": "low", "trend_strength": "high"},
            "env_modifications": {
                "fee_rate_multiplier": 0.5,
                "position_change_penalty": 0.0001,
                "enable_technical_filters": False,
                "episode_length_multiplier": 0.5,
            },
            "training": {"timesteps": 50_000, "learning_rate": 3e-4}
        },
        # 阶段2：中等环境
        {
            "name": "phase2_medium",
            "description": "中等交易环境", 
            "data_filters": {"volatility_filter": "medium", "trend_strength": "medium"},
            "env_modifications": {
                "fee_rate_multiplier": 0.8,
                "position_change_penalty": 0.0005,
                "enable_technical_filters": True,
                "episode_length_multiplier": 0.8,
            },
            "training": {"timesteps": 30_000, "learning_rate": 2e-4}
        },
        # 阶段3：严格环境
        {
            "name": "phase3_hard",
            "description": "严格交易环境",
            "data_filters": {"volatility_filter": "all", "trend_strength": "all"},
            "env_modifications": {
                "fee_rate_multiplier": 1.0,
                "position_change_penalty": 0.001,
                "enable_technical_filters": True,
                "episode_length_multiplier": 1.0,
            },
            "training": {"timesteps": 20_000, "learning_rate": 1e-4}
        }
    ]
}

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
    "MAX_POSITION_SIZE": 1.0,
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

    # 半定量相关参数
    "POSITION_CHANGE_PENALTY": 0.001,
    
    # 半定量配置
    "SEMI_DISCRETE": SEMI_DISCRETE_CONFIG
}

REQUIRED_COLS = [
    "timestamp","open","high","low","close","volume",
    "sma_s","sma_l","mfi","atvmf","atvmf_ma","cross_state","ret1"
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
    df = df.dropna(subset=["timestamp","close"]).sort_values("timestamp").reset_index(drop=True)
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

    print("[DEBUG] 全量:", df[t].min(), "->", df[t].max(), "| 行数=", len(df))
    print("[DEBUG] 训练切片:", train_start, "->", train_end, "| 行数=", len(tr_raw))
    print("[DEBUG] 测试切片:", test_start, "->", test_end, "| 行数=", len(te_raw))

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

# ==================== 强化学习环境（半定量版本）====================
@dataclass
class EpisodeClock:
    """Episode时钟"""
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

        # 特征数据
        atvmf = self.df["atvmf"].values.astype(float)
        self.atvmf_z = (atvmf - np.nanmean(atvmf)) / (np.nanstd(atvmf) + 1e-9)
        self.ret1 = self.df["ret1"].values.astype(float)
        self.cross_state = self.df["cross_state"].values.astype(float)
        self.mfi = self.df["mfi"].values.astype(float)
        self.atvmf = atvmf
        self.atvmf_ma = self.df["atvmf_ma"].values.astype(float)

        # === MODIFIED: 9个离散仓位节点 ===
        self.position_nodes = cfg["SEMI_DISCRETE"]["POSITION_NODES"]
        self.action_space = spaces.Discrete(len(self.position_nodes))
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_win*4 + 1,), dtype=np.float32
        )

        self._make_episode_slices()
        self.reset()

    def _make_episode_slices(self):
        """创建episode切片"""
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
        """获取观察状态"""
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
        """重置环境"""
        super().reset(seed=seed)
        ep = self.np_random.integers(0, len(self.episodes))
        self.ep_slice = self.episodes[ep]
        self.idx = self.ep_slice.start_idx + self.obs_win
        self.position = 0.0
        self.cash = self.cfg["INIT_CASH"]
        self.asset = self.cfg["INIT_CASH"]
        self.equity_curve = [self.asset]
        self.returns = [0.0]
        self.max_equity = self.asset
        self.episode_id = getattr(self, 'episode_id', -1) + 1
        self._step_logs = []
        return self._get_obs(self.idx), {}

    def _log_step(self, i, action_idx, target_position, position_change, 
                  price_now, price_next, fee, pnl_step, reward):
        """记录步骤日志"""
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
            # 对于离散动作，只有仓位变化时才算交易
            df = df[df["position_change"] != 0].reset_index(drop=True)
        return df

    def step(self, action: int):
        """环境步骤"""
        i = self.idx
        price_now = self.close[i]
        price_next = self.close[i+1] if i+1 < len(self.close) else price_now
        
        # === MODIFIED: 9个离散动作处理 ===
        action_idx = int(action)
        target_position = self.position_nodes[action_idx]
        pos_before = float(self.position)
        
        # 技术指标过滤
        cross = self.cross_state[i]
        mfi_t = self.mfi[i]
        av = self.atvmf[i]
        av_ma = self.atvmf_ma[i]
        
        # 检查技术指标是否有效
        technical_valid = not (np.isnan(mfi_t) or np.isnan(av) or np.isnan(av_ma))
        
        if technical_valid:
            allow_long = (cross == 1) and (mfi_t < self.cfg["MFI_LOW"]) and (av > av_ma)
            allow_short = (cross == -1) and (mfi_t > self.cfg["MFI_HIGH"]) and (av < av_ma)
        else:
            allow_long = allow_short = False
        
        # 应用技术过滤
        if not allow_long and target_position > 0:
            target_position = 0.0  # 不允许做多时设为空仓
        elif not allow_short and target_position < 0:
            target_position = 0.0  # 不允许做空时设为空仓
        
        # 计算仓位变化和费用
        position_change = target_position - pos_before
        abs_change = abs(position_change)
        
        if abs_change > 0:  # 只有仓位变化时才收费
            trade_value = abs_change * price_now * self.cfg["MAX_POSITION_SIZE"]
            fee = trade_value * self.cfg["FEE_RATE"]
            self.position = target_position
        else:
            fee = 0.0
        
        # 计算收益和奖励
        pnl_step = (price_next - price_now) * self.position * self.cfg["MAX_POSITION_SIZE"]
        rew = pnl_step - fee
        
        # 仓位变化惩罚
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

        # 记录日志
        self._log_step(i, action_idx, target_position, position_change,
                       price_now, price_next, fee, pnl_step, rew)

        self.idx += 1
        done = self.idx >= self.ep_slice.end_idx - 2
        return self._get_obs(self.idx), float(rew), bool(done), False, {}

# ==================== 课程学习组件 ====================
class CourseLearningDataProcessor:
    """课程学习数据处理器"""
    
    def __init__(self, df: pd.DataFrame, cfg: Dict[str, Any]):
        self.df = df.copy()
        self.cfg = cfg
        self._precompute_market_metrics()
    
    def _precompute_market_metrics(self):
        """预计算市场指标"""
        returns = self.df['close'].pct_change()
        self.df['volatility_20'] = returns.rolling(window=20).std()
        self.df['volatility_50'] = returns.rolling(window=50).std()
        
        high, low, close = self.df['high'], self.df['low'], self.df['close']
        tr = np.maximum(high - low, 
                       np.maximum(abs(high - close.shift()), 
                                 abs(low - close.shift())))
        self.df['trend_strength'] = tr.rolling(14).mean() / close * 100
        self.df = self.df.fillna(method='bfill').fillna(method='ffill')
    
    def get_phase_data(self, phase_config: Dict) -> pd.DataFrame:
        """获取阶段数据"""
        df_filtered = self.df.copy()
        vol_filter = phase_config["data_filters"]["volatility_filter"]
        trend_filter = phase_config["data_filters"]["trend_strength"]
        
        # 波动率过滤
        if vol_filter == "low":
            vol_threshold = self.df['volatility_20'].quantile(0.3)
            df_filtered = df_filtered[df_filtered['volatility_20'] <= vol_threshold]
        elif vol_filter == "medium":
            vol_low = self.df['volatility_20'].quantile(0.3)
            vol_high = self.df['volatility_20'].quantile(0.7)
            df_filtered = df_filtered[
                (df_filtered['volatility_20'] >= vol_low) & 
                (df_filtered['volatility_20'] <= vol_high)
            ]
        
        # 趋势强度过滤
        if trend_filter == "high":
            trend_threshold = self.df['trend_strength'].quantile(0.7)
            df_filtered = df_filtered[df_filtered['trend_strength'] >= trend_threshold]
        elif trend_filter == "medium":
            trend_low = self.df['trend_strength'].quantile(0.3)
            trend_high = self.df['trend_strength'].quantile(0.7)
            df_filtered = df_filtered[
                (df_filtered['trend_strength'] >= trend_low) & 
                (df_filtered['trend_strength'] <= trend_high)
            ]
        
        print(f"[Course Learning] Phase {phase_config['name']}: "
              f"原始数据 {len(self.df)} -> 过滤后 {len(df_filtered)} 行")
        
        return df_filtered.reset_index(drop=True)

class CourseLearningTradingEnv(gym.Wrapper):
    """课程学习环境包装器"""
    
    def __init__(self, env, phase_config: Dict):
        super().__init__(env)
        self.phase_config = phase_config
        self.env_modifications = phase_config["env_modifications"]
        self._apply_phase_modifications()
    
    def _apply_phase_modifications(self):
        """应用阶段修改"""
        fee_multiplier = self.env_modifications["fee_rate_multiplier"]
        self.env.cfg["FEE_RATE"] = self.env.cfg.get("BASE_FEE_RATE", 0.001) * fee_multiplier
        
        self.env.cfg["POSITION_CHANGE_PENALTY"] = self.env_modifications["position_change_penalty"]
        
        length_multiplier = self.env_modifications["episode_length_multiplier"]
        original_episode_minutes = self.env.cfg.get("BASE_EPISODE_MINUTES", 
                                                   self.env.cfg["EPISODE_MINUTES"])
        self.env.cfg["EPISODE_MINUTES"] = int(original_episode_minutes * length_multiplier)
        
        self.enable_technical_filters = self.env_modifications["enable_technical_filters"]
        
        print(f"[Course Learning] 环境调整: 费用率x{fee_multiplier}, "
              f"技术过滤{self.enable_technical_filters}")

class CourseLearningCallback(BaseCallback):
    """课程学习回调函数"""
    
    def __init__(self, main_trainer, verbose=0):
        super().__init__(verbose)
        self.main_trainer = main_trainer
    
    def _on_step(self) -> bool:
        return True

class CourseLearningTrainer:
    """课程学习主训练器"""
    
    def __init__(self, config: Dict[str, Any], full_train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.config = config
        self.full_train_df = full_train_df
        self.test_df = test_df
        self.phase_configs = config["SEMI_DISCRETE"]["phase_configs"]
        self.current_phase = 0
        self.best_model = None
        self.performance_history = []
        self.data_processor = CourseLearningDataProcessor(full_train_df, config)
        self.config["BASE_FEE_RATE"] = config["FEE_RATE"]
        self.config["BASE_EPISODE_MINUTES"] = config["EPISODE_MINUTES"]
    
    def train(self):
        """执行训练"""
        print("=" * 60)
        print("开始半定量课程学习训练 - 9个离散仓位节点")
        print(f"仓位节点: {self.config['SEMI_DISCRETE']['POSITION_NODES']}")
        print("=" * 60)
        
        model = None
        
        for phase_idx, phase_config in enumerate(self.phase_configs):
            self.current_phase = phase_idx
            print(f"\n🎯 开始训练阶段 {phase_idx + 1}/{len(self.phase_configs)}: {phase_config['name']}")
            print(f"📝 {phase_config['description']}")
            
            # 获取阶段数据
            phase_train_df = self.data_processor.get_phase_data(phase_config)
            
            if len(phase_train_df) < 100:
                print(f"⚠️ 阶段数据过少，跳过")
                continue
            
            # 创建环境
            env = self._create_phase_environment(phase_train_df, phase_config)
            
            # 创建或继续训练模型
            if model is None:
                model = self._create_model(env, phase_config)
            else:
                model.set_env(env)
                model.learning_rate = phase_config["training"]["learning_rate"]
            
            # 训练
            phase_timesteps = phase_config["training"]["timesteps"]
            print(f"🔧 训练参数: LR={phase_config['training']['learning_rate']:.2e}, "
                  f"Timesteps={phase_timesteps}")
            
            model.learn(
                total_timesteps=phase_timesteps,
                callback=CourseLearningCallback(self, verbose=1),
                reset_num_timesteps=False
            )
            
            # 评估
            phase_performance = self._evaluate_phase(model, phase_config["name"])
            self.performance_history.append({
                'phase': phase_idx,
                'name': phase_config['name'],
                'performance': phase_performance
            })
            
            print(f"✅ 阶段 {phase_idx + 1} 完成 - 性能: {phase_performance:.4f}")
        
        self.best_model = model
        return model
    
    def _create_phase_environment(self, train_df: pd.DataFrame, phase_config: Dict) -> DummyVecEnv:
        """创建阶段环境"""
        base_env = SemiDiscreteTradingEnv(train_df, self.config)
        phase_env = CourseLearningTradingEnv(base_env, phase_config)
        return DummyVecEnv([lambda: phase_env])
    
    def _create_model(self, env, phase_config: Dict) -> PPO:
        """创建PPO模型"""
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=phase_config["training"]["learning_rate"],
            gamma=self.config["GAMMA"],
            n_steps=self.config["N_STEPS"],
            batch_size=self.config["BATCH_SIZE"],
            n_epochs=self.config["N_EPOCHS"],
            clip_range=self.config["CLIP_RANGE"],
            verbose=1,
            seed=self.config["SEED"],
            policy_kwargs=dict(net_arch=[256, 256]),
            ent_coef=self.config.get("ENT_COEF", 0.02),
        )
    
    def _evaluate_phase(self, model: PPO, phase_name: str) -> float:
        """评估阶段性能"""
        test_env = SemiDiscreteTradingEnv(self.test_df, self.config)
        mean_reward, std_reward = evaluate_policy(
            model, 
            test_env,
            n_eval_episodes=3,
            deterministic=True
        )
        
        print(f"📊 阶段 '{phase_name}' 测试评估: {mean_reward:.4f} ± {std_reward:.4f}")
        return mean_reward

# ==================== 评估函数 ====================
class EquityLogger(BaseCallback):
    """权益日志回调"""
    def _on_step(self) -> bool: 
        return True

def evaluate(env: SemiDiscreteTradingEnv, model: PPO, name: str):
    """评估模型"""
    obs, _ = env.reset(seed=CONFIG["SEED"])
    eq, rets = [], []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        eq.append(env.asset)
        if len(env.equity_curve) >= 2:
            rets.append(env.returns[-1])
        if done: 
            break

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
        position_nodes = CONFIG["SEMI_DISCRETE"]["POSITION_NODES"]
        print(f"仓位节点: {position_nodes}")
        print(f"仓位分布:")
        for pos in position_nodes:
            count = len(action_df[action_df['position_after'] == pos])
            percentage = count / len(action_df) * 100
            print(f"  {pos}: {count}次 ({percentage:.1f}%)")

    # 可视化
    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(equity.values)
    plt.title(f"{name} 权益曲线")
    plt.ylabel("权益")
    
    plt.subplot(4, 1, 2)
    plt.plot(action_df['position_after'].values)
    plt.title("仓位变化")
    plt.ylabel("仓位")
    plt.ylim(-1.1, 1.1)
    
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
    
    plt.tight_layout()
    plt.show()

    # 导出日志
    os.makedirs(LOG_DIR, exist_ok=True)
    out_csv = os.path.join(LOG_DIR, f"actions_{name.replace(' ', '_')}_9nodes.csv")
    action_df = env.get_action_log_df(trades_only=LOG_TRADES_ONLY)
    action_df.to_csv(out_csv, index=False)
    print(f"[ACTIONS] 保存至 {out_csv}")
    if not action_df.empty:
        print(action_df.head(10))

# ==================== 主函数 ====================
def main():
    """主函数"""
    np.random.seed(CONFIG["SEED"])
    
    # 加载数据
    df = load_features(CONFIG)
    train_df, test_df = split_by_date(df, CONFIG)
    
    if len(train_df) == 0: 
        raise ValueError("[ERROR] 训练集为空")
    if len(test_df) == 0: 
        raise ValueError("[ERROR] 测试集为空")
    
    print(f"训练区间: {train_df[CONFIG['TIME_COL']].iloc[0]} -> {train_df[CONFIG['TIME_COL']].iloc[-1]}")
    print(f"测试区间: {test_df[CONFIG['TIME_COL']].iloc[0]} -> {test_df[CONFIG['TIME_COL']].iloc[-1]}")
    
    # 使用课程学习训练器
    course_trainer = CourseLearningTrainer(CONFIG, train_df, test_df)
    model = course_trainer.train()
    
    # 最终评估
    print("\n" + "="*60)
    print("最终模型评估 - 9个离散仓位节点")
    print("="*60)
    test_env = SemiDiscreteTradingEnv(test_df, CONFIG)
    evaluate(test_env, model, "PPO Semi-Discrete 9 Nodes (Final)")
    
    # 保存模型
    os.makedirs(LOG_DIR, exist_ok=True)
    model_path = os.path.join(LOG_DIR, "ppo_semi_discrete_9nodes_final.zip")
    model.save(model_path)
    print(f"💾 模型已保存至: {model_path}")

if __name__ == "__main__":
    main()