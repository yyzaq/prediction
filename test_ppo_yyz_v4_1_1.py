"""
4.1.1版本
4.1.x版本主要处理的是4.1版本买卖量化导致的训练困难
4.1.1使用了课程学习的方式来训练模型
"""

"""
连续动作版本的PPO股价预测 - 课程学习增强版
新增功能：
1. 多阶段课程学习训练
2. 从简单到复杂的市场环境
3. 渐进式难度增加
4. 自动阶段过渡
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
from test_ppo_yyz_v4_1 import (
    load_features, split_by_date, max_drawdown, annualized_return,
    sharpe_ratio, sortino_ratio, RLTradingEnv, EpisodeClock, 
    evaluate, EquityLogger
)
warnings.filterwarnings('ignore')

# ==================== 课程学习配置 ====================
COURSE_LEARNING_CONFIG = {
    "total_training_phases": 3,
    "performance_threshold": 0.15,  # 阶段性能阈值（年化收益率）
    "min_episodes_per_phase": 5,
    "max_phases_without_improvement": 2,
    
    "phase_configs": [
        # 阶段1：简单环境
        {
            "name": "phase1_easy",
            "description": "低波动率简单市场",
            "data_filters": {
                "volatility_filter": "low",  # 低波动率时期
                "trend_strength": "high",    # 强趋势市场
            },
            "env_modifications": {
                "fee_rate_multiplier": 0.5,      # 降低交易费用
                "position_change_penalty": 0.0001,  # 减少仓位变化惩罚
                "min_position_change": 0.1,      # 增大最小变化阈值
                "enable_technical_filters": False, # 禁用技术指标过滤
                "episode_length_multiplier": 0.5, # 缩短episode
            },
            "training": {
                "timesteps": 50_000,
                "learning_rate": 3e-4,
            }
        },
        # 阶段2：中等环境
        {
            "name": "phase2_medium", 
            "description": "正常市场环境",
            "data_filters": {
                "volatility_filter": "medium",
                "trend_strength": "medium",
            },
            "env_modifications": {
                "fee_rate_multiplier": 0.8,
                "position_change_penalty": 0.0005,
                "min_position_change": 0.07,
                "enable_technical_filters": True,
                "episode_length_multiplier": 0.8,
            },
            "training": {
                "timesteps": 30_000,
                "learning_rate": 2e-4,
            }
        },
        # 阶段3：困难环境
        {
            "name": "phase3_hard",
            "description": "全难度真实市场",
            "data_filters": {
                "volatility_filter": "all",  # 所有波动率
                "trend_strength": "all",     # 所有趋势强度
            },
            "env_modifications": {
                "fee_rate_multiplier": 1.0,
                "position_change_penalty": 0.001,
                "min_position_change": 0.05,
                "enable_technical_filters": True,
                "episode_length_multiplier": 1.0,
            },
            "training": {
                "timesteps": 20_000,
                "learning_rate": 1e-4,
            }
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

    "POSITION_CHANGE_PENALTY": 0.001,
    "MIN_POSITION_CHANGE": 0.05,
    
    # 课程学习配置
    "COURSE_LEARNING": COURSE_LEARNING_CONFIG
}

REQUIRED_COLS = [
    "timestamp","open","high","low","close","volume",
    "sma_s","sma_l","mfi","atvmf","atvmf_ma","cross_state","ret1"
]

# ==================== 课程学习数据处理器 ====================
class CourseLearningDataProcessor:
    """课程学习数据处理器 - 根据阶段选择不同难度的训练数据"""
    
    def __init__(self, df: pd.DataFrame, cfg: Dict[str, Any]):
        self.df = df.copy()
        self.cfg = cfg
        self._precompute_market_metrics()
    
    def _precompute_market_metrics(self):
        """预计算市场指标用于数据筛选"""
        # 计算波动率（滚动标准差）
        returns = self.df['close'].pct_change()
        self.df['volatility_20'] = returns.rolling(window=20).std()
        self.df['volatility_50'] = returns.rolling(window=50).std()
        
        # 计算趋势强度（ADX近似）
        high, low, close = self.df['high'], self.df['low'], self.df['close']
        tr = np.maximum(high - low, 
                       np.maximum(abs(high - close.shift()), 
                                 abs(low - close.shift())))
        self.df['trend_strength'] = tr.rolling(14).mean() / close * 100
        
        # 填充NaN值
        self.df = self.df.fillna(method='bfill').fillna(method='ffill')
    
    def get_phase_data(self, phase_config: Dict) -> pd.DataFrame:
        """获取指定阶段的训练数据"""
        df_filtered = self.df.copy()
        
        # 应用波动率过滤
        vol_filter = phase_config["data_filters"]["volatility_filter"]
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
        # "all" 不进行过滤
        
        # 应用趋势强度过滤
        trend_filter = phase_config["data_filters"]["trend_strength"]
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
        # "all" 不进行过滤
        
        print(f"[Course Learning] Phase {phase_config['name']}: "
              f"原始数据 {len(self.df)} -> 过滤后 {len(df_filtered)} 行 "
              f"({len(df_filtered)/len(self.df)*100:.1f}%)")
        
        return df_filtered.reset_index(drop=True)

# ==================== 课程学习环境包装器 ====================
class CourseLearningTradingEnv(gym.Wrapper):
    """课程学习环境包装器 - 根据训练阶段调整环境难度"""
    
    def __init__(self, env, phase_config: Dict):
        super().__init__(env)
        self.phase_config = phase_config
        self.env_modifications = phase_config["env_modifications"]
        self._apply_phase_modifications()
    
    def _apply_phase_modifications(self):
        """应用阶段特定的环境修改"""
        # 调整交易费用
        fee_multiplier = self.env_modifications["fee_rate_multiplier"]
        self.env.cfg["FEE_RATE"] = self.env.cfg.get("BASE_FEE_RATE", 0.001) * fee_multiplier
        
        # 调整仓位变化惩罚
        self.env.cfg["POSITION_CHANGE_PENALTY"] = self.env_modifications["position_change_penalty"]
        
        # 调整最小仓位变化阈值
        self.env.cfg["MIN_POSITION_CHANGE"] = self.env_modifications["min_position_change"]
        
        # 调整episode长度
        length_multiplier = self.env_modifications["episode_length_multiplier"]
        original_episode_minutes = self.env.cfg.get("BASE_EPISODE_MINUTES", 
                                                   self.env.cfg["EPISODE_MINUTES"])
        self.env.cfg["EPISODE_MINUTES"] = int(original_episode_minutes * length_multiplier)
        
        # 保存技术指标过滤状态
        self.enable_technical_filters = self.env_modifications["enable_technical_filters"]
        
        print(f"[Course Learning] 环境调整: "
              f"费用率x{fee_multiplier}, "
              f"仓位惩罚{self.env_modifications['position_change_penalty']}, "
              f"技术过滤{self.enable_technical_filters}")

# ==================== 课程学习回调函数 ====================
class CourseLearningCallback(BaseCallback):
    """课程学习回调 - 监控训练进度并管理阶段过渡"""
    
    def __init__(self, main_trainer, verbose=0):
        super().__init__(verbose)
        self.main_trainer = main_trainer
        self.phase_performance = []
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """在每个rollout结束时评估性能"""
        current_phase = self.main_trainer.current_phase
        if current_phase >= len(self.main_trainer.phase_configs) - 1:
            return  # 最后一阶段不需要评估过渡
        
        # 定期评估性能
        if self.num_timesteps % (self.main_trainer.phase_configs[current_phase]["training"]["timesteps"] // 5) == 0:
            performance = self._evaluate_current_performance()
            self.phase_performance.append(performance)
            
            if self.verbose:
                print(f"[Course Learning] 阶段 {current_phase} 性能评估: {performance:.4f}")
    
    def _evaluate_current_performance(self) -> float:
        """评估当前模型性能"""
        # 使用验证环境进行评估
        if hasattr(self.main_trainer, 'validation_env'):
            mean_reward, _ = evaluate_policy(
                self.model, 
                self.main_trainer.validation_env,
                n_eval_episodes=3,
                deterministic=True
            )
            return mean_reward
        return 0.0

# ==================== 课程学习主训练器 ====================
class CourseLearningTrainer:
    """课程学习主训练器 - 管理多阶段训练流程"""
    
    def __init__(self, config: Dict[str, Any], full_train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.config = config
        self.full_train_df = full_train_df
        self.test_df = test_df
        self.phase_configs = config["COURSE_LEARNING"]["phase_configs"]
        self.current_phase = 0
        self.best_model = None
        self.performance_history = []
        
        # 初始化数据处理器
        self.data_processor = CourseLearningDataProcessor(full_train_df, config)
        
        # 保存原始配置
        self.config["BASE_FEE_RATE"] = config["FEE_RATE"]
        self.config["BASE_EPISODE_MINUTES"] = config["EPISODE_MINUTES"]


    def plot_training_progress(self):
        """绘制课程学习训练进度"""
        if not hasattr(self, 'performance_history') or not self.performance_history:
            print("⚠️ 没有训练进度数据可绘制")
            return
            
        try:
            phases = [f"Phase{i+1}\n{hist['name']}" for i, hist in enumerate(self.performance_history)]
            performances = [hist['performance'] for hist in self.performance_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(phases, performances, 'o-', linewidth=2, markersize=8)
            plt.title('课程学习训练进度')
            plt.xlabel('训练阶段')
            plt.ylabel('性能指标')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️ 绘制训练进度失败: {e}")
        return
    
    def train(self):
        """执行多阶段课程学习训练"""
        print("=" * 60)
        print("开始课程学习训练")
        print("=" * 60)
        
        model = None
        
        for phase_idx, phase_config in enumerate(self.phase_configs):
            self.current_phase = phase_idx
            print(f"\n🎯 开始训练阶段 {phase_idx + 1}/{len(self.phase_configs)}: {phase_config['name']}")
            print(f"📝 {phase_config['description']}")
            
            # 获取阶段特定的训练数据
            phase_train_df = self.data_processor.get_phase_data(phase_config)
            
            if len(phase_train_df) < 100:
                print(f"⚠️  阶段 {phase_idx} 数据过少 ({len(phase_train_df)} 行)，跳过该阶段")
                continue
            
            # 创建阶段特定的环境
            env = self._create_phase_environment(phase_train_df, phase_config)
            
            # 创建或继续训练模型
            if model is None:
                # 第一阶段创建新模型
                model = self._create_model(env, phase_config)
            else:
                # 后续阶段使用前一阶段的模型继续训练
                model.set_env(env)
                model.learning_rate = phase_config["training"]["learning_rate"]
            
            # 训练当前阶段
            phase_timesteps = phase_config["training"]["timesteps"]
            print(f"🔧 训练参数: LR={phase_config['training']['learning_rate']:.2e}, "
                  f"Timesteps={phase_timesteps}")
            
            model.learn(
                total_timesteps=phase_timesteps,
                callback=CourseLearningCallback(self, verbose=1),
                reset_num_timesteps=False  # 重要：不重置时间步计数
            )
            
            # 阶段结束评估
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
        """创建阶段特定的训练环境"""
        base_env = RLTradingEnv(train_df, self.config)
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
        test_env = RLTradingEnv(self.test_df, self.config)
    
    # 移除 return_episode_rewards 参数
        mean_reward, std_reward = evaluate_policy(
            model, 
            test_env,
            n_eval_episodes=3,
            deterministic=True,
        # 移除: return_episode_rewards=True
        )
        
        print(f"📊 阶段 '{phase_name}' 测试评估: {mean_reward:.4f} ± {std_reward:.4f}")
        return mean_reward

# ==================== 修改主函数以使用课程学习 ====================
def main():
    np.random.seed(CONFIG["SEED"])
    
    # 加载数据
    df = load_features(CONFIG)
    train_df, test_df = split_by_date(df, CONFIG)
    
    if len(train_df) == 0: 
        raise ValueError("[ERROR] 训练集为空")
    if len(test_df) == 0: 
        raise ValueError("[ERROR] 测试集为空")
    
    print(f"训练区间: {train_df[CONFIG['TIME_COL']].iloc[0]} -> {train_df[CONFIG['TIME_COL']].iloc[-1]} | 行数={len(train_df)}")
    print(f"测试区间: {test_df[CONFIG['TIME_COL']].iloc[0]} -> {test_df[CONFIG['TIME_COL']].iloc[-1]} | 行数={len(test_df)}")
    
    # 使用课程学习训练器
    course_trainer = CourseLearningTrainer(CONFIG, train_df, test_df)
    model = course_trainer.train()
    
    # 绘制训练进度
    course_trainer.plot_training_progress()
    test_env = RLTradingEnv(test_df, CONFIG)  # 添加这行
    
    # 最终评估
    print("\n" + "="*60)
    print("最终模型评估")
    print("="*60)
    evaluate(test_env, model, "PPO with Course Learning (Final)")
    
    # 保存最终模型
    os.makedirs(LOG_DIR, exist_ok=True)
    model_path = os.path.join(LOG_DIR, "ppo_course_learning_final.zip")
    model.save(model_path)
    print(f"💾 模型已保存至: {model_path}")

# 保留原有的工具函数（load_features, split_by_date, 评估指标, RLTradingEnv, evaluate等）
# 这些函数保持不变，与原始代码相同

if __name__ == "__main__":
    main()