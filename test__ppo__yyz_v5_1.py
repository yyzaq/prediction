"""
PPO模型投票集成 - 在不修改原代码基础上的增强版本
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from math import sqrt
import torch

# ==================== 集成学习配置 ====================
ENSEMBLE_CONFIG = {
    "NUM_MODELS": 3,                      # 集成模型数量
    "TRAINING_RATIO": 0.7,                # 每个模型训练原步长的比例
    "VOTE_METHOD": "majority",            # 投票方式: majority, weighted, soft
    "SEED_OFFSET": 100,                   # 不同模型间的种子偏移量
}

# 保持原有的CONFIG不变
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
    "POSITION_NODES": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
    "POSITION_CHANGE_PENALTY": 0.001,
    "PPO_TIMESTEPS": 200000,
    "PPO_LR": 3e-4,
    "GAMMA": 0.99,
    "N_STEPS": 2048,
    "BATCH_SIZE": 256,
    "N_EPOCHS": 10,
    "CLIP_RANGE": 0.2,
    "ENT_COEF": 0.05,
    "SEED": 42,
}

# ==================== 集成模型管理器 ====================
class EnsemblePPOManager:
    """
    PPO模型集成管理器 - 包装原有代码，不修改核心逻辑
    """
    
    def __init__(self, ensemble_config: Dict[str, Any], base_config: Dict[str, Any]):
        self.ensemble_config = ensemble_config
        self.base_config = base_config
        self.models = []  # 存储所有集成模型
        self.model_performances = []  # 记录每个模型的性能
        self.training_histories = []  # 训练历史记录
        
    def create_individual_model(self, model_id: int, train_env) -> PPO:
        """创建单个PPO模型，使用不同的随机种子"""
        # 复制基础配置
        model_config = self.base_config.copy()
        
        # 为每个模型设置不同的随机种子
        model_config["SEED"] = self.base_config["SEED"] + model_id * self.ensemble_config["SEED_OFFSET"]
        
        print(f"🎯 创建模型 {model_id + 1}/{self.ensemble_config['NUM_MODELS']}")
        print(f"   随机种子: {model_config['SEED']}")
        
        # 使用原有参数创建PPO模型
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=model_config["PPO_LR"],
            gamma=model_config["GAMMA"],
            n_steps=model_config["N_STEPS"],
            batch_size=model_config["BATCH_SIZE"],
            n_epochs=model_config["N_EPOCHS"],
            clip_range=model_config["CLIP_RANGE"],
            ent_coef=model_config["ENT_COEF"],
            verbose=1,
            seed=model_config["SEED"],
            policy_kwargs=dict(
                net_arch=[128, 128],
                activation_fn=torch.nn.ReLU,
            )
        )
        return model
    
    def train_ensemble(self, train_env, test_env, original_train_df, original_test_df):
        """训练集成模型集合"""
        num_models = self.ensemble_config["NUM_MODELS"]
        total_timesteps = int(self.base_config["PPO_TIMESTEPS"] * self.ensemble_config["TRAINING_RATIO"])
        
        print("=" * 60)
        print("🚀 开始训练PPO模型集成")
        print(f"📊 集成规模: {num_models} 个模型")
        print(f"⏱️  每个模型训练步数: {total_timesteps}")
        print("=" * 60)
        
        for model_id in range(num_models):
            print(f"\n📚 训练第 {model_id + 1}/{num_models} 个模型...")
            
            # 创建单个模型
            model = self.create_individual_model(model_id, train_env)
            
            # 训练回调：监控训练进度
            class ModelTrainingCallback(BaseCallback):
                def __init__(self, model_id, verbose=0):
                    super().__init__(verbose)
                    self.model_id = model_id
                    self.step_count = 0
                
                def _on_step(self):
                    self.step_count += 1
                    if self.step_count % 5000 == 0:
                        print(f"  模型 {self.model_id + 1} - 训练进度: {self.step_count}/{total_timesteps}")
                    return True
            
            # 训练模型
            callback = ModelTrainingCallback(model_id)
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                reset_num_timesteps=False
            )
            
            # 评估单个模型性能
            model_performance = self.evaluate_single_model(model, test_env, f"Model_{model_id + 1}")
            self.model_performances.append(model_performance)
            self.models.append(model)
            
            print(f"✅ 模型 {model_id + 1} 训练完成")
            print(f"   最终权益: {model_performance['final_equity']:.2f}")
            print(f"   夏普比率: {model_performance['sharpe_ratio']:.3f}")
        
        print(f"\n🎉 集成训练完成! 共训练 {len(self.models)} 个模型")
        
    def evaluate_single_model(self, model, test_env, model_name: str) -> Dict[str, float]:
        """评估单个模型性能"""
        # 使用原有的环境类
        from semi_discrete_env import SemiDiscreteTradingEnv  # 假设原有环境类在这个文件中
        
        obs, _ = test_env.reset()
        equity_curve = []
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = test_env.step(action)
            equity_curve.append(test_env.asset)
            if done:
                break
        
        # 计算性能指标
        equity_series = pd.Series(equity_curve)
        returns = pd.Series([equity_curve[i] / equity_curve[i-1] - 1 for i in range(1, len(equity_curve))])
        
        performance = {
            'final_equity': equity_curve[-1],
            'total_return': (equity_curve[-1] / equity_curve[0] - 1) * 100,
            'sharpe_ratio': self.calculate_sharpe(returns),
            'max_drawdown': self.calculate_max_drawdown(equity_series),
            'model_name': model_name
        }
        
        return performance
    
    def calculate_sharpe(self, returns: pd.Series, freq_per_year=365*24*4) -> float:
        """计算夏普比率"""
        if returns.std() == 0 or returns.isna().all():
            return 0.0
        return returns.mean() / returns.std() * sqrt(freq_per_year)
    
    def calculate_max_drawdown(self, equity: pd.Series) -> float:
        """计算最大回撤"""
        peaks = equity.cummax()
        dd = (equity - peaks) / peaks
        return dd.min()
    
    def ensemble_predict(self, observation, method: str = None) -> int:
        """
        集成预测 - 多个模型投票决定最终动作
        
        Args:
            observation: 环境观察值
            method: 投票方法 ('majority', 'weighted', 'soft')
        
        Returns:
            最终动作索引
        """
        if method is None:
            method = self.ensemble_config["VOTE_METHOD"]
        
        if not self.models:
            raise ValueError("没有可用的训练模型，请先训练集成模型")
        
        # 收集所有模型的预测
        all_predictions = []
        for model in self.models:
            action, _ = model.predict(observation, deterministic=True)
            all_predictions.append(int(action))
        
        # 根据投票方法决定最终动作
        if method == "majority":
            return self._majority_vote(all_predictions)
        elif method == "weighted":
            return self._weighted_vote(all_predictions)
        elif method == "soft":
            return self._soft_vote(observation, all_predictions)
        else:
            raise ValueError(f"不支持的投票方法: {method}")
    
    def _majority_vote(self, predictions: List[int]) -> int:
        """多数投票"""
        vote_counts = {}
        for pred in predictions:
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
        
        # 返回得票最多的动作
        return max(vote_counts.items(), key=lambda x: x[1])[0]
    
    def _weighted_vote(self, predictions: List[int]) -> int:
        """加权投票 - 根据模型性能分配权重"""
        if not self.model_performances:
            return self._majority_vote(predictions)
        
        # 使用夏普比率作为权重
        weights = []
        for perf in self.model_performances:
            # 将夏普比率转换为正权重
            weight = max(0.1, perf['sharpe_ratio'] + 1.0)  # 确保权重为正
            weights.append(weight)
        
        # 归一化权重
        weights = np.array(weights) / np.sum(weights)
        
        # 计算加权投票
        weighted_votes = {}
        for pred, weight in zip(predictions, weights):
            weighted_votes[pred] = weighted_votes.get(pred, 0.0) + weight
        
        return max(weighted_votes.items(), key=lambda x: x[1])[0]
    
    def _soft_vote(self, observation, predictions: List[int]) -> int:
        """软投票 - 基于动作概率分布"""
        action_probs = np.zeros(len(CONFIG["POSITION_NODES"]))
        
        for model in self.models:
            # 获取动作概率（需要访问模型内部）
            try:
                # 尝试获取动作概率分布
                action_prob = model.policy.get_distribution(observation).distribution.probs
                action_probs += action_prob.detach().numpy().flatten()
            except:
                # 如果无法获取概率，回退到硬投票
                action, _ = model.predict(observation, deterministic=True)
                action_probs[action] += 1
        
        # 返回概率最高的动作
        return np.argmax(action_probs)
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """获取集成模型摘要信息"""
        if not self.models:
            return {"error": "没有训练好的模型"}
        
        summary = {
            "ensemble_size": len(self.models),
            "vote_method": self.ensemble_config["VOTE_METHOD"],
            "individual_performances": self.model_performances,
            "average_sharpe": np.mean([p['sharpe_ratio'] for p in self.model_performances]),
            "average_return": np.mean([p['total_return'] for p in self.model_performances]),
            "consistency": self._calculate_consistency()
        }
        return summary
    
    def _calculate_consistency(self) -> float:
        """计算模型间的一致性"""
        if len(self.models) < 2:
            return 1.0
        
        # 使用测试数据评估模型间的一致性
        from semi_discrete_env import SemiDiscreteTradingEnv
        
        # 创建测试环境
        test_env = self._create_test_env()
        obs, _ = test_env.reset()
        
        agreements = 0
        total_steps = 0
        
        for step in range(100):  # 测试100步
            predictions = []
            for model in self.models:
                action, _ = model.predict(obs, deterministic=True)
                predictions.append(action)
            
            # 检查所有模型预测是否一致
            if len(set(predictions)) == 1:
                agreements += 1
            total_steps += 1
            
            obs, _, done, _, _ = test_env.step(predictions[0])
            if done:
                break
        
        return agreements / total_steps if total_steps > 0 else 0.0
    
    def _create_test_env(self):
        """创建测试环境（需要根据实际情况调整）"""
        # 这里需要您原有的环境创建代码
        # 返回一个测试环境实例
        pass

# ==================== 集成评估函数 ====================
def evaluate_ensemble(ensemble_manager: EnsemblePPOManager, test_env, test_df, name: str = "集成模型"):
    """评估集成模型性能"""
    print(f"\n{'='*60}")
    print(f"📊 评估{name}")
    print(f"{'='*60}")
    
    # 使用集成模型进行预测
    obs, _ = test_env.reset()
    equity_curve = []
    actions_taken = []
    
    step_count = 0
    while True:
        # 使用集成预测
        ensemble_action = ensemble_manager.ensemble_predict(obs)
        actions_taken.append(ensemble_action)
        
        obs, reward, done, _, _ = test_env.step(ensemble_action)
        equity_curve.append(test_env.asset)
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"   评估进度: {step_count} 步, 当前权益: {test_env.asset:.2f}")
        
        if done:
            break
    
    # 计算性能指标
    equity_series = pd.Series(equity_curve)
    returns = pd.Series([equity_curve[i] / equity_curve[i-1] - 1 for i in range(1, len(equity_curve))])
    
    # 输出结果
    print(f"\n📈 {name} 最终评估结果:")
    print(f"   最终权益: {equity_curve[-1]:.2f}")
    print(f"   总收益率: {(equity_curve[-1] / equity_curve[0] - 1) * 100:.2f}%")
    print(f"   夏普比率: {ensemble_manager.calculate_sharpe(returns):.3f}")
    print(f"   最大回撤: {ensemble_manager.calculate_max_drawdown(equity_series) * 100:.2f}%")
    
    # 显示集成摘要
    summary = ensemble_manager.get_ensemble_summary()
    print(f"\n🤝 集成模型摘要:")
    print(f"   模型数量: {summary['ensemble_size']}")
    print(f"   投票方法: {summary['vote_method']}")
    print(f"   平均夏普: {summary['average_sharpe']:.3f}")
    print(f"   平均收益: {summary['average_return']:.2f}%")
    print(f"   模型一致性: {summary['consistency']:.2f}")
    
    # 可视化结果
    visualize_ensemble_results(equity_curve, actions_taken, ensemble_manager, name)

def visualize_ensemble_results(equity_curve, actions_taken, ensemble_manager, name):
    """可视化集成模型结果"""
    plt.figure(figsize=(15, 10))
    
    # 1. 权益曲线
    plt.subplot(2, 2, 1)
    plt.plot(equity_curve)
    plt.title(f"{name} - 权益曲线")
    plt.ylabel("权益")
    plt.grid(True, alpha=0.3)
    
    # 2. 动作分布
    plt.subplot(2, 2, 2)
    action_counts = pd.Series(actions_taken).value_counts().sort_index()
    plt.bar([f"动作{i}" for i in action_counts.index], action_counts.values)
    plt.title("动作分布")
    plt.xticks(rotation=45)
    
    # 3. 单个模型性能比较
    plt.subplot(2, 2, 3)
    model_names = [p['model_name'] for p in ensemble_manager.model_performances]
    sharpe_ratios = [p['sharpe_ratio'] for p in ensemble_manager.model_performances]
    plt.bar(model_names, sharpe_ratios)
    plt.title("单个模型夏普比率")
    plt.xticks(rotation=45)
    
    # 4. 收益分布
    plt.subplot(2, 2, 4)
    returns = [p['total_return'] for p in ensemble_manager.model_performances]
    plt.bar(model_names, returns)
    plt.title("单个模型总收益(%)")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# ==================== 主函数 ====================
def main():
    """主函数 - 集成版本"""
    print("🎯 PPO模型投票集成系统")
    print("📝 不修改原代码，通过包装器实现集成学习")
    
    # 初始化集成管理器
    ensemble_manager = EnsemblePPOManager(ENSEMBLE_CONFIG, CONFIG)
    
    # 加载数据（使用原有函数）
    from data_loader import load_features, split_by_date  # 假设原有函数在这些文件中
    
    print("📊 加载数据...")
    df = load_features(CONFIG)
    train_df, test_df = split_by_date(df, CONFIG)
    
    print(f"训练集: {len(train_df)} 行")
    print(f"测试集: {len(test_df)} 行")
    
    # 创建环境（使用原有环境类）
    from semi_discrete_env import SemiDiscreteTradingEnv
    
    print("🔧 创建训练环境...")
    train_env = SemiDiscreteTradingEnv(train_df, CONFIG)
    train_env = DummyVecEnv([lambda: train_env])
    
    test_env = SemiDiscreteTradingEnv(test_df, CONFIG)
    
    # 训练集成模型
    ensemble_manager.train_ensemble(train_env, test_env, train_df, test_df)
    
    # 评估集成模型
    evaluate_ensemble(ensemble_manager, test_env, test_df, "PPO模型集成")
    
    # 保存集成模型
    save_ensemble_models(ensemble_manager)
    
    print("\n✅ 集成学习完成!")

def save_ensemble_models(ensemble_manager: EnsemblePPOManager):
    """保存集成模型"""
    os.makedirs("./ensemble_models", exist_ok=True)
    
    for i, model in enumerate(ensemble_manager.models):
        model_path = f"./ensemble_models/ppo_model_{i+1}.zip"
        model.save(model_path)
    
    # 保存集成信息
    ensemble_info = {
        'config': ENSEMBLE_CONFIG,
        'model_performances': ensemble_manager.model_performances,
        'base_config': CONFIG
    }
    
    import json
    with open("./ensemble_models/ensemble_info.json", "w") as f:
        json.dump(ensemble_info, f, indent=2)
    
    print(f"💾 集成模型已保存至 ./ensemble_models/")

if __name__ == "__main__":
    main()