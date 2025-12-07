#!/usr/bin/env python3
"""
优化策略实施工具
基于 Bad Case 分析结果,提供针对性的优化方案
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


class OptimizationStrategy:
    """优化策略基类"""
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def apply(self, config: Dict) -> Dict:
        """应用优化策略,修改训练配置"""
        raise NotImplementedError


class LongTailBoostStrategy(OptimizationStrategy):
    """长尾物品增强策略"""
    def __init__(self):
        super().__init__(
            name="LongTailBoost",
            description="增加长尾物品的训练权重,提升覆盖率"
        )

    def apply(self, config: Dict) -> Dict:
        print(f"\n应用策略: {self.name}")
        print(f"描述: {self.description}")

        # 修改数据采样策略
        config['data_sampling'] = 'frequency_balanced'  # 频率平衡采样
        config['long_tail_weight'] = 2.0  # 长尾物品权重翻倍

        # 在 RL 阶段增加多样性奖励
        config['diversity_reward_weight'] = 0.1
        config['coverage_penalty'] = True

        print("  - 启用频率平衡采样")
        print("  - 长尾物品权重: 2.0")
        print("  - 多样性奖励权重: 0.1")

        return config


class ColdStartEnhancementStrategy(OptimizationStrategy):
    """冷启动物品增强策略"""
    def __init__(self):
        super().__init__(
            name="ColdStartEnhancement",
            description="增强物品侧特征,改善冷启动物品推荐"
        )

    def apply(self, config: Dict) -> Dict:
        print(f"\n应用策略: {self.name}")
        print(f"描述: {self.description}")

        # 启用物品特征对齐任务
        config['enable_item_feature_alignment'] = True
        config['item_feature_weight'] = 0.3

        # 增加物品标题/描述的使用频率
        config['use_item_title_in_prompt'] = True
        config['title_embedding_weight'] = 0.2

        # 在 SFT 阶段增加物品内容理解任务
        config['additional_tasks'] = config.get('additional_tasks', [])
        config['additional_tasks'].append('item_content_understanding')

        print("  - 启用物品特征对齐任务 (权重: 0.3)")
        print("  - 在提示词中包含物品标题")
        print("  - 添加物品内容理解任务")

        return config


class ShortHistoryAdaptationStrategy(OptimizationStrategy):
    """短历史用户适应策略"""
    def __init__(self):
        super().__init__(
            name="ShortHistoryAdaptation",
            description="优化短历史用户的推荐效果"
        )

    def apply(self, config: Dict) -> Dict:
        print(f"\n应用策略: {self.name}")
        print(f"描述: {self.description}")

        # 引入用户画像信息
        config['use_user_profile'] = True
        config['profile_features'] = ['category_preference', 'price_range']

        # 调整历史序列处理方式
        config['min_history_length'] = 2  # 降低最小历史长度要求
        config['use_global_popularity'] = True  # 对短历史用户引入全局流行度

        # 启用协同过滤增强
        config['enable_collaborative_filtering'] = True
        config['cf_weight'] = 0.15

        print("  - 启用用户画像特征")
        print("  - 降低最小历史长度要求: 2")
        print("  - 引入全局流行度信号")
        print("  - 启用协同过滤增强 (权重: 0.15)")

        return config


class SIDQualityImprovementStrategy(OptimizationStrategy):
    """SID 质量改进策略"""
    def __init__(self):
        super().__init__(
            name="SIDQualityImprovement",
            description="改进 Semantic ID 的质量,减少碰撞"
        )

    def apply(self, config: Dict) -> Dict:
        print(f"\n应用策略: {self.name}")
        print(f"描述: {self.description}")

        # 调整 SID 构建参数
        config['sid_method'] = 'rqkmeans_plus'  # 使用 RQ-Kmeans+ 方法
        config['num_quantization_layers'] = 4  # 增加量化层数
        config['codebook_size'] = 512  # 增加 codebook 大小

        # 启用碰撞检测和处理
        config['detect_sid_collision'] = True
        config['collision_resolution'] = 'add_discriminator_layer'

        print("  - 使用 RQ-Kmeans+ 方法")
        print("  - 量化层数: 4")
        print("  - Codebook 大小: 512")
        print("  - 启用碰撞检测和解决")

        return config


class RecencyBiasFixStrategy(OptimizationStrategy):
    """最近偏好修正策略"""
    def __init__(self):
        super().__init__(
            name="RecencyBiasFix",
            description="增强模型对用户最近偏好的捕捉"
        )

    def apply(self, config: Dict) -> Dict:
        print(f"\n应用策略: {self.name}")
        print(f"描述: {self.description}")

        # 调整序列编码方式
        config['use_position_encoding'] = True
        config['recency_weight_decay'] = 0.9  # 时间衰减因子

        # 增加对最近交互的注意力
        config['recent_items_attention_boost'] = 1.5
        config['num_recent_items_focus'] = 3  # 关注最近 3 个物品

        # 在提示词中强调最近偏好
        config['prompt_template'] = 'recency_aware'

        print("  - 启用位置编码")
        print("  - 时间衰减因子: 0.9")
        print("  - 增强对最近 3 个物品的注意力 (1.5x)")
        print("  - 使用感知最近偏好的提示模板")

        return config


class HyperparameterTuningStrategy(OptimizationStrategy):
    """超参数调优策略"""
    def __init__(self):
        super().__init__(
            name="HyperparameterTuning",
            description="基于 Bad Case 分析调整训练超参数"
        )

    def apply(self, config: Dict) -> Dict:
        print(f"\n应用策略: {self.name}")
        print(f"描述: {self.description}")

        # SFT 阶段调整
        config['sft_learning_rate'] = 3e-4  # 降低学习率,更稳定
        config['sft_num_epochs'] = 5  # 增加训练轮数
        config['warmup_ratio'] = 0.1  # 添加预热

        # RL 阶段调整
        config['rl_learning_rate'] = 5e-6  # 更小的 RL 学习率
        config['rl_beta'] = 5e-4  # 调整 KL 惩罚系数
        config['num_generations'] = 32  # 增加生成候选数

        # 正则化
        config['weight_decay'] = 0.01
        config['dropout'] = 0.1

        print("  - SFT 学习率: 3e-4")
        print("  - SFT 训练轮数: 5")
        print("  - RL 学习率: 5e-6")
        print("  - RL 候选生成数: 32")
        print("  - 添加正则化 (weight_decay: 0.01, dropout: 0.1)")

        return config


class OptimizationManager:
    """优化策略管理器"""

    STRATEGIES = {
        'long_tail': LongTailBoostStrategy,
        'cold_start': ColdStartEnhancementStrategy,
        'short_history': ShortHistoryAdaptationStrategy,
        'sid_quality': SIDQualityImprovementStrategy,
        'recency_bias': RecencyBiasFixStrategy,
        'hyperparameter': HyperparameterTuningStrategy,
    }

    def __init__(self, badcase_report_path: str):
        with open(badcase_report_path, 'r', encoding='utf-8') as f:
            self.report = json.load(f)

    def recommend_strategies(self) -> List[str]:
        """基于 Bad Case 分析推荐优化策略"""
        failure_patterns = self.report.get('failure_patterns', {})
        sid_quality = self.report.get('sid_quality', {})
        total_bad_cases = self.report.get('summary', {}).get('bad_cases_count', 0)

        recommended = []

        # 根据失败模式推荐策略
        if failure_patterns.get('long_tail', 0) > total_bad_cases * 0.3:
            recommended.append('long_tail')

        if failure_patterns.get('cold_start', 0) > total_bad_cases * 0.2:
            recommended.append('cold_start')

        if failure_patterns.get('short_history', 0) > total_bad_cases * 0.25:
            recommended.append('short_history')

        if failure_patterns.get('recency_bias', 0) > total_bad_cases * 0.2:
            recommended.append('recency_bias')

        # 根据 SID 质量推荐策略
        if sid_quality.get('collision', 0) > 0:
            recommended.append('sid_quality')

        # 总是推荐超参数调优
        recommended.append('hyperparameter')

        return recommended

    def apply_strategies(self, strategy_names: List[str], base_config: Dict) -> Dict:
        """应用多个优化策略"""
        config = base_config.copy()

        print("\n" + "=" * 60)
        print("应用优化策略")
        print("=" * 60)

        for name in strategy_names:
            if name not in self.STRATEGIES:
                print(f"警告: 未知策略 '{name}',跳过")
                continue

            strategy = self.STRATEGIES[name]()
            config = strategy.apply(config)

        print("\n" + "=" * 60)
        print("所有策略应用完成")
        print("=" * 60)

        return config

    def generate_optimized_config(self, output_path: str, auto_recommend: bool = True):
        """生成优化后的配置文件"""
        # 加载基础配置
        base_config = {
            'model_size': '0.5B',
            'category': 'Industrial_and_Scientific',
            'seed': 42,
        }

        # 推荐或手动选择策略
        if auto_recommend:
            strategy_names = self.recommend_strategies()
            print(f"\n自动推荐的优化策略: {', '.join(strategy_names)}")
        else:
            print("\n可用的优化策略:")
            for i, (key, strategy_class) in enumerate(self.STRATEGIES.items(), 1):
                strategy = strategy_class()
                print(f"  {i}. {key}: {strategy.description}")

            selected = input("\n请输入要应用的策略编号 (逗号分隔,或输入 'all'): ").strip()

            if selected.lower() == 'all':
                strategy_names = list(self.STRATEGIES.keys())
            else:
                indices = [int(x.strip()) - 1 for x in selected.split(',')]
                strategy_names = [list(self.STRATEGIES.keys())[i] for i in indices]

        # 应用策略
        optimized_config = self.apply_strategies(strategy_names, base_config)

        # 保存配置
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_config, f, indent=2, ensure_ascii=False)

        print(f"\n优化配置已保存到: {output_path}")

        return optimized_config


def main():
    parser = argparse.ArgumentParser(description='优化策略生成工具')
    parser.add_argument('--badcase_report', type=str, required=True,
                        help='Bad Case 分析报告路径')
    parser.add_argument('--output_config', type=str, default='./optimized_config.json',
                        help='输出配置文件路径')
    parser.add_argument('--auto_recommend', action='store_true',
                        help='自动推荐优化策略')
    parser.add_argument('--strategies', type=str, nargs='+',
                        help='手动指定策略 (可选: long_tail, cold_start, short_history, sid_quality, recency_bias, hyperparameter)')

    args = parser.parse_args()

    manager = OptimizationManager(args.badcase_report)

    if args.strategies:
        # 使用指定的策略
        base_config = {'model_size': '0.5B', 'category': 'Industrial_and_Scientific'}
        optimized_config = manager.apply_strategies(args.strategies, base_config)

        with open(args.output_config, 'w', encoding='utf-8') as f:
            json.dump(optimized_config, f, indent=2, ensure_ascii=False)

        print(f"\n优化配置已保存到: {args.output_config}")
    else:
        # 自动推荐或交互式选择
        manager.generate_optimized_config(args.output_config, args.auto_recommend)


if __name__ == '__main__':
    main()
