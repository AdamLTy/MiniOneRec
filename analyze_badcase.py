#!/usr/bin/env python3
"""
Bad Case 分析工具
分析推荐模型失败的样本,找出优化方向
"""

import json
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class BadCaseAnalyzer:
    def __init__(self, eval_results_path, test_data_path, item_meta_path, sid_index_path):
        """
        Args:
            eval_results_path: 评估结果文件路径 (包含预测和真实标签)
            test_data_path: 测试数据路径
            item_meta_path: 物品元数据路径
            sid_index_path: SID 索引路径
        """
        self.eval_results = self.load_eval_results(eval_results_path)
        self.test_data = pd.read_csv(test_data_path)

        with open(item_meta_path, 'r', encoding='utf-8') as f:
            self.item_meta = json.load(f)

        with open(sid_index_path, 'r', encoding='utf-8') as f:
            self.sid_index = json.load(f)

        self.bad_cases = []
        self.analysis_results = {}

    def load_eval_results(self, path):
        """加载评估结果"""
        results = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results

    def identify_bad_cases(self, top_k=10):
        """识别 Bad Cases: 真实物品不在 Top-K 预测中"""
        print(f"识别 Top-{top_k} 失败的样本...")

        for idx, result in enumerate(self.eval_results):
            predictions = result.get('predictions', [])[:top_k]
            ground_truth = result.get('ground_truth', '')

            if ground_truth not in predictions:
                bad_case = {
                    'index': idx,
                    'user_id': result.get('user_id', 'unknown'),
                    'history': result.get('history', []),
                    'ground_truth': ground_truth,
                    'predictions': predictions,
                    'ground_truth_rank': self._get_rank(ground_truth, result.get('all_predictions', predictions)),
                    'confidence_scores': result.get('confidence_scores', [])[:top_k]
                }
                self.bad_cases.append(bad_case)

        print(f"发现 {len(self.bad_cases)} 个 Bad Cases (占比: {len(self.bad_cases)/len(self.eval_results)*100:.2f}%)")
        return self.bad_cases

    def _get_rank(self, item_id, predictions):
        """获取物品在预测列表中的排名"""
        try:
            return predictions.index(item_id) + 1
        except ValueError:
            return -1  # 不在预测列表中

    def analyze_failure_patterns(self):
        """分析失败模式"""
        print("\n===== 失败模式分析 =====")

        patterns = {
            'cold_start': 0,  # 冷启动物品
            'long_tail': 0,   # 长尾物品
            'short_history': 0,  # 短历史用户
            'category_mismatch': 0,  # 类别不匹配
            'recency_bias': 0,  # 忽略最近交互
        }

        for case in self.bad_cases:
            # 冷启动检测
            gt_item = case['ground_truth']
            if self._is_cold_start_item(gt_item):
                patterns['cold_start'] += 1

            # 长尾检测
            if self._is_long_tail_item(gt_item):
                patterns['long_tail'] += 1

            # 短历史检测
            if len(case['history']) < 5:
                patterns['short_history'] += 1

            # 类别不匹配检测
            if self._has_category_mismatch(case):
                patterns['category_mismatch'] += 1

            # 最近偏好忽略
            if self._ignores_recent_preference(case):
                patterns['recency_bias'] += 1

        self.analysis_results['failure_patterns'] = patterns

        # 打印结果
        total = len(self.bad_cases)
        for pattern, count in patterns.items():
            print(f"{pattern}: {count} ({count/total*100:.2f}%)")

        return patterns

    def _is_cold_start_item(self, item_id):
        """判断是否为冷启动物品 (交互次数少)"""
        # 简化实现: 检查物品在训练集中的出现频率
        # 实际应该统计物品的历史交互次数
        return False  # TODO: 实现基于统计的冷启动检测

    def _is_long_tail_item(self, item_id):
        """判断是否为长尾物品"""
        # TODO: 基于物品流行度分布判断
        return False

    def _has_category_mismatch(self, case):
        """判断预测物品类别是否与历史不匹配"""
        # TODO: 需要物品类别信息
        return False

    def _ignores_recent_preference(self, case):
        """判断是否忽略了用户最近的偏好"""
        if len(case['history']) < 3:
            return False

        # 检查最近3次交互的物品是否与预测相似
        recent_items = case['history'][-3:]
        # TODO: 计算相似度
        return False

    def analyze_sid_quality(self):
        """分析 SID 质量问题"""
        print("\n===== SID 质量分析 =====")

        sid_issues = {
            'collision': 0,  # SID 碰撞
            'semantic_gap': 0,  # 语义差距大
            'code_imbalance': 0,  # Code 使用不均衡
        }

        # 统计 SID 分布
        sid_distribution = Counter()
        for item_id, sid in self.sid_index.items():
            sid_str = '-'.join(map(str, sid))
            sid_distribution[sid_str] += 1

        # 检测碰撞
        collisions = {sid: count for sid, count in sid_distribution.items() if count > 1}
        sid_issues['collision'] = len(collisions)

        print(f"SID 碰撞数: {sid_issues['collision']}")
        print(f"SID 唯一性: {len(sid_distribution) / len(self.sid_index) * 100:.2f}%")

        # Code 分布均衡性
        code_usage = self._analyze_code_distribution()
        print(f"Code 分布熵: {self._calculate_entropy(code_usage):.4f}")

        self.analysis_results['sid_quality'] = sid_issues
        return sid_issues

    def _analyze_code_distribution(self):
        """分析每层 Code 的使用分布"""
        num_layers = len(next(iter(self.sid_index.values())))
        code_usage = [Counter() for _ in range(num_layers)]

        for sid in self.sid_index.values():
            for layer, code in enumerate(sid):
                code_usage[layer][code] += 1

        return code_usage

    def _calculate_entropy(self, code_usage):
        """计算分布熵"""
        entropy = 0
        for layer_usage in code_usage:
            total = sum(layer_usage.values())
            probs = [count / total for count in layer_usage.values()]
            layer_entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            entropy += layer_entropy
        return entropy / len(code_usage)

    def generate_case_studies(self, num_cases=10):
        """生成典型 Bad Case 样例"""
        print(f"\n===== 典型 Bad Cases =====")

        # 随机选择一些 bad cases
        import random
        selected_cases = random.sample(self.bad_cases, min(num_cases, len(self.bad_cases)))

        for i, case in enumerate(selected_cases, 1):
            print(f"\n--- Case {i} ---")
            print(f"User ID: {case['user_id']}")
            print(f"History Length: {len(case['history'])}")
            print(f"Ground Truth: {case['ground_truth']}")
            print(f"  Title: {self.item_meta.get(case['ground_truth'], {}).get('title', 'Unknown')}")
            print(f"GT Rank: {case['ground_truth_rank']}")
            print(f"\nTop-5 Predictions:")
            for j, pred in enumerate(case['predictions'][:5], 1):
                title = self.item_meta.get(pred, {}).get('title', 'Unknown')
                confidence = case['confidence_scores'][j-1] if j <= len(case['confidence_scores']) else 'N/A'
                print(f"  {j}. {pred} (conf: {confidence})")
                print(f"     {title}")

    def suggest_improvements(self):
        """基于分析结果提供优化建议"""
        print("\n===== 优化建议 =====")

        suggestions = []

        patterns = self.analysis_results.get('failure_patterns', {})

        # 基于失败模式给出建议
        if patterns.get('cold_start', 0) > len(self.bad_cases) * 0.2:
            suggestions.append({
                'issue': '冷启动物品推荐失败率高',
                'suggestion': [
                    '1. 增加物品侧特征 (标题、描述、类别) 的训练权重',
                    '2. 使用物品内容相似度作为辅助奖励',
                    '3. 在 SFT 阶段增加物品特征对齐任务'
                ]
            })

        if patterns.get('long_tail', 0) > len(self.bad_cases) * 0.3:
            suggestions.append({
                'issue': '长尾物品覆盖不足',
                'suggestion': [
                    '1. 调整训练数据采样策略,增加长尾物品曝光',
                    '2. 在 RL 阶段使用多样性奖励 (如覆盖率、新颖性)',
                    '3. 使用对比学习增强长尾物品的表示'
                ]
            })

        if patterns.get('short_history', 0) > len(self.bad_cases) * 0.25:
            suggestions.append({
                'issue': '短历史用户推荐效果差',
                'suggestion': [
                    '1. 引入用户画像信息 (如偏好类别)',
                    '2. 使用元学习方法快速适应新用户',
                    '3. 利用相似用户的历史进行协同过滤增强'
                ]
            })

        sid_quality = self.analysis_results.get('sid_quality', {})
        if sid_quality.get('collision', 0) > 0:
            suggestions.append({
                'issue': f'SID 碰撞问题 ({sid_quality["collision"]} 个)',
                'suggestion': [
                    '1. 增加量化层数或 codebook 大小',
                    '2. 使用 RQ-Kmeans+ 方法替代当前方法',
                    '3. 对碰撞物品添加额外区分层'
                ]
            })

        # 打印建议
        for i, sug in enumerate(suggestions, 1):
            print(f"\n{i}. {sug['issue']}")
            for item in sug['suggestion']:
                print(f"   {item}")

        return suggestions

    def export_report(self, output_path):
        """导出分析报告"""
        report = {
            'summary': {
                'total_samples': len(self.eval_results),
                'bad_cases_count': len(self.bad_cases),
                'failure_rate': len(self.bad_cases) / len(self.eval_results)
            },
            'failure_patterns': self.analysis_results.get('failure_patterns', {}),
            'sid_quality': self.analysis_results.get('sid_quality', {}),
            'bad_cases_sample': self.bad_cases[:20]  # 保存前20个样例
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n分析报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Bad Case 分析工具')
    parser.add_argument('--eval_results', type=str, required=True,
                        help='评估结果文件路径')
    parser.add_argument('--test_data', type=str, required=True,
                        help='测试数据路径')
    parser.add_argument('--item_meta', type=str, required=True,
                        help='物品元数据路径')
    parser.add_argument('--sid_index', type=str, required=True,
                        help='SID 索引路径')
    parser.add_argument('--output_dir', type=str, default='./analysis_output',
                        help='输出目录')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top-K 阈值')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化分析器
    analyzer = BadCaseAnalyzer(
        args.eval_results,
        args.test_data,
        args.item_meta,
        args.sid_index
    )

    # 执行分析
    analyzer.identify_bad_cases(top_k=args.top_k)
    analyzer.analyze_failure_patterns()
    analyzer.analyze_sid_quality()
    analyzer.generate_case_studies(num_cases=10)
    analyzer.suggest_improvements()

    # 导出报告
    report_path = output_dir / 'badcase_report.json'
    analyzer.export_report(report_path)


if __name__ == '__main__':
    main()
