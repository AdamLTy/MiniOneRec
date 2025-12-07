#!/usr/bin/env python3
"""
可视化分析工具
生成 Bad Case 分析的可视化图表
"""

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')


class AnalysisVisualizer:
    def __init__(self, report_path, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'r', encoding='utf-8') as f:
            self.report = json.load(f)

    def plot_failure_patterns(self):
        """绘制失败模式分布"""
        patterns = self.report.get('failure_patterns', {})

        if not patterns:
            print("无失败模式数据")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        pattern_names = list(patterns.keys())
        pattern_counts = list(patterns.values())

        # 翻译模式名称
        name_mapping = {
            'cold_start': 'Cold Start',
            'long_tail': 'Long Tail',
            'short_history': 'Short History',
            'category_mismatch': 'Category Mismatch',
            'recency_bias': 'Recency Bias'
        }
        pattern_names = [name_mapping.get(name, name) for name in pattern_names]

        bars = ax.barh(pattern_names, pattern_counts, color=sns.color_palette('Set2'))

        # 添加数值标签
        for i, (name, count) in enumerate(zip(pattern_names, pattern_counts)):
            ax.text(count + max(pattern_counts) * 0.01, i, str(count),
                    va='center', fontsize=10)

        ax.set_xlabel('Count', fontsize=12)
        ax.set_title('Failure Pattern Distribution', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(pattern_counts) * 1.15)

        plt.tight_layout()
        output_path = self.output_dir / 'failure_patterns.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"失败模式分布图已保存: {output_path}")

    def plot_failure_rate_pie(self):
        """绘制失败率饼图"""
        summary = self.report.get('summary', {})
        total = summary.get('total_samples', 0)
        bad_cases = summary.get('bad_cases_count', 0)
        success_cases = total - bad_cases

        if total == 0:
            print("无统计数据")
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        labels = ['Success', 'Failure']
        sizes = [success_cases, bad_cases]
        colors = ['#90EE90', '#FFB6C1']
        explode = (0, 0.1)

        wedges, texts, autotexts = ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct='%1.2f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )

        # 加粗百分比
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)

        ax.set_title(f'Top-K Recommendation Success Rate\n(Total: {total} samples)',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / 'failure_rate.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"失败率饼图已保存: {output_path}")

    def plot_sid_distribution(self):
        """绘制 SID 质量分析"""
        sid_quality = self.report.get('sid_quality', {})

        if not sid_quality:
            print("无 SID 质量数据")
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        metrics = list(sid_quality.keys())
        values = list(sid_quality.values())

        name_mapping = {
            'collision': 'SID Collision',
            'semantic_gap': 'Semantic Gap',
            'code_imbalance': 'Code Imbalance'
        }
        metrics = [name_mapping.get(m, m) for m in metrics]

        bars = ax.bar(metrics, values, color=sns.color_palette('pastel'))

        # 添加数值标签
        for i, (metric, value) in enumerate(zip(metrics, values)):
            ax.text(i, value + max(values) * 0.02, str(value),
                    ha='center', fontsize=11, fontweight='bold')

        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('SID Quality Issues', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.2 if values else 1)

        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        output_path = self.output_dir / 'sid_quality.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"SID 质量图已保存: {output_path}")

    def plot_rank_distribution(self):
        """绘制真实物品排名分布"""
        bad_cases = self.report.get('bad_cases_sample', [])

        if not bad_cases:
            print("无 Bad Case 样本数据")
            return

        ranks = [case.get('ground_truth_rank', -1) for case in bad_cases]
        ranks = [r for r in ranks if r > 0]  # 过滤掉不在列表中的

        if not ranks:
            print("无有效排名数据")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        bins = [1, 10, 20, 30, 50, 100, 200, max(ranks)+1]
        hist, bin_edges = np.histogram(ranks, bins=bins)

        bin_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1]-1)}' for i in range(len(bin_edges)-1)]

        bars = ax.bar(range(len(hist)), hist, color=sns.color_palette('coolwarm', len(hist)))

        # 添加数值标签
        for i, count in enumerate(hist):
            if count > 0:
                ax.text(i, count + max(hist) * 0.01, str(count),
                        ha='center', fontsize=10)

        ax.set_xlabel('Ground Truth Rank Range', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Ground Truth Rankings in Bad Cases', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')

        plt.tight_layout()
        output_path = self.output_dir / 'rank_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"排名分布图已保存: {output_path}")

    def generate_summary_report(self):
        """生成文本摘要报告"""
        summary = self.report.get('summary', {})
        patterns = self.report.get('failure_patterns', {})
        sid_quality = self.report.get('sid_quality', {})

        report_lines = [
            "=" * 60,
            "Bad Case Analysis Summary Report",
            "=" * 60,
            "",
            "## Overall Statistics",
            f"Total Samples: {summary.get('total_samples', 0)}",
            f"Bad Cases: {summary.get('bad_cases_count', 0)}",
            f"Failure Rate: {summary.get('failure_rate', 0) * 100:.2f}%",
            "",
            "## Failure Patterns",
        ]

        for pattern, count in patterns.items():
            report_lines.append(f"  - {pattern}: {count}")

        report_lines.extend([
            "",
            "## SID Quality Issues",
        ])

        for issue, count in sid_quality.items():
            report_lines.append(f"  - {issue}: {count}")

        report_lines.extend([
            "",
            "=" * 60,
            "For detailed analysis, see the JSON report and visualizations.",
            "=" * 60,
        ])

        report_text = '\n'.join(report_lines)

        output_path = self.output_dir / 'summary_report.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n摘要报告已保存: {output_path}")
        print("\n" + report_text)

    def generate_all_plots(self):
        """生成所有可视化图表"""
        print("生成可视化图表...")
        self.plot_failure_rate_pie()
        self.plot_failure_patterns()
        self.plot_sid_distribution()
        self.plot_rank_distribution()
        self.generate_summary_report()
        print(f"\n所有可视化图表已保存到: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='可视化分析工具')
    parser.add_argument('--report_path', type=str, required=True,
                        help='分析报告 JSON 文件路径')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='输出目录')

    args = parser.parse_args()

    visualizer = AnalysisVisualizer(args.report_path, args.output_dir)
    visualizer.generate_all_plots()


if __name__ == '__main__':
    main()
