"""
合并三个RAG结果文件的脚本
将naiveRAG、lightRAG和miniRAG的评分结果合并为一个统一的CSV文件
"""

import pandas as pd
from pathlib import Path


def merge_rag_results(
    naive_path: str,
    light_path: str,
    mini_path: str,
    output_path: str = "merged_rag_results.csv"
):
    """
    合并三个RAG结果文件

    Args:
        naive_path: naiveRAG结果文件路径
        light_path: lightRAG结果文件路径
        mini_path: miniRAG结果文件路径
        output_path: 输出文件路径
    """
    # 读取三个CSV文件
    print(f"读取naiveRAG结果: {naive_path}")
    df_naive = pd.read_csv(naive_path)

    print(f"读取lightRAG结果: {light_path}")
    df_light = pd.read_csv(light_path)

    print(f"读取miniRAG结果: {mini_path}")
    df_mini = pd.read_csv(mini_path)

    # 检查行数是否一致
    n_rows = len(df_naive)
    if len(df_light) != n_rows or len(df_mini) != n_rows:
        print(f"⚠️  警告: 文件行数不一致!")
        print(f"  naiveRAG: {n_rows} 行")
        print(f"  lightRAG: {len(df_light)} 行")
        print(f"  miniRAG: {len(df_mini)} 行")

    # 创建合并后的DataFrame
    merged_df = pd.DataFrame()

    # 基础列（从naiveRAG文件中获取）
    merged_df['Question'] = df_naive['Question']
    merged_df['Gold Answer'] = df_naive['Gold Answer']

    # 添加各个RAG的回答列
    merged_df['naiveRAG'] = df_naive['naiveRAG']
    merged_df['lightRAG'] = df_light['lightRAG']
    merged_df['miniRAG'] = df_mini['miniRAG']

    # 添加各个RAG的评分列
    merged_df['naiveRAG_score'] = df_naive['naiveRAG_score']
    merged_df['lightRAG_score'] = df_light['lightRAG_score']
    merged_df['miniRAG_score'] = df_mini['miniRAG_score']

    # 保存合并结果
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 合并完成! 结果已保存到: {output_path}")
    print(f"   共 {len(merged_df)} 行数据")
    print(f"\n列名: {list(merged_df.columns)}")

    # 显示统计信息
    print(f"\n📊 评分统计:")
    print(f"   naiveRAG 平均分: {merged_df['naiveRAG_score'].mean():.3f}")
    print(f"   lightRAG 平均分: {merged_df['lightRAG_score'].mean():.3f}")
    print(f"   miniRAG 平均分: {merged_df['miniRAG_score'].mean():.3f}")

    return merged_df


if __name__ == "__main__":
    # 设置文件路径
    base_dir = Path(__file__).parent / "tests" / "Qwen3-4B-Instruct-2507_vllm_debug"

    naive_file = base_dir / "qwen_naive_scored.csv"
    light_file = base_dir / "qwen_light_scored.csv"
    mini_file = base_dir / "qwen_mini_scored.csv"
    output_file = base_dir / "qwen_all_rag_merged.csv"

    # 执行合并
    merge_rag_results(
        naive_path=str(naive_file),
        light_path=str(light_file),
        mini_path=str(mini_file),
        output_path=str(output_file)
    )
