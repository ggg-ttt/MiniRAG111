import time
import argparse
import pandas as pd
from openai import OpenAI

# ─────────────────────────────────────────────
# 配置区
# ─────────────────────────────────────────────
API_BASE_URL = "https://api.siliconflow.cn/v1"
API_KEY      = ""
EVAL_MODEL   = "deepseek-ai/DeepSeek-V3.2"

INPUT_CSV    = "tests/Qwen3-4B-Instruct-2507_vllm_debug/qwen_mini_output.csv"
OUTPUT_CSV   = "tests/Qwen3-4B-Instruct-2507_vllm_debug/qwen_mini_scored.csv"

# 评估的答案列名（可扩展为多列，如 ["lightRAG", "miniRAG"]）
ANSWER_COLS  = ["miniRAG"]

MAX_RETRIES  = 3
RETRY_DELAY  = 5   # seconds between retries

# ─────────────────────────────────────────────
# 提示词
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是一个专业的问答评估专家。你的任务是判断候选答案与标准答案在语义上是否一致。\n"
    "评分规则：\n"
    "  - 候选答案的核心结论与标准答案一致 → 输出 1\n"
    "  - 候选答案错误或与标准答案矛盾（但仍试图回答问题）→ 输出 0\n"
    "  - 候选答案与问题完全无关，或拒绝回答、内容空洞 → 输出 2\n"
    "只输出单个数字 0、1 或 2，不要输出任何其他内容。"
)

def build_user_prompt(question: str, gold: str, pred: str) -> str:
    return (
        f"问题：{question}\n\n"
        f"标准答案：{gold}\n\n"
        f"候选答案：{pred}\n\n"
        "请判断候选答案：\n"
        "  - 与标准答案一致 → 输出 1\n"
        "  - 回答了问题但答案错误 → 输出 0\n"
        "  - 与问题完全无关或拒绝回答 → 输出 2\n"
        "只输出 0、1 或 2。"
    )

# ─────────────────────────────────────────────
# API 调用（含重试）
# ─────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def call_eval_api(question: str, gold: str, pred: str) -> int | None:
    """调用大模型对候选答案打分，返回 0（错误）/ 1（正确）/ 2（无关），失败返回 None。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(question, gold, pred)},
                ],
                max_tokens=5,
                temperature=0,
                stream=False,
            )
            raw = response.choices[0].message.content.strip()
            if "2" in raw:
                return 2
            if "1" in raw:
                return 1
            if "0" in raw:
                return 0
            print(f"  [警告] 无法解析模型输出: {raw!r}，默认记为 0")
            return 0
        except Exception as e:
            print(f"  [错误] 第 {attempt} 次调用失败: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return None   # 全部重试均失败

# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main(input_csv: str, output_csv: str):
    print(f"读取文件：{input_csv}")
    df = pd.read_csv(input_csv)

    # 验证必要列
    for col in ["Question", "Gold Answer"] + ANSWER_COLS:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必要列：{col}")

    total_rows = len(df)
    print(f"共 {total_rows} 行，评估列：{ANSWER_COLS}\n")

    # 逐列评估
    for col in ANSWER_COLS:
        score_col = f"{col}_score"
        scores = []

        for row_idx in range(len(df)):
            question = str(df.iloc[row_idx]["Question"]).strip()
            gold     = str(df.iloc[row_idx]["Gold Answer"]).strip()
            pred     = str(df.iloc[row_idx][col]).strip()

            # 跳过空行
            if not question or not gold or not pred or pred.lower() == "nan":
                scores.append(None)
                print(f"  行 {row_idx+1:>3} [{col}] ⚠  数据缺失，跳过")
                continue

            score = call_eval_api(question, gold, pred)
            scores.append(score)

            mark = "✓" if score == 1 else ("✗" if score == 0 else ("⊘" if score == 2 else "?"))
            q_short = question[:70] + ("…" if len(question) > 70 else "")
            print(f"  行 {row_idx+1:>3} [{col}] {mark}  {q_short}")

            # 每100行或最后一行保存一次
            if (row_idx + 1) % 100 == 0 or row_idx == len(df) - 1:
                df[score_col] = scores + [None] * (len(df) - len(scores))  # 填充未处理行
                df.to_csv(output_csv, index=False, encoding="utf-8-sig")
                print(f"  💾 已保存前 {row_idx+1} 行结果到 {output_csv}")

        df[score_col] = scores

    # ─── 统计汇总 ───
    print("\n" + "=" * 55)
    print("评估结果汇总")
    print("=" * 55)

    summary_rows = []
    for col in ANSWER_COLS:
        score_col = f"{col}_score"
        valid      = [s for s in df[score_col] if s is not None]
        correct    = sum(1 for s in valid if s == 1)
        wrong      = sum(1 for s in valid if s == 0)
        irrelevant = sum(1 for s in valid if s == 2)
        total      = len(valid)
        acc        = correct    / total if total else 0.0
        wrong_rate = wrong      / total if total else 0.0
        irr_rate   = irrelevant / total if total else 0.0

        print(f"  {col}")
        print(f"    有效题数        ：{total}")
        print(f"    回答正确        ：{correct}  ({acc:.2%})")
        print(f"    回答错误        ：{wrong}  ({wrong_rate:.2%})")
        print(f"    与问题无关/拒答 ：{irrelevant}  ({irr_rate:.2%})\n")

        summary_rows.append({
            "answer_col":       col,
            "total":            total,
            "correct":          correct,
            "accuracy":         f"{acc:.4f}",
            "wrong":            wrong,
            "wrong_rate":       f"{wrong_rate:.4f}",
            "irrelevant":       irrelevant,
            "irrelevant_rate":  f"{irr_rate:.4f}",
        })

    # ─── 保存结果 ───
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"详细结果已保存：{output_csv}")

    summary_path = output_csv.replace(".csv", "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"汇总统计已保存：{summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用大模型评估 RAG 答案准确率")
    parser.add_argument("--input",  default=INPUT_CSV,  help="输入 CSV 文件路径")
    parser.add_argument("--output", default=OUTPUT_CSV, help="输出 CSV 文件路径")
    args = parser.parse_args()

    main(args.input, args.output)
