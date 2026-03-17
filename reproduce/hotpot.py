import re
import json
import torch
import networkx as nx
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==============================================================================
# 1. 标准 HotpotQA 指标计算（论文原版）
# ==============================================================================
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def f1_score(pred, truth):
    pred_tokens = normalize_answer(pred).split()
    truth_tokens = normalize_answer(truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(truth_tokens)
    return 2 * prec * rec / (prec + rec)

def exact_match(pred, truth):
    return int(normalize_answer(pred) == normalize_answer(truth))

def compute_hotpot_metrics(predictions, references):
    em_sum = f1_sum = 0
    for p, t in zip(predictions, references):
        em_sum += exact_match(p, t)
        f1_sum += f1_score(p, t)
    return {
        "EM": em_sum / len(predictions),
        "F1": f1_sum / len(predictions),
        "num_samples": len(predictions)
    }

# ==============================================================================
# 2. 轻量化 GraphRAG 核心（专为 7B/8B 小模型优化）
# ==============================================================================
class LightGraphRAG:
    def __init__(self, llm_pipeline):
        self.llm = llm_pipeline
        self.graph = nx.Graph()

    def extract_entities_relations(self, text):
        """小模型友好：抽取实体+关系（极简版，不耗token）"""
        prompt = f"""Extract entities and their relationships from text.
Output format: ENTITY1 | RELATION | ENTITY2
One per line. Only output triples.

Text: {text}
"""
        out = self.llm(prompt, max_new_tokens=128, temperature=0)[0]["generated_text"]
        triples = []
        for line in out.split("\n"):
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3:
                    triples.append(parts)
        return triples

    def build_graph(self, contexts):
        """从10段上下文构建知识图谱"""
        self.graph.clear()
        for ctx in contexts:
            title = ctx["title"]
            content = " ".join(ctx["sentences"])
            triples = self.extract_entities_relations(title + " " + content)
            for h, r, t in triples:
                self.graph.add_edge(h, t, relation=r)

    def retrieve_subgraph(self, question, depth=2):
        """多跳子图检索（GraphRAG 核心）"""
        seed_ents = self.extract_entities_relations(question)
        seed_nodes = [h for h, _, t in seed_ents] + [t for h, _, t in seed_ents]
        seed_nodes = [n for n in seed_nodes if n in self.graph]

        subgraph_nodes = set(seed_nodes)
        for node in seed_nodes:
            subgraph_nodes.update(nx.neighbors(self.graph, node))
            for neighbor in list(nx.neighbors(self.graph, node)):
                subgraph_nodes.update(nx.neighbors(self.graph, neighbor))

        subgraph = self.graph.subgraph(list(subgraph_nodes))
        return subgraph

    def subgraph_to_text(self, subgraph):
        """子图转文本（给小模型看的格式）"""
        lines = []
        for h, t, d in subgraph.edges(data=True):
            rel = d.get("relation", "related_to")
            lines.append(f"- {h} {rel} {t}")
        return "\n".join(lines)

# ==============================================================================
# 3. GraphRAG 提示词（专为多跳推理优化）
# ==============================================================================
def build_graphrag_prompt(question, subgraph_text):
    return f"""Use the knowledge graph to answer the question.
Only give the answer, no extra words.

Knowledge Graph:
{subgraph_text}

Question: {question}
Answer:"""

# ==============================================================================
# 4. 加载 HotpotQA 验证集（distractor 标准）
# ==============================================================================
def load_hotpot_val(n_samples=200):
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    return ds.select(range(n_samples))

# ==============================================================================
# 5. 7B/8B 模型加载（通用）
# ==============================================================================
def build_model(model_path="meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False
    )

# ==============================================================================
# 6. GraphRAG + HotpotQA 主评测流程
# ==============================================================================
def run_graphrag_hotpot_eval(model_name, n_samples=200):
    llm = build_model(model_name)
    graphrag = LightGraphRAG(llm)
    dataset = load_hotpot_val(n_samples)

    preds, truths = [], []

    for sample in tqdm(dataset, desc="Evaluating GraphRAG"):
        question = sample["question"]
        contexts = sample["context"]
        answer = sample["answer"]

        # === GraphRAG 核心流程 ===
        graphrag.build_graph(contexts)       # 建图
        subgraph = graphrag.retrieve_subgraph(question)  # 多跳检索
        subgraph_text = graphrag.subgraph_to_text(subgraph)

        # 生成答案
        prompt = build_graphrag_prompt(question, subgraph_text)
        output = llm(prompt)[0]["generated_text"].split("Answer:")[-1].strip()

        preds.append(output)
        truths.append(answer)

    # 计算指标
    metrics = compute_hotpot_metrics(preds, truths)
    print("\n" + "="*50)
    print(" HotpotQA GraphRAG 评测结果 (7B/8B)")
    print("="*50)
    print(f"测试样本数: {metrics['num_samples']}")
    print(f"EM (精确匹配): {metrics['EM']:.4f}")
    print(f"F1 (语义重叠): {metrics['F1']:.4f}")

    # 保存结果
    with open("graphrag_hotpot_result.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "predictions": preds, "references": truths}, f, indent=2)

if __name__ == "__main__":
    run_graphrag_hotpot_eval(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        n_samples=200  # 先跑200条验证
    )