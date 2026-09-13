#!/usr/bin/env python
"""
文档问答评估示例

1. 使用 parameter.py 中的配置构建知识图谱
2. 逐个回答 13_qa.jsonl 中的问题
3. 将结果整理到 Markdown 文件
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.builder import MMKGBuilder
from src.retrieval.query import GraphRAGQuery
from src.parameter import QueryParam, OUTPUT_DIR, WORKING_DIR, INPUT_PDF_PATH


def build_knowledge_graph():
    """构建知识图谱"""
    pdf_path = INPUT_PDF_PATH
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF 文件不存在: {pdf_path}")
        return False
    
    # 检查是否已构建
    output_graph = os.path.join(OUTPUT_DIR, "example_mmkg.graphml")
    if os.path.exists(output_graph):
        print(f"✅ 检测到已构建的知识图谱: {output_graph}")
        return True
    
    print(f"🔨 开始构建知识图谱: {pdf_path}")
    try:
        builder = MMKGBuilder()
        builder.index(pdf_path)
        print("✅ 知识图谱构建完成")
        return True
    except Exception as e:
        print(f"❌ 构建失败: {e}")
        return False


async def answer_questions(qa_file: str) -> list:
    """回答问题并返回结果"""
    results = []
    
    # 加载问题
    questions = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    
    print(f"📝 加载了 {len(questions)} 个问题")
    
    # 初始化 RAG
    rag = GraphRAGQuery()
    param = QueryParam()
    
    for i, qa in enumerate(questions, 1):
        question = qa.get("question", "")
        standard_answer = qa.get("answer", "")
        evidence = qa.get("evidence", "")
        qa_type = qa.get("type", "unknown")
        
        print(f"\n[{i}/{len(questions)}] 正在回答: {question[:50]}...")
        
        try:
            response = await rag.query(question, param)
        except Exception as e:
            print(f"  ⚠️ 查询失败: {e}")
            response = f"查询失败: {e}"
        
        results.append({
            "id": i,
            "type": qa_type,
            "question": question,
            "standard_answer": standard_answer,
            "evidence": evidence,
            "model_answer": response
        })
        
        print(f"  ✅ 已完成")
    
    return results


def generate_report(results: list, output_path: str):
    """生成 Markdown 报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""# 文档问答评估报告

**生成时间**: {timestamp}  
**问题数量**: {len(results)}  
**知识图谱**: `{OUTPUT_DIR}`

---

"""
    
    for r in results:
        content += f"""## 问题 {r['id']}

**类型**: `{r['type']}`

### 问题
{r['question']}

### 标准答案
{r['standard_answer']}

### 依据说明
{r['evidence']}

### 模型回答
{r['model_answer']}

---

"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n📄 报告已保存: {output_path}")


def main():
    """主函数"""
    # 路径配置
    examples_dir = Path(__file__).parent
    qa_file = examples_dir / "example_input" / "13_qa.jsonl"
    output_file = examples_dir / "docqa_results.md"
    
    print("=" * 60)
    print("文档问答评估示例")
    print("=" * 60)
    
    # 1. 构建知识图谱
    print("\n[Step 1] 构建知识图谱")
    if not build_knowledge_graph():
        print("❌ 知识图谱构建失败，退出")
        return
    
    # 2. 回答问题
    print("\n[Step 2] 回答问题")
    if not os.path.exists(qa_file):
        print(f"❌ 问题文件不存在: {qa_file}")
        return
    
    results = asyncio.run(answer_questions(str(qa_file)))
    
    # 3. 生成报告
    print("\n[Step 3] 生成报告")
    generate_report(results, str(output_file))
    
    print("\n" + "=" * 60)
    print("✅ 评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
