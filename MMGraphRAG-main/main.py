"""
MMKG构建模块 - 自动化多模态知识图谱生成
"""

import argparse
import asyncio
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

# Suppress Ultralytics warnings
os.environ['YOLO_VERBOSE'] = 'False'
warnings.filterwarnings("ignore")

# 确保项目根目录在Python路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.builder import MMKGBuilder
from src import parameter
from src.retrieval import GraphRAGQuery
from src.parameter import QueryParam

# 配置基础日志
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("main")

# Suppress other loggers
logging.getLogger("ultralytics").setLevel(logging.ERROR)


def setup_config(args):
    """根据命令行参数配置全局环境"""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    if args.working_dir:
        parameter.WORKING_DIR = args.working_dir
        
    if args.output_dir:
        parameter.OUTPUT_DIR = args.output_dir
        
    if args.method:
        parameter.USE_MINERU = (args.method.lower() == 'mineru')
        
    # 如果强制重新构建，清空工作目录
    if args.force and os.path.exists(parameter.WORKING_DIR):
        logger.warning(f"正在清空工作目录: {parameter.WORKING_DIR}")
        shutil.rmtree(parameter.WORKING_DIR)
        
    # 打印最终配置
    logger.info("=" * 40)
    logger.info("当前运行配置:")
    if args.query:
         logger.info(f"- 模式: 查询 (RAG)")
    else:
        logger.info(f"- 模式: 构建")
        logger.info(f"- 输入路径: {args.input_path}")
        logger.info(f"- 预处理方法: {'MinerU' if parameter.USE_MINERU else 'PyMuPDF'}")
        
    logger.info(f"- 工作目录: {parameter.WORKING_DIR}")
    logger.info(f"- 输出目录: {parameter.OUTPUT_DIR}")
    logger.info("=" * 40)


def process_file(pdf_path: str):
    """处理单个 PDF 文件"""
    if not os.path.exists(pdf_path):
        logger.error(f"文件不存在: {pdf_path}")
        return

    logger.info(f"🔨 开始处理: {pdf_path}")
    try:
        builder = MMKGBuilder()
        builder.index(pdf_path)
        logger.info(f"✅ 处理完成: {pdf_path}")
    except Exception as e:
        logger.error(f"❌ 处理失败 {pdf_path}: {e}", exc_info=True)


def run_query(args):
    """执行查询模式"""
    logger.info("🔍 执行 RAG 查询...")
    logger.info(f"Query: {args.query}")
    
    # 初始化参数
    param = QueryParam()
    if args.top_k:
        param.top_k = args.top_k
    if args.response_type:
        param.response_type = args.response_type
        
    rag = GraphRAGQuery()
    
    try:
        response = asyncio.run(rag.query(args.query, param))
        print("\n" + "="*30 + " 最终回答 " + "="*30 + "\n")
        print(response)
        print("\n" + "="*72 + "\n")
        logger.info(f"✅ 查询完成，日志已保存至 {os.path.join(parameter.OUTPUT_DIR, 'retrieval_log.md')}")
    except Exception as e:
        logger.error(f"❌ 查询失败: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description='多模态知识图谱构建工具 (MMKG Builder)')
    
    # 核心参数
    parser.add_argument('-i', '--input', dest='input_path', type=str, required=False,
                        help='PDF文件路径或包含PDF的目录')
    parser.add_argument('-w', '--working', dest='working_dir', type=str,
                        help='中间工作目录 (默认: working)', default=None)
    parser.add_argument('-o', '--output', dest='output_dir', type=str,
                        help='最终输出目录 (默认: output)', default=None)
    
    # 构建控制
    parser.add_argument('-m', '--method', choices=['mineru', 'pymupdf'],
                        help='PDF预处理方法 (默认使用 parameter.py 配置)')
    parser.add_argument('-f', '--force', action='store_true',
                        help='强制清空工作目录重新构建')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='显示详细调试日志')
                        
    # 查询参数
    parser.add_argument('-q', '--query', dest='query', type=str,
                        help='执行RAG查询')
    parser.add_argument('--top_k', type=int, help='检索实体数量', default=None)
    parser.add_argument('--response_type', type=str, help='回答风格', default=None)
    
    # 可视化服务器
    parser.add_argument('-s', '--server', action='store_true',
                        help='启动知识图谱可视化服务器')
    parser.add_argument('--port', type=int, default=8080,
                        help='服务器端口 (默认: 8080)')
    parser.add_argument('--graph', dest='graph_path', type=str,
                        help='指定图谱文件路径 (.graphml)')

    args = parser.parse_args()

    # 参数依赖校验
    if not args.query and (args.top_k or args.response_type):
        parser.error("参数 --top_k 和 --response_type 仅在查询模式 (-q) 下有效")

    # 配置环境
    setup_config(args)
    
    # 可视化服务器模式
    if args.server:
        from src.visualization.server import run_visualization_server
        run_visualization_server(
            parameter.OUTPUT_DIR, 
            parameter.WORKING_DIR, 
            args.port,
            args.graph_path
        )
        return

    # 确定输入路径 (所有模式都需要，用于构建或作为默认上下文)
    input_path = None
    if args.input_path:
        input_path = Path(args.input_path)
    elif parameter.INPUT_PDF_PATH:
        input_path = Path(parameter.INPUT_PDF_PATH)
    
    # RAG 查询模式
    if args.query:
        # 检查图谱是否存在
        graph_exists = (
            os.path.exists(os.path.join(parameter.OUTPUT_DIR, "mmkg.graphml")) or 
            os.path.exists(os.path.join(parameter.OUTPUT_DIR, "kg.graphml"))
        )
        
        if not graph_exists:
            logger.warning("⚠️ 未检测到已构建的知识图谱，将优先执行构建流程...")
            if not input_path:
                 parser.error("需要执行构建但未指定输入路径，且 parameter.py 中无默认路径")
            
            # 执行构建
            if input_path.is_file() and input_path.suffix.lower() == '.pdf':
                process_file(str(input_path))
            else:
                 logger.error(f"无法构建: 无效的文件路径 {input_path}")
                 return

        # 执行查询
        run_query(args)
        return

    # 纯构建模式
    if not input_path:
        parser.error("未指定输入路径，且 parameter.py 中无默认路径")

    if input_path.is_file():
        if input_path.suffix.lower() == '.pdf':
            process_file(str(input_path))
        else:
            logger.error(f"不支持的文件格式: {input_path.suffix}，仅支持 .pdf 文件")
    else:
        logger.error(f"无效的文件路径: {input_path} (仅支持单个 PDF 文件)")


if __name__ == "__main__":
    main()
