"""
测试嵌入模型部署兼容性
检查系统是否能顺利运行 sentence-transformers/all-MiniLM-L6-v2
"""

import sys
import os
import torch
import psutil
import platform
from datetime import datetime

def print_section(title):
    """打印分隔线"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_system_info():
    """检查系统基本信息"""
    print_section("1. 系统信息")

    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"处理器: {platform.processor()}")

    # 内存信息
    memory = psutil.virtual_memory()
    print(f"总内存: {memory.total / (1024**3):.2f} GB")
    print(f"可用内存: {memory.available / (1024**3):.2f} GB")
    print(f"内存使用率: {memory.percent}%")

def check_python_packages():
    """检查Python依赖包"""
    print_section("2. Python依赖包检查")

    packages = {
        'torch': 'PyTorch深度学习框架',
        'transformers': 'HuggingFace transformers库',
        'sentence_transformers': 'Sentence transformers库',
        'numpy': 'NumPy数值计算库',
    }

    missing_packages = []

    for package, description in packages.items():
        try:
            if package == 'sentence_transformers':
                import sentence_transformers
                version = sentence_transformers.__version__
            else:
                mod = __import__(package)
                version = getattr(mod, '__version__', '未知版本')

            print(f"✅ {package:20s} {version:15s} - {description}")
        except ImportError:
            print(f"❌ {package:20s} {'未安装':15s} - {description}")
            missing_packages.append(package)

    return missing_packages

def check_gpu_availability():
    """检查GPU可用性"""
    print_section("3. GPU/CUDA 检查")

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        print(f"✅ CUDA可用: 是")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n   GPU {i}: {props.name}")
            print(f"   显存: {props.total_memory / (1024**3):.2f} GB")
            print(f"   计算能力: {props.major}.{props.minor}")

        # 检查当前GPU内存使用
        if torch.cuda.device_count() > 0:
            print(f"\n   GPU 0 显存使用:")
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"   已分配: {memory_allocated:.2f} GB")
            print(f"   已保留: {memory_reserved:.2f} GB")
    else:
        print(f"❌ CUDA可用: 否")
        print(f"   将使用CPU运行（速度较慢）")

    return cuda_available

def check_disk_space():
    """检查磁盘空间"""
    print_section("4. 磁盘空间检查")

    # 检查当前目录磁盘空间
    disk = psutil.disk_usage(os.getcwd())
    print(f"当前目录: {os.getcwd()}")
    print(f"总空间: {disk.total / (1024**3):.2f} GB")
    print(f"已使用: {disk.used / (1024**3):.2f} GB")
    print(f"可用空间: {disk.free / (1024**3):.2f} GB")
    print(f"使用率: {disk.percent}%")

    # 模型缓存大约需要500MB-1GB空间
    if disk.free < 2 * (1024**3):  # 小于2GB
        print(f"⚠️  警告: 可用磁盘空间较少，建议至少保留2GB以上")
    else:
        print(f"✅ 磁盘空间充足")

def test_model_loading():
    """测试模型加载"""
    print_section("5. 模型加载测试")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"目标模型: {model_name}")
    print(f"模型大小: 约 120MB (下载后)")
    print(f"嵌入维度: 384")
    print(f"最大序列长度: 512 tokens")

    try:
        print("\n开始加载模型...")
        from transformers import AutoTokenizer, AutoModel

        # 测试tokenizer加载
        print("📥 加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer 加载成功")

        # 测试模型加载
        print("📥 加载模型...")

        # 检查是否有GPU
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        print(f"   设备映射: {device_map}")

        model = AutoModel.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("✅ 模型加载成功")

        # 测试推理
        print("\n🧪 测试推理...")
        test_text = "这是一个测试句子。"
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)

        # 将输入移动到正确的设备
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        print(f"✅ 推理成功!")
        print(f"   输入文本: {test_text}")
        print(f"   输出形状: {outputs.last_hidden_state.shape}")

        return True

    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        print(f"\n可能的解决方案:")
        print(f"1. 检查网络连接（需要从HuggingFace下载模型）")
        print(f"2. 安装缺失的依赖包")
        print(f"3. 检查磁盘空间是否充足")
        return False

def provide_recommendations(cuda_available, missing_packages):
    """提供部署建议"""
    print_section("6. 部署建议")

    print("📋 系统要求总结:")
    print("   • 最低内存: 4GB RAM")
    print("   • 推荐内存: 8GB+ RAM")
    print("   • 磁盘空间: 至少2GB可用空间")
    print("   • 网络: 首次运行需要下载模型")

    print("\n🚀 性能建议:")
    if cuda_available:
        print("   ✅ 检测到GPU，将使用GPU加速（推荐）")
        print("   • GPU推理速度比CPU快10-50倍")
        print("   • 建议batch_size设置为4-16")
    else:
        print("   ⚠️  未检测到GPU，将使用CPU运行")
        print("   • CPU推理速度较慢，建议batch_size设置为1-4")
        print("   • 考虑安装CUDA版本PyTorch以获得GPU加速")

    if missing_packages:
        print(f"\n📦 安装缺失依赖:")
        print(f"   pip install {' '.join(missing_packages)}")

    print("\n📝 配置建议:")
    print("   • 如果内存充足，可以增加 batch_size")
    print("   • 如果内存受限，减小 batch_size 和 max_workers")
    print("   • 使用量化模型可以减少内存占用")

def main():
    """主函数"""
    print("🔍 MiniRAG 嵌入模型部署兼容性检查")
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 系统信息
    check_system_info()

    # 2. Python包检查
    missing_packages = check_python_packages()

    # 3. GPU检查
    cuda_available = check_gpu_availability()

    # 4. 磁盘空间
    check_disk_space()

    # 5. 如果没有缺失包，测试模型加载
    if not missing_packages:
        success = test_model_loading()
    else:
        print("\n⚠️  检测到缺失依赖包，跳过模型加载测试")
        print("   请先安装缺失的依赖包，然后重新运行此脚本")
        success = False

    # 6. 提供建议
    provide_recommendations(cuda_available, missing_packages)

    # 最终结果
    print_section("检查结果")
    if success and not missing_packages:
        print("✅ 你的系统可以顺利部署该嵌入模型！")
        print("   可以开始使用 MiniRAG 进行图索引构建。")
    else:
        print("❌ 检测到问题，请根据上述建议进行修复")
        print("   修复后可以重新运行此脚本进行验证。")

    print(f"\n检查完成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()