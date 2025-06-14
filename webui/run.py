#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF OCR 统一启动脚本
支持多种运行模式，通过页面选择
"""

import os
import sys
import platform
import logging
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from app import app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """打印启动横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    基于光学字符识别和本地大语言模型的文档转换平台                      ║
    ║                                                              ║
    ║  📄 支持中文文件名处理                                        ║
    ║  🚀 多种运行模式可选                                          ║
    ║  🌐 支持外网访问                                              ║
    ║  🎨 优化的用户界面                                            ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """检查依赖项"""
    logger.info("检查依赖项...")
    
    missing_deps = []
    
    # 检查必需的包
    required_packages = [
        ('flask', 'Flask'),
        ('pdf_craft', 'PDF-Craft'),
        ('psutil', 'psutil'),
        ('fitz', 'PyMuPDF')
    ]
    
    for package, name in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {name} 已安装")
        except ImportError:
            missing_deps.append(name)
            logger.warning(f"❌ {name} 未安装")
    
    if missing_deps:
        print(f"\n⚠️  缺少依赖项: {', '.join(missing_deps)}")
        print("请运行以下命令安装:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_system_info():
    """检查系统信息"""
    print(f"\n📊 系统信息:")
    print(f"  操作系统: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"  Python版本: {platform.python_version()}")
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device_count > 0 else 0
            print(f"  GPU状态: 可用 ({device_count} 个设备, {device_name}, {memory_gb:.1f}GB)")
        else:
            print(f"  GPU状态: 不可用 (CUDA未启用)")
    except ImportError:
        print(f"  GPU状态: 不可用 (PyTorch未安装)")
    
    # 检查内存
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"  系统内存: {memory.total / 1024**3:.1f}GB (可用: {memory.available / 1024**3:.1f}GB)")
    except:
        pass
    
    # 检查CPU
    cpu_count = os.cpu_count() or 1
    print(f"  CPU核心数: {cpu_count}")

def setup_directories():
    """设置必要的目录"""
    directories = [
        app.config['UPLOAD_FOLDER'],
        app.config['RESULTS_FOLDER'],
        app.config['MODEL_DIR']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"\n📂 目录配置:")
    print(f"  模型目录: {app.config['MODEL_DIR']}")
    print(f"  上传目录: {app.config['UPLOAD_FOLDER']}")
    print(f"  结果目录: {app.config['RESULTS_FOLDER']}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PDF OCR 智能处理平台')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址 (默认: 0.0.0.0，支持外网访问)')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口 (默认: 5000)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 打印启动横幅
    print_banner()
    
    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)
    
    # 检查系统信息
    check_system_info()
    
    # 设置目录
    setup_directories()
    
    # 设置默认配置（兼容模式）
    app.config['ENABLE_MULTIPROCESSING'] = False  # 默认禁用多进程
    app.config['PRELOAD_MODELS'] = False  # 默认禁用模型预加载
    app.config['USE_GPU'] = False  # 默认使用CPU
    
    print("\n⚙️  默认配置:")
    print("  运行模式: 兼容模式（可在页面中更改）")
    print("  GPU加速: 默认禁用（可在页面中启用）")
    print("  多进程: 默认禁用（可在页面中启用）")
    
    # 启动应用
    print(f"\n🚀 启动OCR服务...")
    print(f"   本地访问: http://localhost:{args.port}")
    if args.host == '0.0.0.0':
        print(f"   外网访问: http://[您的IP地址]:{args.port}")
    print(f"   调试模式: {'开启' if args.debug else '关闭'}")
    print("\n按 Ctrl+C 停止服务")
    print("=" * 60)
    
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
        logger.info("应用正常退出")
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        print(f"\n❌ 启动失败: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()