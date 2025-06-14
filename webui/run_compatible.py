#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兼容模式启动脚本
专门解决pdf_craft库兼容性问题，禁用可能导致错误的高级功能
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
    ║                PDF OCR 兼容模式启动器                         ║
    ║                                                              ║
    ║  🔧 自动适配pdf_craft库API                                   ║
    ║  🛡️ 禁用可能导致错误的高级功能                               ║
    ║  🌐 支持外网访问和本地部署                                    ║
    ║  ⚠️ 使用简化配置以确保稳定性                                 ║
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

def check_pdf_craft_version():
    """检查pdf_craft库版本和功能"""
    try:
        from pdf_craft import PDFPageExtractor
        
        # 检查PDFPageExtractor支持的参数
        import inspect
        extractor_signature = inspect.signature(PDFPageExtractor.__init__)
        supported_params = list(extractor_signature.parameters.keys())
        
        print(f"\n📦 PDF-Craft库信息:")
        print(f"  支持的初始化参数: {', '.join(supported_params)}")
        
        # 检查是否支持批处理
        supports_batch = 'batch_size' in supported_params
        print(f"  支持批处理: {'✅ 是' if supports_batch else '❌ 否'}")
        
        # 检查是否支持设备选择
        supports_device = 'device' in supported_params
        print(f"  支持设备选择: {'✅ 是' if supports_device else '❌ 否'}")
        
        # 检查是否支持优化
        extractor = PDFPageExtractor()
        supports_optimization = hasattr(extractor, 'enable_optimization')
        print(f"  支持优化选项: {'✅ 是' if supports_optimization else '❌ 否'}")
        
        # 检查是否支持单页提取
        supports_page_extraction = hasattr(extractor, 'extract_page')
        print(f"  支持单页提取: {'✅ 是' if supports_page_extraction else '❌ 否'}")
        
        return {
            'supports_batch': supports_batch,
            'supports_device': supports_device,
            'supports_optimization': supports_optimization,
            'supports_page_extraction': supports_page_extraction
        }
        
    except Exception as e:
        logger.error(f"检查PDF-Craft库失败: {str(e)}")
        return {
            'supports_batch': False,
            'supports_device': False,
            'supports_optimization': False,
            'supports_page_extraction': False
        }

def apply_compatible_config(pdf_craft_features):
    """应用兼容性配置"""
    print("\n🔧 应用兼容性配置...")
    
    # 禁用批处理相关功能
    if not pdf_craft_features['supports_batch']:
        app.config['ENABLE_MULTIPROCESSING'] = False
        print("  ❌ 已禁用多进程处理 (不支持批处理)")
    
    # 禁用设备选择
    if not pdf_craft_features['supports_device']:
        app.config['USE_GPU'] = False
        print("  ❌ 已禁用GPU加速 (不支持设备选择)")
    
    # 禁用优化
    if not pdf_craft_features['supports_optimization']:
        app.config['ENABLE_MIXED_PRECISION'] = False
        app.config['OPTIMIZE_MEMORY'] = False
        print("  ❌ 已禁用优化选项 (不支持优化)")
    
    # 禁用模型预加载
    app.config['PRELOAD_MODELS'] = False
    print("  ❌ 已禁用模型预加载 (确保兼容性)")
    
    # 确保目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
    
    print("  ✅ 已创建必要目录")
    print("  ✅ 兼容性配置已应用")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PDF OCR 兼容模式启动器')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址 (默认: 0.0.0.0，支持外网访问)')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口 (默认: 5000)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 打印启动横幅
    print_banner()
    
    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)
    
    # 检查pdf_craft版本和功能
    pdf_craft_features = check_pdf_craft_version()
    
    # 应用兼容性配置
    apply_compatible_config(pdf_craft_features)
    
    # 显示系统信息
    print(f"\n📊 系统信息:")
    print(f"  操作系统: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"  Python版本: {platform.python_version()}")
    
    # 检查GPU可用性
    gpu_info = "不可用 (已禁用)"
    if pdf_craft_features['supports_device'] and app.config['USE_GPU']:
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device_count > 0 else 0
                gpu_info = f"可用 ({device_count} 个设备, {device_name}, {memory_gb:.1f}GB)"
            else:
                gpu_info = "不可用 (CUDA未启用)"
        except ImportError:
            gpu_info = "不可用 (PyTorch未安装)"
    
    print(f"  GPU状态: {gpu_info}")
    
    print(f"\n📂 目录配置:")
    print(f"  模型目录: {app.config['MODEL_DIR']}")
    print(f"  上传目录: {app.config['UPLOAD_FOLDER']}")
    print(f"  结果目录: {app.config['RESULTS_FOLDER']}")
    
    # 启动应用
    print(f"\n🚀 启动OCR服务 (兼容模式)...")
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