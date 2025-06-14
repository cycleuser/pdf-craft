#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的PDF OCR启动脚本
结合了基础功能检查和性能优化配置
支持外网访问和完整的系统检测
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
from optimization_config import OptimizationConfigManager, auto_configure_optimization, get_preset_config, list_preset_configs

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
    ║                PDF OCR 智能处理平台                           ║
    ║                                                              ║
    ║  🚀 自动检测硬件配置并应用最优设置                            ║
    ║  ⚡ 最大化利用GPU、CPU和内存资源                             ║
    ║  📈 显著提升PDF OCR处理速度                                  ║
    ║  🌐 支持外网访问和本地部署                                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_gpu_availability():
    """检查GPU是否可用"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device_count > 0 else 0
            return True, f"可用 ({device_count} 个设备, {device_name}, {memory_gb:.1f}GB)"
        else:
            return False, "不可用 (CUDA未启用)"
    except ImportError:
        return False, "不可用 (PyTorch未安装)"
    except Exception as e:
        return False, f"不可用 (错误: {str(e)})"

def check_models():
    """检查模型文件"""
    model_dir = app.config['MODEL_DIR']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        return f"模型目录已创建: {model_dir}，首次运行时将自动下载模型"

    model_files = os.listdir(model_dir)
    if not model_files:
        return f"模型目录为空: {model_dir}，首次运行时将自动下载模型"

    # 检查关键模型文件
    key_models = {
        "DocLayout-YOLO": ["doclayout_yolo.pt", "yolov8n.pt"],
        "OnnxOCR": ["model_cn.onnx", "model_en.onnx"],
        "LayoutReader": ["layoutreader.onnx"]
    }

    model_status = {}
    for model_name, files in key_models.items():
        found = []
        for root, _, filenames in os.walk(model_dir):
            for filename in filenames:
                if any(filename.endswith(f) or f in filename for f in files):
                    found.append(filename)
        model_status[model_name] = f"{'✓' if found else '✗'} ({', '.join(found[:2]) if found else '未找到'})"

    return model_status

def check_onnxruntime():
    """检查onnxruntime版本"""
    try:
        import onnxruntime
        version = onnxruntime.__version__
        providers = onnxruntime.get_available_providers()
        gpu_support = "CUDAExecutionProvider" in providers
        return f"版本 {version}, GPU支持: {'是' if gpu_support else '否'}"
    except ImportError:
        return "未安装"
    except Exception as e:
        return f"错误: {str(e)}"

def check_dependencies():
    """检查依赖项"""
    logger.info("检查依赖项...")
    
    missing_deps = []
    
    # 检查必需的包
    required_packages = [
        ('torch', 'PyTorch'),
        ('pdf_craft', 'PDF-Craft'),
        ('flask', 'Flask'),
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

def setup_environment():
    """设置环境变量"""
    logger.info("设置环境变量...")
    
    # 设置CUDA相关环境变量
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    
    # 设置内存增长策略
    os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
    
    # 设置线程数
    cpu_count = os.cpu_count() or 1
    os.environ.setdefault('OMP_NUM_THREADS', str(min(8, cpu_count)))
    os.environ.setdefault('MKL_NUM_THREADS', str(min(8, cpu_count)))
    
    logger.info("环境变量设置完成")

def apply_optimization_config(preset_name=None):
    """应用优化配置"""
    print("\n🔧 应用性能优化配置...")
    
    try:
        if preset_name:
            # 使用预设配置
            preset = get_preset_config(preset_name)
            print(f"🎛️  应用预设配置: {preset['name']}")
            print(f"   描述: {preset['description']}")
            
            config = preset['config']
            for key, value in config.items():
                config_key = key.upper()
                if hasattr(app.config, config_key):
                    app.config[config_key] = value
                    
            return True, f"已应用预设配置: {preset['name']}"
        else:
            # 自动检测和配置
            optimal_config, recommendations = auto_configure_optimization()
            
            # 应用优化配置
            app.config['USE_GPU'] = optimal_config.get('device') == 'cuda'
            app.config['ENABLE_MULTIPROCESSING'] = optimal_config.get('enable_multiprocessing', False)
            app.config['ENABLE_MIXED_PRECISION'] = optimal_config.get('enable_mixed_precision', False)
            app.config['OPTIMIZE_MEMORY'] = optimal_config.get('optimize_memory', True)
            app.config['PRELOAD_MODELS'] = optimal_config.get('preload_models', False)
            app.config['PROCESS_POOL_SIZE'] = optimal_config.get('process_pool_size', 2)
            app.config['GPU_BATCH_SIZE'] = optimal_config.get('gpu_batch_size', 4)
            app.config['CPU_BATCH_SIZE'] = optimal_config.get('cpu_batch_size', 2)
            app.config['MAX_WORKERS'] = optimal_config.get('max_workers', 4)
            
            print(f"✅ GPU加速: {'启用' if app.config['USE_GPU'] else '禁用'}")
            print(f"✅ 多进程处理: {'启用' if app.config['ENABLE_MULTIPROCESSING'] else '禁用'}")
            print(f"✅ 混合精度: {'启用' if app.config['ENABLE_MIXED_PRECISION'] else '禁用'}")
            print(f"✅ 内存优化: {'启用' if app.config['OPTIMIZE_MEMORY'] else '禁用'}")
            print(f"✅ 模型预加载: {'启用' if app.config['PRELOAD_MODELS'] else '禁用'}")
            print(f"📊 性能等级: {recommendations['performance_tier']}")
            print(f"📈 预期提升: {recommendations['estimated_speedup']}")
            
            return True, f"自动优化完成，性能等级: {recommendations['performance_tier']}"
            
    except Exception as e:
        logger.error(f"优化配置应用失败: {str(e)}")
        return False, f"优化失败: {str(e)}"

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PDF OCR 智能处理平台')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址 (默认: 0.0.0.0，支持外网访问)')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口 (默认: 5000)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--no-auto-config', action='store_true', help='禁用自动优化配置')
    parser.add_argument('--preset', choices=['maximum_speed', 'balanced', 'maximum_quality', 'low_resource'], 
                       help='使用预设配置')
    parser.add_argument('--list-presets', action='store_true', help='列出所有预设配置')
    
    args = parser.parse_args()
    
    # 列出预设配置
    if args.list_presets:
        presets = list_preset_configs()
        print("\n可用的预设配置:")
        for name, info in presets.items():
            print(f"  {name}: {info['name']} - {info['description']}")
        return
    
    # 打印启动横幅
    print_banner()
    
    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)
    
    # 设置环境
    setup_environment()
    
    # 确保所有必需目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
    
    # 显示系统信息
    print(f"\n📊 系统信息:")
    print(f"  操作系统: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"  Python版本: {platform.python_version()}")
    
    # 检查GPU可用性
    gpu_available, gpu_info = check_gpu_availability()
    print(f"  GPU状态: {gpu_info}")
    
    # 检查ONNX Runtime
    onnx_info = check_onnxruntime()
    print(f"  ONNX Runtime: {onnx_info}")
    
    # 检查模型
    model_status = check_models()
    if isinstance(model_status, dict):
        print(f"\n📁 模型状态:")
        for model_name, status in model_status.items():
            print(f"  {model_name}: {status}")
    else:
        print(f"\n📁 模型状态: {model_status}")
    
    print(f"\n📂 目录配置:")
    print(f"  模型目录: {app.config['MODEL_DIR']}")
    print(f"  上传目录: {app.config['UPLOAD_FOLDER']}")
    print(f"  结果目录: {app.config['RESULTS_FOLDER']}")
    
    # 应用优化配置
    if not args.no_auto_config:
        success, message = apply_optimization_config(args.preset)
        if success:
            print(f"✅ {message}")
        else:
            print(f"⚠️  {message}")
            print("将使用默认配置启动...")
    else:
        print("⚠️  已禁用自动优化配置")
    
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