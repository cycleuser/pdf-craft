#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化启动脚本
自动检测系统配置并应用最优设置启动OCR应用
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from optimization_config import OptimizationConfigManager, auto_configure_optimization
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
    ║                    OCR性能优化启动器                          ║
    ║                                                              ║
    ║  🚀 自动检测硬件配置并应用最优设置                            ║
    ║  ⚡ 最大化利用GPU、CPU和内存资源                             ║
    ║  📈 显著提升PDF OCR处理速度                                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def detect_and_configure():
    """检测系统配置并应用优化设置"""
    logger.info("开始检测系统配置...")
    
    try:
        # 创建优化配置管理器
        config_manager = OptimizationConfigManager()
        
        # 获取系统信息
        system_info = config_manager.system_info
        optimal_config = config_manager.optimal_config
        recommendations = config_manager.get_recommendations()
        
        # 打印系统信息
        print("\n📊 系统配置检测结果:")
        print(f"  CPU核心数: {system_info['cpu_count']}")
        print(f"  内存总量: {system_info['memory_total_gb']:.1f} GB")
        print(f"  可用内存: {system_info['memory_available_gb']:.1f} GB")
        
        if system_info['has_gpu']:
            print(f"  GPU数量: {system_info['gpu_count']}")
            print(f"  GPU型号: {', '.join(system_info['gpu_names'])}")
            print(f"  GPU显存: {system_info['gpu_memory_gb']:.1f} GB")
        else:
            print("  GPU: 未检测到CUDA兼容GPU")
        
        print(f"\n🎯 性能等级: {recommendations['performance_tier']}")
        print(f"📈 预期性能提升: {recommendations['estimated_speedup']}")
        
        # 应用优化配置
        print("\n⚙️ 应用优化配置:")
        
        # GPU配置
        if optimal_config['device'] == 'cuda':
            app.config['USE_GPU'] = True
            print(f"  ✅ 启用GPU加速 (批处理大小: {optimal_config['gpu_batch_size']})")
        else:
            app.config['USE_GPU'] = False
            print(f"  ⚠️  使用CPU模式 (批处理大小: {optimal_config['cpu_batch_size']})")
        
        # 多进程配置
        app.config['ENABLE_MULTIPROCESSING'] = optimal_config['enable_multiprocessing']
        if optimal_config['enable_multiprocessing']:
            app.config['PROCESS_POOL_SIZE'] = optimal_config['process_pool_size']
            print(f"  ✅ 启用多进程处理 (进程数: {optimal_config['process_pool_size']})")
        else:
            print("  ⚠️  禁用多进程处理")
        
        # 混合精度
        app.config['ENABLE_MIXED_PRECISION'] = optimal_config['enable_mixed_precision']
        if optimal_config['enable_mixed_precision']:
            print("  ✅ 启用混合精度训练")
        
        # 内存优化
        app.config['OPTIMIZE_MEMORY'] = optimal_config['optimize_memory']
        if optimal_config['optimize_memory']:
            print("  ✅ 启用内存优化")
        
        # 模型预加载
        app.config['PRELOAD_MODELS'] = optimal_config['preload_models']
        if optimal_config['preload_models']:
            print("  ✅ 启用模型预加载")
        
        # 其他配置
        app.config['MAX_WORKERS'] = optimal_config['max_workers']
        app.config['GPU_BATCH_SIZE'] = optimal_config['gpu_batch_size']
        app.config['CPU_BATCH_SIZE'] = optimal_config['cpu_batch_size']
        
        # 打印优化建议
        if recommendations['recommendations']:
            print("\n💡 优化建议:")
            for i, rec in enumerate(recommendations['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        logger.info("系统配置检测和优化设置完成")
        return True
        
    except Exception as e:
        logger.error(f"配置检测失败: {str(e)}")
        print(f"\n❌ 配置检测失败: {str(e)}")
        print("将使用默认配置启动...")
        return False

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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OCR性能优化启动器')
    parser.add_argument('--host', default='127.0.0.1', help='服务器地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--no-auto-config', action='store_true', help='禁用自动配置')
    parser.add_argument('--preset', choices=['maximum_speed', 'balanced', 'maximum_quality', 'low_resource'], 
                       help='使用预设配置')
    
    args = parser.parse_args()
    
    # 打印启动横幅
    print_banner()
    
    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)
    
    # 设置环境
    setup_environment()
    
    # 配置优化设置
    if not args.no_auto_config:
        if args.preset:
            logger.info(f"使用预设配置: {args.preset}")
            from optimization_config import get_preset_config
            preset = get_preset_config(args.preset)
            print(f"\n🎛️  应用预设配置: {preset['name']}")
            print(f"   描述: {preset['description']}")
            
            # 应用预设配置
            config = preset['config']
            for key, value in config.items():
                if hasattr(app.config, key.upper()):
                    app.config[key.upper()] = value
        else:
            # 自动检测和配置
            detect_and_configure()
    
    # 启动应用
    print(f"\n🚀 启动OCR服务...")
    print(f"   地址: http://{args.host}:{args.port}")
    print(f"   调试模式: {'开启' if args.debug else '关闭'}")
    print("\n按 Ctrl+C 停止服务")
    
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