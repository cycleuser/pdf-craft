#!/usr/bin/env python
"""
Run script for the PDF to Markdown Web Converter
"""

import os
import sys
import platform
import logging
import subprocess
from pathlib import Path
from app import app

def check_gpu_availability():
    """检查GPU是否可用"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return True, f"可用 ({device_count} 个设备, {device_name})"
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
        found = [f for f in files if any(os.path.join(root, f).endswith(f) for root, _, files in os.walk(model_dir) for f in files)]
        model_status[model_name] = f"{'✓' if found else '✗'} ({', '.join(found) if found else '未找到'})"

    return model_status

def check_onnxruntime():
    """检查onnxruntime版本"""
    try:
        import onnxruntime
        version = onnxruntime.__version__
        providers = onnxruntime.get_available_providers()
        gpu_support = "CUDAExecutionProvider" in providers
        return f"版本 {version}, GPU支持: {'是' if gpu_support else '否'}, 可用提供者: {', '.join(providers)}"
    except ImportError:
        return "未安装"
    except Exception as e:
        return f"错误: {str(e)}"

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("pdf-markdown-converter")

    # Ensure all required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_DIR'], exist_ok=True)

    # Get port from command line arguments or use default
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    # 显示系统信息
    print("\n===== PDF to Markdown Web Converter =====")
    print(f"系统信息: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Python版本: {platform.python_version()}")

    # 检查GPU可用性
    gpu_available, gpu_info = check_gpu_availability()
    print(f"GPU加速: {gpu_info}")

    # 检查ONNX Runtime
    onnx_info = check_onnxruntime()
    print(f"ONNX Runtime: {onnx_info}")

    # 设置默认设备
    if gpu_available:
        app.config['USE_GPU'] = True
        print("已自动启用GPU加速")
    else:
        app.config['USE_GPU'] = False
        print("使用CPU进行处理")

    # 检查模型
    model_status = check_models()
    if isinstance(model_status, dict):
        print("\n模型状态:")
        for model_name, status in model_status.items():
            print(f"  - {model_name}: {status}")
    else:
        print(f"\n模型状态: {model_status}")

    print(f"\n模型目录: {app.config['MODEL_DIR']}")
    print(f"上传目录: {app.config['UPLOAD_FOLDER']}")
    print(f"结果目录: {app.config['RESULTS_FOLDER']}")

    print(f"\n启动Web服务器，端口: {port}")
    print(f"请在浏览器中访问: http://localhost:{port}")
    print("====================================\n")

    # 启动应用
    app.run(host='0.0.0.0', port=port, debug=True)