#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR引擎安装脚本
自动安装和配置多种OCR引擎
"""

import os
import sys
import subprocess
import platform
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, check=True):
    """运行命令"""
    logger.info(f"执行命令: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"输出: {result.stdout}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e}")
        if e.stderr:
            logger.error(f"错误: {e.stderr}")
        return False

def install_tesseract():
    """安装Tesseract OCR"""
    logger.info("安装Tesseract OCR...")
    
    system = platform.system().lower()
    
    if system == "windows":
        logger.info("Windows系统，请手动下载安装Tesseract:")
        logger.info("1. 访问: https://github.com/UB-Mannheim/tesseract/wiki")
        logger.info("2. 下载Windows安装包")
        logger.info("3. 安装后将tesseract.exe路径添加到系统PATH")
        logger.info("4. 或者使用conda: conda install -c conda-forge tesseract")
        
    elif system == "darwin":  # macOS
        logger.info("macOS系统，尝试使用Homebrew安装...")
        if run_command("brew install tesseract", check=False):
            logger.info("Tesseract安装成功")
        else:
            logger.warning("Homebrew安装失败，请手动安装:")
            logger.info("brew install tesseract")
            
    elif system == "linux":
        logger.info("Linux系统，尝试使用包管理器安装...")
        # 尝试不同的包管理器
        if run_command("which apt-get", check=False):
            run_command("sudo apt-get update", check=False)
            if run_command("sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra", check=False):
                logger.info("Tesseract安装成功")
            else:
                logger.warning("apt-get安装失败")
        elif run_command("which yum", check=False):
            if run_command("sudo yum install -y tesseract tesseract-langpack-chi_sim tesseract-langpack-chi_tra", check=False):
                logger.info("Tesseract安装成功")
            else:
                logger.warning("yum安装失败")
        else:
            logger.warning("未找到支持的包管理器，请手动安装Tesseract")

def install_python_packages():
    """安装Python包"""
    logger.info("安装Python OCR包...")
    
    packages = [
        "pytesseract",
        "Pillow",
        "opencv-python",
        "easyocr",
        "paddlepaddle",
        "paddleocr",
        "rapidocr-onnxruntime",
        "pdfplumber",
        "pdf2image",
        "numpy",
        "scikit-image"
    ]
    
    for package in packages:
        logger.info(f"安装 {package}...")
        if run_command(f"{sys.executable} -m pip install {package}", check=False):
            logger.info(f"{package} 安装成功")
        else:
            logger.warning(f"{package} 安装失败，可能需要手动安装")

def test_ocr_engines():
    """测试OCR引擎"""
    logger.info("测试OCR引擎...")
    
    # 测试Tesseract
    try:
        import pytesseract
        from PIL import Image
        test_image = Image.new('RGB', (100, 50), color='white')
        pytesseract.image_to_string(test_image)
        logger.info("✓ Tesseract OCR 可用")
    except Exception as e:
        logger.warning(f"✗ Tesseract OCR 不可用: {e}")
    
    # 测试EasyOCR
    try:
        import easyocr
        logger.info("✓ EasyOCR 可用")
    except Exception as e:
        logger.warning(f"✗ EasyOCR 不可用: {e}")
    
    # 测试PaddleOCR
    try:
        from paddleocr import PaddleOCR
        logger.info("✓ PaddleOCR 可用")
    except Exception as e:
        logger.warning(f"✗ PaddleOCR 不可用: {e}")
    
    # 测试RapidOCR
    try:
        from rapidocr_onnxruntime import RapidOCR
        logger.info("✓ RapidOCR 可用")
    except Exception as e:
        logger.warning(f"✗ RapidOCR 不可用: {e}")

def setup_gpu_support():
    """设置GPU支持"""
    logger.info("检查GPU支持...")
    
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA可用，GPU数量: {torch.cuda.device_count()}")
            
            # 安装GPU版本的包
            logger.info("安装GPU版本的OCR包...")
            run_command(f"{sys.executable} -m pip install paddlepaddle-gpu", check=False)
            
        else:
            logger.info("CUDA不可用，使用CPU版本")
    except ImportError:
        logger.warning("PyTorch未安装，无法检查CUDA支持")

def create_test_script():
    """创建测试脚本"""
    test_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
OCR引擎测试脚本
\"\"\"

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ocr_engines import OCREngineManager

def main():
    print("测试OCR引擎管理器...")
    
    # 初始化OCR管理器
    manager = OCREngineManager(
        model_dir="./models",
        use_gpu=False
    )
    
    # 获取可用引擎
    engines = manager.get_available_engines()
    
    print("\\n可用的OCR引擎:")
    for name, info in engines.items():
        status = "✓" if info['available'] else "✗"
        print(f"{status} {info['name']}: {info['description']}")
        if not info['available'] and info['error_message']:
            print(f"  错误: {info['error_message']}")
    
    print("\\n测试完成！")

if __name__ == "__main__":
    main()
"""
    
    with open("test_ocr_engines.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    logger.info("已创建测试脚本: test_ocr_engines.py")

def main():
    """主函数"""
    logger.info("开始安装OCR引擎...")
    
    # 安装Tesseract
    install_tesseract()
    
    # 安装Python包
    install_python_packages()
    
    # 设置GPU支持
    setup_gpu_support()
    
    # 测试引擎
    test_ocr_engines()
    
    # 创建测试脚本
    create_test_script()
    
    logger.info("OCR引擎安装完成！")
    logger.info("运行 'python test_ocr_engines.py' 来测试所有引擎")

if __name__ == "__main__":
    main() 