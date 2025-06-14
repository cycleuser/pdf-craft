#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR引擎测试脚本
"""

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
    
    print("\n可用的OCR引擎:")
    for name, info in engines.items():
        status = "✓" if info['available'] else "✗"
        print(f"{status} {info['name']}: {info['description']}")
        if not info['available'] and info['error_message']:
            print(f"  错误: {info['error_message']}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()
