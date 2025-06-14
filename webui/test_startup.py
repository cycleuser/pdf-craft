#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动脚本测试工具
用于验证不同启动脚本的功能是否正常
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def test_import():
    """测试关键模块导入"""
    print("🔍 测试模块导入...")
    
    modules = [
        ('flask', 'Flask'),
        ('pdf_craft', 'PDF-Craft'),
        ('optimization_config', '优化配置模块'),
        ('app', '应用模块')
    ]
    
    for module, name in modules:
        try:
            __import__(module)
            print(f"  ✅ {name} 导入成功")
        except ImportError as e:
            print(f"  ❌ {name} 导入失败: {e}")
            return False
        except Exception as e:
            print(f"  ⚠️  {name} 导入警告: {e}")
    
    return True

def test_config():
    """测试配置功能"""
    print("\n🔧 测试配置功能...")
    
    try:
        from optimization_config import auto_configure_optimization, get_preset_config, list_preset_configs
        
        # 测试自动配置
        optimal_config, recommendations = auto_configure_optimization()
        print(f"  ✅ 自动配置成功，性能等级: {recommendations['performance_tier']}")
        
        # 测试预设配置
        presets = list_preset_configs()
        print(f"  ✅ 预设配置加载成功，共 {len(presets)} 个配置")
        
        # 测试获取特定预设
        balanced = get_preset_config('balanced')
        print(f"  ✅ 平衡模式配置: {balanced['name']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 配置测试失败: {e}")
        return False

def test_app_config():
    """测试应用配置"""
    print("\n⚙️  测试应用配置...")
    
    try:
        from app import app, get_optimal_device_config
        
        # 测试设备配置
        device_config = get_optimal_device_config()
        print(f"  ✅ 设备配置: {device_config['device']}, 批处理大小: {device_config['batch_size']}")
        
        # 测试应用配置
        print(f"  📁 模型目录: {app.config['MODEL_DIR']}")
        print(f"  📁 上传目录: {app.config['UPLOAD_FOLDER']}")
        print(f"  📁 结果目录: {app.config['RESULTS_FOLDER']}")
        
        # 确保目录存在
        for dir_path in [app.config['MODEL_DIR'], app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
            os.makedirs(dir_path, exist_ok=True)
        
        print("  ✅ 目录配置正常")
        return True
        
    except Exception as e:
        print(f"  ❌ 应用配置测试失败: {e}")
        return False

def test_startup_script(script_name, port=5001):
    """测试启动脚本"""
    print(f"\n🚀 测试启动脚本: {script_name}")
    
    if not os.path.exists(script_name):
        print(f"  ❌ 脚本文件不存在: {script_name}")
        return False
    
    try:
        # 启动服务器
        cmd = [sys.executable, script_name, '--host', '127.0.0.1', '--port', str(port), '--no-auto-config']
        print(f"  🔄 启动命令: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待服务器启动
        print("  ⏳ 等待服务器启动...")
        time.sleep(5)
        
        # 检查服务器是否响应
        try:
            response = requests.get(f'http://127.0.0.1:{port}', timeout=5)
            if response.status_code == 200:
                print(f"  ✅ 服务器启动成功，响应状态: {response.status_code}")
                success = True
            else:
                print(f"  ⚠️  服务器响应异常，状态码: {response.status_code}")
                success = False
        except requests.exceptions.RequestException as e:
            print(f"  ❌ 服务器连接失败: {e}")
            success = False
        
        # 终止进程
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        
        return success
        
    except Exception as e:
        print(f"  ❌ 启动脚本测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 PDF OCR 启动脚本测试工具")
    print("=" * 60)
    
    # 测试模块导入
    if not test_import():
        print("\n❌ 模块导入测试失败，请检查依赖项")
        return False
    
    # 测试配置功能
    if not test_config():
        print("\n❌ 配置功能测试失败")
        return False
    
    # 测试应用配置
    if not test_app_config():
        print("\n❌ 应用配置测试失败")
        return False
    
    # 测试启动脚本
    scripts_to_test = [
        'run_improved.py',
        'run_optimized.py', 
        'run.py'
    ]
    
    port = 5001
    for script in scripts_to_test:
        if os.path.exists(script):
            test_startup_script(script, port)
            port += 1
        else:
            print(f"\n⚠️  跳过不存在的脚本: {script}")
    
    print("\n" + "=" * 60)
    print("🎉 测试完成！")
    print("=" * 60)
    
    print("\n💡 使用建议:")
    print("1. 推荐使用: python run_improved.py")
    print("2. 外网访问: python run_improved.py --host 0.0.0.0")
    print("3. 性能优化: python run_improved.py --preset balanced")
    print("4. 查看帮助: python run_improved.py --help")
    
    return True

if __name__ == '__main__':
    main() 