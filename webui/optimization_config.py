# -*- coding: utf-8 -*-
"""
OCR性能优化配置模块
自动检测系统配置并推荐最优设置
"""

import os
import logging
import psutil
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class OptimizationConfigManager:
    """优化配置管理器"""
    
    def __init__(self):
        self.system_info = self._detect_system_info()
        self.optimal_config = self._calculate_optimal_config()
    
    def _detect_system_info(self) -> Dict[str, Any]:
        """检测系统信息"""
        info = {
            'cpu_count': os.cpu_count() or 1,
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'memory_available_gb': psutil.virtual_memory().available / 1024**3,
            'has_gpu': False,
            'gpu_count': 0,
            'gpu_memory_gb': 0,
            'gpu_names': []
        }
        
        # 检测GPU信息
        try:
            import torch
            if torch.cuda.is_available():
                info['has_gpu'] = True
                info['gpu_count'] = torch.cuda.device_count()
                
                for i in range(info['gpu_count']):
                    props = torch.cuda.get_device_properties(i)
                    info['gpu_names'].append(props.name)
                    if i == 0:  # 使用第一个GPU的内存作为参考
                        info['gpu_memory_gb'] = props.total_memory / 1024**3
                        
        except ImportError:
            logger.warning("PyTorch未安装，无法检测GPU信息")
        
        return info
    
    def _calculate_optimal_config(self) -> Dict[str, Any]:
        """计算最优配置"""
        config = {
            'device': 'cpu',
            'batch_size': 2,
            'enable_multiprocessing': False,
            'process_pool_size': 2,
            'max_workers': 4,
            'enable_mixed_precision': False,
            'optimize_memory': True,
            'preload_models': False,
            'gpu_batch_size': 4,
            'cpu_batch_size': 2
        }
        
        # 根据系统配置调整
        if self.system_info['has_gpu']:
            config['device'] = 'cuda'
            config['enable_mixed_precision'] = True
            
            # 根据GPU内存调整批处理大小
            gpu_memory = self.system_info['gpu_memory_gb']
            if gpu_memory >= 24:  # 高端GPU (RTX 4090, A100等)
                config['gpu_batch_size'] = 16
                config['batch_size'] = 16
                config['preload_models'] = True
            elif gpu_memory >= 16:  # 中高端GPU (RTX 4080, RTX 3080Ti等)
                config['gpu_batch_size'] = 12
                config['batch_size'] = 12
                config['preload_models'] = True
            elif gpu_memory >= 12:  # 中端GPU (RTX 4070Ti, RTX 3080等)
                config['gpu_batch_size'] = 8
                config['batch_size'] = 8
                config['preload_models'] = True
            elif gpu_memory >= 8:   # 入门级GPU (RTX 4060Ti, RTX 3070等)
                config['gpu_batch_size'] = 6
                config['batch_size'] = 6
            elif gpu_memory >= 6:   # 低端GPU (RTX 4060, GTX 1660等)
                config['gpu_batch_size'] = 4
                config['batch_size'] = 4
            else:  # 极低端GPU
                config['gpu_batch_size'] = 2
                config['batch_size'] = 2
        
        # 根据CPU和内存调整
        cpu_count = self.system_info['cpu_count']
        memory_gb = self.system_info['memory_total_gb']
        
        # 多进程配置
        if cpu_count >= 8 and memory_gb >= 16:
            config['enable_multiprocessing'] = True
            config['process_pool_size'] = min(4, cpu_count // 2)
            config['max_workers'] = min(32, cpu_count + 4)
        elif cpu_count >= 4 and memory_gb >= 8:
            config['enable_multiprocessing'] = True
            config['process_pool_size'] = min(2, cpu_count // 2)
            config['max_workers'] = min(16, cpu_count + 2)
        
        # CPU批处理大小
        if cpu_count >= 16:
            config['cpu_batch_size'] = 8
        elif cpu_count >= 8:
            config['cpu_batch_size'] = 4
        elif cpu_count >= 4:
            config['cpu_batch_size'] = 2
        else:
            config['cpu_batch_size'] = 1
        
        # 内存优化
        if memory_gb < 8:
            config['optimize_memory'] = True
            config['preload_models'] = False
        elif memory_gb >= 32:
            config['preload_models'] = True
        
        return config
    
    def get_performance_tier(self) -> str:
        """获取性能等级"""
        if self.system_info['has_gpu']:
            gpu_memory = self.system_info['gpu_memory_gb']
            if gpu_memory >= 16:
                return "高性能"
            elif gpu_memory >= 8:
                return "中高性能"
            elif gpu_memory >= 6:
                return "中等性能"
            else:
                return "入门级"
        else:
            cpu_count = self.system_info['cpu_count']
            memory_gb = self.system_info['memory_total_gb']
            
            if cpu_count >= 16 and memory_gb >= 32:
                return "高性能CPU"
            elif cpu_count >= 8 and memory_gb >= 16:
                return "中高性能CPU"
            elif cpu_count >= 4 and memory_gb >= 8:
                return "中等性能CPU"
            else:
                return "入门级CPU"
    
    def get_recommendations(self) -> Dict[str, str]:
        """获取优化建议"""
        recommendations = []
        
        # GPU相关建议
        if not self.system_info['has_gpu']:
            recommendations.append("建议安装支持CUDA的GPU以大幅提升OCR处理速度")
        elif self.system_info['gpu_memory_gb'] < 6:
            recommendations.append("GPU显存较小，建议升级到8GB以上显存的GPU")
        
        # 内存相关建议
        if self.system_info['memory_total_gb'] < 8:
            recommendations.append("系统内存不足8GB，建议升级内存以提升处理大文档的能力")
        elif self.system_info['memory_total_gb'] < 16:
            recommendations.append("建议升级到16GB以上内存以启用更多优化功能")
        
        # CPU相关建议
        if self.system_info['cpu_count'] < 4:
            recommendations.append("CPU核心数较少，建议升级到4核以上CPU以支持并行处理")
        
        # 软件配置建议
        if self.system_info['has_gpu']:
            recommendations.append("已检测到GPU，建议启用混合精度训练以提升速度")
            recommendations.append("建议启用模型预加载以减少重复加载时间")
        
        if self.system_info['cpu_count'] >= 4:
            recommendations.append("建议启用多进程处理以充分利用CPU资源")
        
        return {
            'performance_tier': self.get_performance_tier(),
            'recommendations': recommendations,
            'estimated_speedup': self._estimate_speedup()
        }
    
    def _estimate_speedup(self) -> str:
        """估算性能提升"""
        base_speed = 1.0
        
        if self.system_info['has_gpu']:
            gpu_memory = self.system_info['gpu_memory_gb']
            if gpu_memory >= 16:
                base_speed *= 8.0  # 高端GPU
            elif gpu_memory >= 8:
                base_speed *= 5.0  # 中端GPU
            else:
                base_speed *= 3.0  # 入门GPU
        
        if self.optimal_config['enable_multiprocessing']:
            base_speed *= min(2.0, self.system_info['cpu_count'] / 4)
        
        if self.optimal_config['enable_mixed_precision']:
            base_speed *= 1.3
        
        if base_speed >= 8:
            return "8-12倍"
        elif base_speed >= 5:
            return "5-8倍"
        elif base_speed >= 3:
            return "3-5倍"
        elif base_speed >= 2:
            return "2-3倍"
        else:
            return "1.5-2倍"
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置"""
        return {
            'system_info': self.system_info,
            'optimal_config': self.optimal_config,
            'recommendations': self.get_recommendations()
        }

def auto_configure_optimization() -> Tuple[Dict[str, Any], Dict[str, str]]:
    """自动配置优化设置"""
    manager = OptimizationConfigManager()
    return manager.optimal_config, manager.get_recommendations()

def get_system_performance_report() -> Dict[str, Any]:
    """获取系统性能报告"""
    manager = OptimizationConfigManager()
    return manager.export_config()

# 预设配置模板
PRESET_CONFIGS = {
    'maximum_speed': {
        'name': '最大速度',
        'description': '牺牲一些准确性换取最快处理速度',
        'config': {
            'enable_multiprocessing': True,
            'enable_mixed_precision': True,
            'optimize_memory': True,
            'preload_models': True,
            'ocr_level': 'fast',
            'extract_table_format': 'simple'
        }
    },
    'balanced': {
        'name': '平衡模式',
        'description': '在速度和准确性之间取得平衡',
        'config': {
            'enable_multiprocessing': True,
            'enable_mixed_precision': True,
            'optimize_memory': True,
            'preload_models': True,
            'ocr_level': 'standard',
            'extract_table_format': 'standard'
        }
    },
    'maximum_quality': {
        'name': '最高质量',
        'description': '优先保证处理质量，速度较慢',
        'config': {
            'enable_multiprocessing': False,
            'enable_mixed_precision': False,
            'optimize_memory': False,
            'preload_models': False,
            'ocr_level': 'detailed',
            'extract_table_format': 'advanced'
        }
    },
    'low_resource': {
        'name': '低资源模式',
        'description': '适合内存和CPU资源有限的设备',
        'config': {
            'enable_multiprocessing': False,
            'enable_mixed_precision': False,
            'optimize_memory': True,
            'preload_models': False,
            'batch_size': 1,
            'process_pool_size': 1
        }
    }
}

def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """获取预设配置"""
    return PRESET_CONFIGS.get(preset_name, PRESET_CONFIGS['balanced'])

def list_preset_configs() -> Dict[str, Dict[str, str]]:
    """列出所有预设配置"""
    return {name: {'name': config['name'], 'description': config['description']} 
            for name, config in PRESET_CONFIGS.items()} 