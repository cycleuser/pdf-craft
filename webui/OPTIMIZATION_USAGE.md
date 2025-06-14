# OCR性能优化使用说明

## 🚀 快速开始

### 方法1：使用优化启动脚本（推荐）

```bash
# 自动检测并优化配置
python run_optimized.py

# 使用预设配置
python run_optimized.py --preset balanced

# 指定端口和主机
python run_optimized.py --host 0.0.0.0 --port 8080
```

### 方法2：Web界面一键优化

1. 启动应用：`python app.py`
2. 访问：http://localhost:5000
3. 点击"系统信息"按钮
4. 点击"自动优化配置"按钮

## 📊 预设配置模式

### 最大速度模式

```bash
python run_optimized.py --preset maximum_speed
```

- 适合：追求最快处理速度
- 特点：启用所有优化，可能牺牲少量质量

### 平衡模式（推荐）

```bash
python run_optimized.py --preset balanced
```

- 适合：大多数用户
- 特点：在速度和质量之间取得最佳平衡

### 最高质量模式

```bash
python run_optimized.py --preset maximum_quality
```

- 适合：追求最佳OCR质量
- 特点：优先保证质量，处理速度较慢

### 低资源模式

```bash
python run_optimized.py --preset low_resource
```

- 适合：内存和CPU资源有限的设备
- 特点：最小化资源占用

## ⚙️ 手动配置优化

### 通过Web API

```bash
# 更新优化设置
curl -X POST http://localhost:5000/optimization_settings \
  -H "Content-Type: application/json" \
  -d '{
    "enable_multiprocessing": true,
    "enable_mixed_precision": true,
    "gpu_batch_size": 8
  }'

# 应用预设配置
curl -X POST http://localhost:5000/apply_preset/balanced

# 自动优化
curl -X POST http://localhost:5000/auto_optimize
```

### 通过配置文件

编辑 `app.py` 中的配置：

```python
# GPU配置
app.config['USE_GPU'] = True
app.config['GPU_BATCH_SIZE'] = 8

# 多进程配置
app.config['ENABLE_MULTIPROCESSING'] = True
app.config['PROCESS_POOL_SIZE'] = 4

# 优化选项
app.config['ENABLE_MIXED_PRECISION'] = True
app.config['OPTIMIZE_MEMORY'] = True
app.config['PRELOAD_MODELS'] = True
```

## 📈 性能监控

### 查看系统信息

```bash
curl http://localhost:5000/system_info
```

### 查看性能统计

```bash
curl http://localhost:5000/performance_stats
```

### 获取性能报告

```bash
curl http://localhost:5000/system_performance_report
```

## 🔧 常用命令

### 启动选项

```bash
# 基本启动
python run_optimized.py

# 调试模式
python run_optimized.py --debug

# 禁用自动配置
python run_optimized.py --no-auto-config

# 指定网络配置
python run_optimized.py --host 0.0.0.0 --port 8080
```

### 管理命令

```bash
# 清空所有任务
curl -X POST http://localhost:5000/clear_all_jobs

# 清空模型缓存
curl -X POST http://localhost:5000/clear_model_cache

# 切换GPU模式
curl -X POST http://localhost:5000/toggle_gpu
```

## 💡 优化建议

### 根据硬件选择配置

- **有GPU**：使用 `maximum_speed` 或 `balanced` 模式
- **仅CPU**：使用 `balanced` 或 `low_resource` 模式
- **内存不足**：使用 `low_resource` 模式

### 根据使用场景选择

- **大批量处理**：启用多进程，增大批处理大小
- **实时处理**：启用模型预加载，使用GPU
- **高质量要求**：使用 `maximum_quality` 模式

## 🚨 故障排除

### 常见问题

1. **GPU内存不足**：减小批处理大小
2. **CPU占用过高**：减少进程数
3. **内存泄漏**：禁用模型预加载

### 重置配置

```bash
# 重置为默认配置
curl -X POST http://localhost:5000/reset_config

# 使用默认启动
python app.py
```

## 📞 获取帮助

```bash
# 查看启动脚本帮助
python run_optimized.py --help

# 查看详细优化指南
cat PERFORMANCE_OPTIMIZATION_GUIDE.md
```
