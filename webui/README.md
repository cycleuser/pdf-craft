# PDF OCR 智能处理平台

基于光学字符识别（OCR）和本地大语言模型的PDF文档智能转换平台。

## 主要特性

- 🔧 **多OCR引擎支持**：集成PDF-Craft、Tesseract、EasyOCR、PaddleOCR、RapidOCR
- 🌍 **多语言识别**：支持中文、英文、日文、韩文等10+种语言
- ⚡ **高性能处理**：智能选择最优OCR引擎，大幅提升处理速度
- 📊 **性能对比**：实时对比各OCR引擎的处理速度和准确率
- 📄 **支持中文文件名**：完整保留原始文件名，包括中文字符
- 🚀 **多种运行模式**：兼容模式、平衡模式、性能模式可选
- 🌐 **支持外网访问**：默认监听所有网络接口
- 🎨 **优化的用户界面**：现代化的Bootstrap界面设计
- 📊 **实时处理监控**：显示处理进度和系统资源使用情况
- 💾 **多格式输出**：支持Markdown、Word、带文本PDF、完整压缩包

## 快速开始

### 1. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 自动安装OCR引擎（推荐）
python install_ocr_engines.py

# 或手动安装特定OCR引擎
pip install pytesseract easyocr paddleocr rapidocr-onnxruntime
```

### 2. 启动应用

```bash
python run.py
```

默认在 http://localhost:5000 启动服务。

### 3. 命令行选项

```bash
# 指定端口
python run.py --port 8080

# 启用调试模式
python run.py --debug

# 指定监听地址（默认0.0.0.0，支持外网访问）
python run.py --host 127.0.0.1
```

## 运行模式说明

应用支持三种运行模式，可在页面中随时切换：

### 兼容模式（默认）
- 最高稳定性，适合所有环境
- 禁用GPU加速和多进程处理
- 适合生产环境和资源受限的系统

### 平衡模式
- 平衡性能和稳定性
- 启用GPU加速（如果可用）
- 适合大多数用户

### 性能模式
- 最大化处理速度
- 启用所有优化功能
- 需要高性能硬件支持

## 功能特点

### 文件处理
- 支持批量上传PDF文件
- 保留原始中文文件名
- 实时显示处理进度
- 支持队列管理

### OCR配置
- **OCR引擎**：PDF-Craft、Tesseract、EasyOCR、PaddleOCR、RapidOCR
- **识别语言**：自动检测、中文、英文、日文、韩文等10+种语言
- **OCR级别**：快速、标准、精确、详细
- **表格提取**：无、简单、标准、高级
- **公式提取**：可选的数学公式识别
- **AI增强**：支持Ollama本地大语言模型
- **性能对比**：实时测试各引擎的处理速度和准确率

### 输出格式
- **Markdown**：带图片的标准Markdown文件
- **Word文档**：格式化的DOCX文件
- **带文本PDF**：可搜索的PDF文件
- **完整压缩包**：包含所有输出文件

## 系统要求

- Python 3.8+
- 4GB+ RAM（推荐8GB以上）
- 支持CUDA的GPU（可选，用于加速）

## OCR引擎说明

### PDF-Craft（默认）
- **特点**：支持复杂文档结构，表格和公式识别效果好
- **适用**：学术论文、技术文档、复杂排版文档
- **速度**：中等，但准确率高

### Tesseract
- **特点**：Google开源OCR，支持100+语言
- **适用**：多语言文档，简单文本识别
- **速度**：快速，轻量级

### EasyOCR
- **特点**：深度学习OCR，支持80+语言
- **适用**：手写文字、复杂背景文本
- **速度**：中等，GPU加速效果明显

### PaddleOCR
- **特点**：百度开源，中文识别效果优秀
- **适用**：中文文档，特别是简体中文
- **速度**：快速，支持GPU加速

### RapidOCR
- **特点**：轻量级高速OCR，基于ONNX Runtime
- **适用**：对速度要求高的场景
- **速度**：最快，资源占用少

## 常见问题

### 1. 如何选择最佳OCR引擎？
- 使用"性能对比"功能测试各引擎在您的文档上的表现
- 中文文档推荐：PaddleOCR > PDF-Craft > EasyOCR
- 英文文档推荐：Tesseract > RapidOCR > EasyOCR
- 复杂文档推荐：PDF-Craft > EasyOCR > PaddleOCR

### 2. OCR引擎安装失败
- 运行 `python install_ocr_engines.py` 自动安装
- 运行 `python test_ocr_engines.py` 测试引擎可用性
- 查看具体错误信息，手动安装对应依赖

### 3. 中文文件名显示问题
本应用已完全支持中文文件名，会在处理过程中保留原始文件名。

### 4. GPU加速不可用
- 确保安装了支持CUDA的PyTorch版本
- 检查NVIDIA驱动是否正确安装
- 在页面中查看系统信息确认GPU状态

### 5. 处理速度慢
- 使用性能对比功能选择最快的OCR引擎
- 尝试切换到性能模式
- 启用GPU加速
- 调整批处理大小

### 6. 内存不足
- 选择轻量级OCR引擎（RapidOCR、Tesseract）
- 切换到兼容模式
- 减小批处理大小
- 关闭其他占用内存的程序

## 技术架构

- **后端**：Flask + Python
- **前端**：Bootstrap + jQuery
- **OCR引擎**：PDF-Craft、Tesseract、EasyOCR、PaddleOCR、RapidOCR
- **文档处理**：PyMuPDF (fitz)、pdfplumber
- **图像处理**：OpenCV、Pillow、scikit-image
- **AI模型**：Ollama（可选）

## 更新日志

### v3.0.0 (2024)
- 🔧 **多OCR引擎支持**：集成5种主流OCR引擎
- 🌍 **多语言识别**：支持10+种语言的OCR识别
- ⚡ **性能大幅提升**：根据文档类型智能选择最优引擎
- 📊 **性能对比功能**：实时测试各引擎的处理速度和准确率
- 🛠️ **自动安装脚本**：一键安装所有OCR引擎依赖
- 🎯 **引擎推荐系统**：根据测试结果推荐最佳引擎

### v2.0.0 (2024)
- 统一运行入口，简化启动方式
- 完全支持中文文件名
- 改进的用户界面设计
- 添加运行模式选择功能
- 优化系统信息显示
- 增强错误处理和用户提示

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题或建议，请通过GitHub Issues联系。