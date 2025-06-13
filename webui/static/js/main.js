// PDF转Markdown工具 - 主要JavaScript文件
console.log('开始加载main.js...');

document.addEventListener('DOMContentLoaded', function() {
    console.log('页面DOM加载完成，开始初始化...');

    // 获取DOM元素
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('uploadButton');
    const dropArea = document.getElementById('drop-area');
    const jobsTableBody = document.getElementById('jobsTableBody');
    const useCustomConfig = document.getElementById('use-custom-config');
    const updateBatchSizeBtn = document.getElementById('update-batch-size');
    const batchSizeInput = document.getElementById('batch-size');
    const gpuToggle = document.getElementById('gpu-toggle');
    const deviceLabel = document.getElementById('device-label');
    const saveConfigBtn = document.getElementById('save-config');
    const resetConfigBtn = document.getElementById('reset-config');
    const ocrLevelSelect = document.getElementById('ocr-level');
    const tableFormatSelect = document.getElementById('table-format');
    const extractFormulaCheckbox = document.getElementById('extract-formula');
    const ollamaModelSelect = document.getElementById('ollama-model');
    const toggleConfigBtn = document.getElementById('toggle-config-panel');
    const configPanel = document.getElementById('config-panel');
    const showSystemInfoBtn = document.getElementById('show-system-info');
    const systemInfoModal = document.getElementById('systemInfoModal');
    const closeModalBtn = document.querySelector('.close');
    const clearJobsBtn = document.getElementById('clear-jobs');

    console.log('元素检查结果:');
    console.log('fileInput:', !!fileInput);
    console.log('uploadButton:', !!uploadButton);
    console.log('dropArea:', !!dropArea);
    console.log('toggleConfigBtn:', !!toggleConfigBtn);
    console.log('configPanel:', !!configPanel);
    console.log('showSystemInfoBtn:', !!showSystemInfoBtn);
    console.log('systemInfoModal:', !!systemInfoModal);
    console.log('saveConfigBtn:', !!saveConfigBtn);
    console.log('resetConfigBtn:', !!resetConfigBtn);

    // 拖放文件功能
    if (dropArea && fileInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, function(e) {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, function() {
                dropArea.classList.add('highlight');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, function() {
                dropArea.classList.remove('highlight');
            }, false);
        });

        dropArea.addEventListener('drop', function(e) {
            const dt = e.dataTransfer;
            const files = dt.files;

            const pdfFiles = Array.from(files).filter(file =>
                file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
            );

            if (pdfFiles.length === 0) {
                alert('请选择PDF文件');
                return;
            }

            const dataTransfer = new DataTransfer();
            pdfFiles.forEach(file => dataTransfer.items.add(file));
            fileInput.files = dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));
        }, false);

        // 文件选择变化
        fileInput.addEventListener('change', function() {
            const fileLabel = dropArea.querySelector('.file-label span');
            if (fileLabel) {
                if (this.files.length > 0) {
                    if (this.files.length === 1) {
                        fileLabel.textContent = this.files[0].name;
                    } else {
                        fileLabel.textContent = `已选择 ${this.files.length} 个文件`;
                    }
                } else {
                    fileLabel.textContent = '点击或拖放PDF文件到此处';
                }
            }
        });
    }

    // 上传按钮
    if (uploadButton && fileInput) {
        uploadButton.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('上传按钮被点击');

            if (!fileInput.files || fileInput.files.length === 0) {
                alert('请先选择PDF文件');
                return;
            }

            const formData = new FormData();
            for (let i = 0; i < fileInput.files.length; i++) {
                formData.append('files[]', fileInput.files[i]);
            }

            if (useCustomConfig && useCustomConfig.checked) {
                formData.append('use_custom_config', 'true');
            }

            uploadButton.disabled = true;
            uploadButton.textContent = '上传中...';

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(`成功上传 ${data.jobs ? data.jobs.length : 1} 个文件`);
                    fileInput.value = '';
                    const fileLabel = dropArea.querySelector('.file-label span');
                    if (fileLabel) {
                        fileLabel.textContent = '点击或拖放PDF文件到此处';
                    }
                    loadJobs();
                } else {
                    alert(`上传失败: ${data.error || '未知错误'}`);
                }
            })
            .catch(error => {
                console.error('上传错误:', error);
                alert(`上传出错: ${error.message}`);
            })
            .finally(() => {
                uploadButton.disabled = false;
                uploadButton.textContent = '开始上传';
            });
        });
    }

    // 模型配置面板切换
    if (toggleConfigBtn && configPanel) {
        console.log('配置面板切换功能已启用');
        toggleConfigBtn.addEventListener('click', function() {
            console.log('配置面板切换按钮被点击');
            if (configPanel.style.display === 'none' || configPanel.style.display === '') {
                configPanel.style.display = 'block';
                console.log('配置面板已显示');
            } else {
                configPanel.style.display = 'none';
                console.log('配置面板已隐藏');
            }
        });
    } else {
        console.error('配置面板元素未找到!');
    }

    // 系统信息模态框
    if (showSystemInfoBtn && systemInfoModal) {
        console.log('系统信息功能已启用');
        showSystemInfoBtn.addEventListener('click', function() {
            console.log('系统信息按钮被点击');
            systemInfoModal.style.display = 'block';
            loadSystemInfo();
        });

        if (closeModalBtn) {
            closeModalBtn.addEventListener('click', function() {
                systemInfoModal.style.display = 'none';
            });
        }

        window.addEventListener('click', function(event) {
            if (event.target === systemInfoModal) {
                systemInfoModal.style.display = 'none';
            }
        });
    } else {
        console.error('系统信息元素未找到!');
    }

    // GPU切换
    if (gpuToggle && deviceLabel) {
        gpuToggle.addEventListener('change', function() {
            fetch('/toggle_gpu', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(response => response.json())
            .then(data => {
                deviceLabel.textContent = data.use_gpu ? 'GPU' : 'CPU';
                alert(`已切换到${data.use_gpu ? 'GPU' : 'CPU'}模式`);
            })
            .catch(error => {
                console.error('GPU切换错误:', error);
                gpuToggle.checked = !gpuToggle.checked;
            });
        });
    }

    // 批处理大小更新
    if (updateBatchSizeBtn && batchSizeInput) {
        updateBatchSizeBtn.addEventListener('click', function() {
            const batchSize = parseInt(batchSizeInput.value);
            if (isNaN(batchSize) || batchSize < 1 || batchSize > 10) {
                alert('批处理大小必须是1到10之间的数字');
                return;
            }

            fetch('/batch_size', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ batch_size: batchSize })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(`批处理大小已设置为 ${data.batch_size}`);
                } else {
                    alert(`设置失败: ${data.error}`);
                }
            })
            .catch(error => {
                alert(`设置出错: ${error.message}`);
            });
        });
    }

    // 保存配置
    if (saveConfigBtn) {
        console.log('保存配置功能已启用');
        saveConfigBtn.addEventListener('click', function() {
            console.log('保存配置按钮被点击');
            const config = {};

            if (ocrLevelSelect) config.ocr_level = ocrLevelSelect.value;
            if (tableFormatSelect) config.extract_table_format = tableFormatSelect.value;
            if (extractFormulaCheckbox) config.extract_formula = extractFormulaCheckbox.checked;
            if (ollamaModelSelect) config.ollama_model = ollamaModelSelect.value;

            fetch('/update_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('配置已保存');
                } else {
                    alert(`保存失败: ${data.error}`);
                }
            })
            .catch(error => {
                alert(`保存出错: ${error.message}`);
            });
        });
    } else {
        console.error('保存配置按钮未找到!');
    }

    // 重置配置
    if (resetConfigBtn) {
        console.log('重置配置功能已启用');
        resetConfigBtn.addEventListener('click', function() {
            fetch('/reset_config', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    if (ocrLevelSelect && data.config.ocr_level) ocrLevelSelect.value = data.config.ocr_level;
                    if (tableFormatSelect && data.config.extract_table_format) tableFormatSelect.value = data.config.extract_table_format;
                    if (extractFormulaCheckbox) extractFormulaCheckbox.checked = data.config.extract_formula || false;
                    if (ollamaModelSelect && data.config.ollama_model) ollamaModelSelect.value = data.config.ollama_model;
                    alert('配置已重置为默认值');
                } else {
                    alert(`重置失败: ${data.error}`);
                }
            })
            .catch(error => {
                alert(`重置出错: ${error.message}`);
            });
        });
    } else {
        console.error('重置配置按钮未找到!');
    }

    // 清空所有任务
    if (clearJobsBtn) {
        clearJobsBtn.addEventListener('click', function() {
            if (confirm('确定要清空所有任务和相关文件吗？此操作不可撤销！')) {
                clearJobsBtn.disabled = true;
                clearJobsBtn.textContent = '清空中...';

                fetch('/clear_all_jobs', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('所有任务和文件已清空');
                        loadJobs();
                    } else {
                        alert(`清空失败: ${data.error}`);
                    }
                })
                .catch(error => {
                    alert(`清空出错: ${error.message}`);
                })
                .finally(() => {
                    clearJobsBtn.disabled = false;
                    clearJobsBtn.textContent = '清空所有任务';
                });
            }
        });
    }

    // 加载系统信息
    function loadSystemInfo() {
        fetch('/system_info')
            .then(response => response.json())
            .then(data => {
                const systemInfoContent = document.getElementById('systemInfoContent');
                if (systemInfoContent) {
                    systemInfoContent.innerHTML = `
                        <div class="info-section">
                            <h3>系统信息</h3>
                            <p><strong>操作系统:</strong> ${data.os_info || '未知'}</p>
                            <p><strong>Python版本:</strong> ${data.python_version || '未知'}</p>
                        </div>
                        <div class="info-section">
                            <h3>GPU信息</h3>
                            <p><strong>CUDA可用:</strong> ${data.cuda_available ? '是' : '否'}</p>
                            <p><strong>设备名称:</strong> ${data.cuda_device_name || '无'}</p>
                            <p><strong>当前使用:</strong> ${data.use_gpu ? 'GPU' : 'CPU'}</p>
                        </div>
                        <div class="info-section">
                            <h3>队列信息</h3>
                            <p><strong>排队作业:</strong> ${data.queue_info ? data.queue_info.queued_jobs : '0'}</p>
                            <p><strong>活跃作业:</strong> ${data.queue_info ? data.queue_info.active_jobs : '0'}</p>
                            <p><strong>批处理大小:</strong> ${data.queue_info ? data.queue_info.batch_size : '5'}</p>
                        </div>
                        <div class="info-section">
                            <h3>Ollama状态</h3>
                            <p><strong>服务状态:</strong> ${data.ollama_available ? '可用' : '不可用'}</p>
                            <p><strong>可用模型:</strong> ${data.ollama_models ? data.ollama_models.length : 0} 个</p>
                        </div>
                    `;
                }
            })
            .catch(error => {
                console.error('加载系统信息失败:', error);
                const systemInfoContent = document.getElementById('systemInfoContent');
                if (systemInfoContent) {
                    systemInfoContent.innerHTML = '<p style="color: red;">加载系统信息失败</p>';
                }
            });
    }

    // 加载作业列表
    function loadJobs() {
        fetch('/jobs')
            .then(response => response.json())
            .then(data => {
                if (data.jobs && jobsTableBody) {
                    jobsTableBody.innerHTML = '';

                    if (data.jobs.length === 0) {
                        jobsTableBody.innerHTML = '<tr><td colspan="4" style="text-align: center;">暂无任务</td></tr>';
                        return;
                    }

                    data.jobs.forEach(job => {
                        const row = document.createElement('tr');
                        const progress = job.progress ? job.progress.percentage || 0 : 0;

                        row.innerHTML = `
                            <td>${job.filename || '未知文件'}</td>
                            <td><span class="status-badge status-${job.status}">${getStatusText(job.status)}</span></td>
                            <td>
                                <div class="progress">
                                    <div class="progress-bar" style="width: ${progress}%"></div>
                                </div>
                                <span>${progress}%</span>
                            </td>
                            <td>
                                                    <div class="download-dropdown" ${job.status !== 'completed' ? 'style="display:none"' : ''}>
                        <button class="action-button download-button" onclick="toggleDownloadMenu('${job.id}')" ${job.status !== 'completed' ? 'disabled' : ''} title="下载选项">
                            <i class="fas fa-download"></i>
                            <i class="fas fa-chevron-down"></i>
                        </button>
                        <div class="download-menu" id="download-menu-${job.id}" style="display: none;">
                            <a href="#" onclick="downloadJob('${job.id}', 'markdown')" class="download-option">
                                <i class="fab fa-markdown"></i> Markdown
                            </a>
                            <a href="#" onclick="downloadJob('${job.id}', 'word')" class="download-option">
                                <i class="fas fa-file-word"></i> Word文档
                            </a>
                            <a href="#" onclick="downloadJob('${job.id}', 'pdf')" class="download-option">
                                <i class="fas fa-file-pdf"></i> 带文本PDF
                            </a>
                            <a href="#" onclick="downloadJob('${job.id}', 'zip')" class="download-option">
                                <i class="fas fa-file-archive"></i> 完整压缩包
                            </a>
                        </div>
                    </div>
                                <button class="action-button delete-button" onclick="deleteJob('${job.id}')" title="删除任务">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </td>
                        `;
                        jobsTableBody.appendChild(row);
                    });
                }
            })
            .catch(error => {
                console.error('加载作业失败:', error);
                if (jobsTableBody) {
                    jobsTableBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: red;">加载失败</td></tr>';
                }
            });
    }

    // 状态文本转换
    function getStatusText(status) {
        const statusMap = {
            'queued': '排队中',
            'processing': '处理中',
            'completed': '已完成',
            'failed': '失败'
        };
        return statusMap[status] || status;
    }

    // 全局函数
    // 切换下载菜单显示
    window.toggleDownloadMenu = function(jobId) {
        const menu = document.getElementById(`download-menu-${jobId}`);
        const isVisible = menu.style.display !== 'none';

        // 隐藏所有其他下载菜单
        document.querySelectorAll('.download-menu').forEach(m => {
            m.style.display = 'none';
        });

        // 切换当前菜单
        menu.style.display = isVisible ? 'none' : 'block';
    };

    // 下载文件函数
    window.downloadJob = function(jobId, type = 'markdown') {
        // 隐藏下载菜单
        const menu = document.getElementById(`download-menu-${jobId}`);
        if (menu) {
            menu.style.display = 'none';
        }

        // 根据类型构建下载URL
        let downloadUrl = `/download/${jobId}`;
        if (type !== 'markdown') {
            downloadUrl += `?type=${type}`;
        }

        window.location.href = downloadUrl;
    };

    // 点击其他地方时隐藏下载菜单
    document.addEventListener('click', function(event) {
        if (!event.target.closest('.download-dropdown')) {
            document.querySelectorAll('.download-menu').forEach(menu => {
                menu.style.display = 'none';
            });
        }
    });

    window.deleteJob = function(jobId) {
        if (confirm('确定要删除此作业吗？')) {
            fetch(`/job/${jobId}`, { method: 'DELETE' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('作业已删除');
                        loadJobs();
                    } else {
                        alert(`删除失败: ${data.error}`);
                    }
                })
                .catch(error => {
                    alert(`删除出错: ${error.message}`);
                });
        }
    };

    // 初始加载
    loadJobs();

    console.log('JavaScript初始化完成');
});