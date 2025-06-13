document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    const jobsTable = document.getElementById('jobs-table');
    const fileInput = document.getElementById('pdf-files');
    const toggleGpuBtn = document.getElementById('toggle-gpu');
    const refreshInfoBtn = document.getElementById('refresh-info');
    const setBatchSizeBtn = document.getElementById('set-batch-size');
    const batchSizeInput = document.getElementById('batch-size');
    const refreshJobsBtn = document.getElementById('refresh-jobs');
    const clearCompletedBtn = document.getElementById('clear-completed');

    // Initialize error modal
    const errorModal = new bootstrap.Modal(document.getElementById('errorModal'));
    const errorModalBody = document.getElementById('errorModalBody');

    // Initialize job details modal
    const jobDetailsModal = new bootstrap.Modal(document.getElementById('jobDetailsModal'));
    const jobDetailsModalBody = document.querySelector('.job-details-content');

    // Initialize download options modal
    const downloadOptionsModal = new bootstrap.Modal(document.getElementById('downloadOptionsModal'));
    let currentJobId = null;

    // 存储活跃的轮询间隔
    const activePolls = {};

    // Load system info on page load
    loadSystemInfo();

    // Refresh system info when button is clicked
    refreshInfoBtn.addEventListener('click', loadSystemInfo);

    // Refresh jobs when button is clicked
    refreshJobsBtn.addEventListener('click', loadJobs);

    // Clear completed jobs when button is clicked
    clearCompletedBtn.addEventListener('click', function() {
        const completedRows = document.querySelectorAll('tr[id^="job-"] .status-completed');
        if (completedRows.length === 0) {
            showMessage('没有已完成的作业可清除', 'info');
            return;
        }

        completedRows.forEach(statusBadge => {
            const row = statusBadge.closest('tr');
            if (row) {
                row.remove();
            }
        });

        showMessage(`已清除 ${completedRows.length} 个已完成的作业`, 'success');
    });

    // Set batch size when button is clicked
    setBatchSizeBtn.addEventListener('click', function() {
        const batchSize = parseInt(batchSizeInput.value);
        if (isNaN(batchSize) || batchSize < 1 || batchSize > 10) {
            showError('批处理大小必须是1到10之间的数字');
            return;
        }

        fetch('/batch_size', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ batch_size: batchSize })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                showMessage(`批处理大小已设置为 ${data.batch_size}`, 'success');
                loadSystemInfo(); // 刷新系统信息
            } else {
                showError(data.error || '设置批处理大小失败');
            }
        })
        .catch(error => {
            showError('设置批处理大小失败: ' + error.message);
        });
    });

    // Handle download option selection
    document.querySelectorAll('.download-option').forEach(button => {
        button.addEventListener('click', function() {
            const downloadType = this.getAttribute('data-type');
            if (currentJobId) {
                downloadOptionsModal.hide();
                window.location.href = `/download/${currentJobId}?type=${downloadType}`;
            }
        });
    });

    // Toggle GPU/CPU
    toggleGpuBtn.addEventListener('click', function() {
        fetch('/toggle_gpu', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            const useGpu = data.use_gpu;
            document.getElementById('current-device').textContent = useGpu ? 'GPU' : 'CPU';
            document.getElementById('current-device').className = `badge ${useGpu ? 'bg-success' : 'bg-primary'}`;
            toggleGpuBtn.textContent = `切换到 ${useGpu ? 'CPU' : 'GPU'}`;

            // Show success message
            showMessage(`已切换到 ${useGpu ? 'GPU' : 'CPU'} 模式`, 'success');
        })
        .catch(error => {
            showError('切换设备失败: ' + error.message);
        });
    });

    // Function to load system info
    function loadSystemInfo() {
        fetch('/system_info')
            .then(response => response.json())
            .then(data => {
                // Update device info
                const useGpu = data.use_gpu;
                document.getElementById('current-device').textContent = useGpu ? 'GPU' : 'CPU';
                document.getElementById('current-device').className = `badge ${useGpu ? 'bg-success' : 'bg-primary'}`;
                toggleGpuBtn.textContent = `切换到 ${useGpu ? 'CPU' : 'GPU'}`;

                // Update CUDA info
                const cudaAvailableEl = document.getElementById('cuda-available');
                cudaAvailableEl.textContent = data.cuda_available ? '是' : '否';
                cudaAvailableEl.className = data.cuda_available ? 'text-success' : 'text-danger';

                // Update GPU device info
                document.getElementById('cuda-device').textContent = data.cuda_device_name;

                // Update model info
                document.getElementById('model-dir').textContent = data.model_dir;

                // Update model files
                const modelFilesEl = document.getElementById('model-files');
                if (data.model_files && data.model_files.length > 0) {
                    modelFilesEl.innerHTML = data.model_files.map(file => `<div>${file}</div>`).join('');
                } else {
                    modelFilesEl.innerHTML = '<div class="text-warning">没有找到模型文件</div>';
                }

                // Update queue info
                const queueInfoEl = document.getElementById('queue-info');
                if (data.queue_info) {
                    queueInfoEl.innerHTML = `
                        <div>排队作业: ${data.queue_info.queued_jobs}</div>
                        <div>活跃作业: ${data.queue_info.active_jobs}</div>
                        <div>批处理大小: ${data.queue_info.batch_size}</div>
                    `;

                    // Update batch size input
                    batchSizeInput.value = data.queue_info.batch_size;
                } else {
                    queueInfoEl.innerHTML = '<div class="text-warning">无法获取队列信息</div>';
                }

                // Disable GPU button if CUDA is not available
                if (!data.cuda_available) {
                    toggleGpuBtn.disabled = true;
                    toggleGpuBtn.title = 'CUDA不可用，无法使用GPU加速';
                } else {
                    toggleGpuBtn.disabled = false;
                    toggleGpuBtn.title = '';
                }
            })
            .catch(error => {
                console.error('Error loading system info:', error);
                showError('加载系统信息失败: ' + error.message);
            });
    }

    // Function to show a message
    function showMessage(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.setAttribute('role', 'alert');
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;

        // Insert at the top of the container
        const container = document.querySelector('.container');
        container.insertBefore(alertDiv, container.firstChild);

        // Auto dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 150);
        }, 5000);
    }

    // Add drag and drop functionality
    uploadForm.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.add('drag-active');
    });

    uploadForm.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-active');
    });

    uploadForm.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-active');

        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            // Show file names in the input
            const fileList = Array.from(e.dataTransfer.files)
                .map(file => file.name)
                .join(', ');

            // Display file names or count
            if (fileList.length > 50) {
                fileInput.nextElementSibling.textContent = `${e.dataTransfer.files.length} files selected`;
            } else {
                fileInput.nextElementSibling.textContent = fileList;
            }
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const files = fileInput.files;
        if (files.length === 0) {
            showError('Please select at least one PDF file.');
            return;
        }

        // Check if all files are PDFs
        for (let i = 0; i < files.length; i++) {
            if (!files[i].name.toLowerCase().endsWith('.pdf')) {
                showError('Only PDF files are allowed.');
                return;
            }
        }

        // Create FormData and append files
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files[]', files[i]);
        }

        // Disable button and show loading state
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';

        // Send files to server
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Reset form
            uploadForm.reset();
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload and Convert';

            // Add new jobs to the table
            data.jobs.forEach(job => {
                addJobToTable(job);
            });

            // Show success message
            showMessage(`成功上传 ${data.jobs.length} 个文件并加入处理队列`, 'success');

            // Refresh system info to update queue status
            loadSystemInfo();
        })
        .catch(error => {
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload and Convert';
            showError('Error uploading files: ' + error.message);
        });
    });

    // Function to create a job row
    function createJobRow(job) {
        const row = document.createElement('tr');
        row.id = `job-${job.id || job.job_id}`;

        // 创建文件名单元格
        const fileNameCell = document.createElement('td');
        fileNameCell.textContent = job.filename;
        row.appendChild(fileNameCell);

        // 创建状态单元格
        const statusCell = document.createElement('td');
        const statusBadge = document.createElement('span');
        statusBadge.className = `badge status-badge status-${job.status || 'queued'}`;
        statusBadge.textContent = job.status ? job.status.charAt(0).toUpperCase() + job.status.slice(1) : 'Queued';
        statusCell.appendChild(statusBadge);
        row.appendChild(statusCell);

        // 创建进度单元格
        const progressCell = document.createElement('td');
        const progressDiv = document.createElement('div');
        progressDiv.className = 'progress';

        const percentage = job.progress ? job.progress.percentage : 0;
        const progressBar = document.createElement('div');
        progressBar.className = 'progress-bar';
        progressBar.role = 'progressbar';
        progressBar.style.width = `${percentage}%`;
        progressBar.setAttribute('aria-valuenow', percentage);
        progressBar.setAttribute('aria-valuemin', '0');
        progressBar.setAttribute('aria-valuemax', '100');
        progressBar.textContent = `${percentage}%`;

        progressDiv.appendChild(progressBar);
        progressCell.appendChild(progressDiv);
        row.appendChild(progressCell);

        // 创建操作单元格
        const actionsCell = document.createElement('td');
        const btnGroup = document.createElement('div');
        btnGroup.className = 'btn-group btn-group-sm';

        // 下载按钮
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'btn btn-sm btn-success btn-download';
        downloadBtn.textContent = '下载';
        downloadBtn.disabled = job.status !== 'completed';

        if (job.status === 'completed') {
            downloadBtn.addEventListener('click', function() {
                currentJobId = job.id || job.job_id;
                downloadOptionsModal.show();
            });
        }

        btnGroup.appendChild(downloadBtn);

        // 详情按钮
        const detailsBtn = document.createElement('button');
        detailsBtn.className = 'btn btn-sm btn-info btn-details';
        detailsBtn.textContent = '详情';
        detailsBtn.setAttribute('data-job-id', job.id || job.job_id);

        detailsBtn.addEventListener('click', function() {
            showJobDetails(this.getAttribute('data-job-id'));
        });

        btnGroup.appendChild(detailsBtn);
        actionsCell.appendChild(btnGroup);
        row.appendChild(actionsCell);

        return row;
    }

    // Function to add a job to the table
    function addJobToTable(job) {
        // 检查是否已存在此作业行
        const existingRow = document.getElementById(`job-${job.job_id}`);
        if (existingRow) {
            // 如果已存在，则更新
            const updatedRow = createJobRow(job);
            existingRow.replaceWith(updatedRow);
        } else {
            // 如果不存在，则添加新行
            const row = createJobRow(job);
            jobsTable.appendChild(row);
        }

        // Start polling for job status
        pollJobStatus(job.job_id);
    }

    // Function to show job details
    function showJobDetails(jobId) {
        fetch(`/job/${jobId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(job => {
                // Create job details HTML
                let detailsHtml = `
                    <div class="mb-3">
                        <strong>Job ID:</strong> ${job.id}
                    </div>
                    <div class="mb-3">
                        <strong>Filename:</strong> ${job.filename}
                    </div>
                    <div class="mb-3">
                        <strong>Status:</strong> <span class="badge bg-${getStatusColor(job.status)}">${job.status}</span>
                    </div>
                    <div class="mb-3">
                        <strong>Progress:</strong> ${job.progress ? job.progress.percentage : 0}%
                        (${job.progress ? job.progress.current_page : 0}/${job.progress ? job.progress.total_pages : 0} pages)
                    </div>
                `;

                // Add creation time
                if (job.created_at) {
                    const createdDate = new Date(job.created_at * 1000);
                    detailsHtml += `
                        <div class="mb-3">
                            <strong>创建时间:</strong> ${createdDate.toLocaleString()}
                        </div>
                    `;
                }

                // Add processing time if completed
                if (job.result && job.result.processing_time) {
                    detailsHtml += `
                        <div class="mb-3">
                            <strong>处理时间:</strong> ${job.result.processing_time.toFixed(2)} 秒
                        </div>
                    `;
                }

                // Add device info if available
                if (job.result && job.result.device_used) {
                    detailsHtml += `
                        <div class="mb-3">
                            <strong>Device Used:</strong> ${job.result.device_used}
                        </div>
                    `;
                }

                // Add model info if available
                if (job.result && job.result.model_dir) {
                    detailsHtml += `
                        <div class="mb-3">
                            <strong>Model Directory:</strong> ${job.result.model_dir}
                        </div>
                    `;
                }

                // Add error info if failed
                if (job.status === 'failed' && job.error) {
                    detailsHtml += `
                        <div class="alert alert-danger">
                            <strong>Error:</strong> ${job.error}
                        </div>
                    `;
                }

                // Update modal content
                jobDetailsModalBody.innerHTML = detailsHtml;

                // Show modal
                jobDetailsModal.show();
            })
            .catch(error => {
                console.error('Error getting job details:', error);
                showError('Error getting job details: ' + error.message);
            });
    }

    // Helper function to get status color
    function getStatusColor(status) {
        switch (status) {
            case 'queued': return 'secondary';
            case 'processing': return 'primary';
            case 'completed': return 'success';
            case 'failed': return 'danger';
            default: return 'secondary';
        }
    }

    // Function to poll job status
    function pollJobStatus(jobId) {
        // 如果已经在轮询这个作业，则不重复创建
        if (activePolls[jobId]) {
            return;
        }

        const statusCheck = setInterval(() => {
            fetch(`/job/${jobId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(job => {
                    // 更新作业行
                    const row = document.getElementById(`job-${job.id}`);
                    if (row) {
                        const updatedRow = createJobRow(job);
                        row.replaceWith(updatedRow);
                    }

                    // 如果作业完成或失败，停止轮询
                    if (job.status === 'completed' || job.status === 'failed') {
                        clearInterval(activePolls[jobId]);
                        delete activePolls[jobId];
                    }
                })
                .catch(error => {
                    console.error(`Error polling job status for job ${jobId}:`, error);
                    // 如果出错，也停止轮询
                    clearInterval(activePolls[jobId]);
                    delete activePolls[jobId];
                });
        }, 2000); // Poll every 2 seconds

        // 保存轮询间隔ID
        activePolls[jobId] = statusCheck;
    }

    // Function to show error modal
    function showError(message) {
        errorModalBody.textContent = message;
        errorModal.show();
    }

    // Function to load jobs
    function loadJobs() {
        fetch('/jobs')
            .then(response => response.json())
            .then(data => {
                try {
                    // Clear existing jobs
                    jobsTable.innerHTML = '';

                    // Add jobs to table
                    data.jobs.forEach(job => {
                        const jobRow = createJobRow(job);
                        jobsTable.appendChild(jobRow);

                        // 如果作业仍在进行中，启动轮询
                        if (job.status === 'queued' || job.status === 'processing') {
                            pollJobStatus(job.id);
                        }
                    });

                    showMessage(`已加载 ${data.jobs.length} 个作业`, 'info');
                } catch (err) {
                    console.error('Error rendering jobs:', err);
                    showError('渲染作业列表失败: ' + err.message);
                }
            })
            .catch(error => {
                console.error('Error loading jobs:', error);
                showError('加载作业失败: ' + error.message);
            });
    }

    // Load existing jobs on page load
    loadJobs();
});