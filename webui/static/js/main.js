document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    const jobsTable = document.getElementById('jobs-table');
    const fileInput = document.getElementById('pdf-files');
    const toggleGpuBtn = document.getElementById('toggle-gpu');
    const refreshInfoBtn = document.getElementById('refresh-info');

    // Initialize error modal
    const errorModal = new bootstrap.Modal(document.getElementById('errorModal'));
    const errorModalBody = document.getElementById('errorModalBody');

    // Initialize job details modal
    const jobDetailsModal = new bootstrap.Modal(document.getElementById('jobDetailsModal'));
    const jobDetailsModalBody = document.querySelector('.job-details-content');

    // Load system info on page load
    loadSystemInfo();

    // Refresh system info when button is clicked
    refreshInfoBtn.addEventListener('click', loadSystemInfo);

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
        })
        .catch(error => {
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload and Convert';
            showError('Error uploading files: ' + error.message);
        });
    });

    // Function to add a job to the table
    function addJobToTable(job) {
        const row = document.createElement('tr');
        row.id = `job-${job.job_id}`;
        row.innerHTML = `
            <td>${job.filename}</td>
            <td><span class="badge status-badge status-queued">Queued</span></td>
            <td>
                <div class="progress">
                    <div class="progress-bar" role="progressbar" style="width: 0%;"
                         aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>
                </div>
            </td>
            <td>
                <button class="btn btn-sm btn-success btn-download" disabled>Download</button>
                <button class="btn btn-sm btn-info btn-details" data-job-id="${job.job_id}">Details</button>
            </td>
        `;

        // Add event listener to details button
        row.querySelector('.btn-details').addEventListener('click', function() {
            const jobId = this.getAttribute('data-job-id');
            showJobDetails(jobId);
        });

        jobsTable.appendChild(row);

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
        const statusCheck = setInterval(() => {
            fetch(`/job/${jobId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(job => {
                    updateJobInTable(job);

                    // If job is completed or failed, stop polling
                    if (job.status === 'completed' || job.status === 'failed') {
                        clearInterval(statusCheck);
                    }
                })
                .catch(error => {
                    console.error('Error polling job status:', error);
                });
        }, 2000); // Poll every 2 seconds
    }

    // Function to update job in table
    function updateJobInTable(job) {
        const row = document.getElementById(`job-${job.id}`);
        if (!row) return;

        // Update status
        const statusCell = row.querySelector('td:nth-child(2)');
        statusCell.innerHTML = `<span class="badge status-badge status-${job.status}">${job.status.charAt(0).toUpperCase() + job.status.slice(1)}</span>`;

        // Update progress bar
        const progressBar = row.querySelector('.progress-bar');
        const percentage = job.progress ? job.progress.percentage : 0;
        progressBar.style.width = `${percentage}%`;
        progressBar.setAttribute('aria-valuenow', percentage);
        progressBar.textContent = `${percentage}%`;

        // Enable/disable download button
        const downloadBtn = row.querySelector('.btn-download');
        if (job.status === 'completed') {
            downloadBtn.disabled = false;
            downloadBtn.addEventListener('click', () => {
                window.location.href = `/download/${job.id}`;
            });
        } else {
            downloadBtn.disabled = true;
        }

        // Show error if job failed
        if (job.status === 'failed' && job.error) {
            const errorIcon = document.createElement('i');
            errorIcon.className = 'bi bi-exclamation-circle text-danger ms-2';
            errorIcon.title = job.error;
            statusCell.appendChild(errorIcon);
        }
    }

    // Function to show error modal
    function showError(message) {
        errorModalBody.textContent = message;
        errorModal.show();
    }

    // Load existing jobs on page load
    fetch('/jobs')
        .then(response => response.json())
        .then(data => {
            data.jobs.forEach(job => {
                const jobRow = document.createElement('tr');
                jobRow.id = `job-${job.id}`;
                jobsTable.appendChild(jobRow);
                updateJobInTable(job);
            });
        })
        .catch(error => {
            console.error('Error loading jobs:', error);
        });
});