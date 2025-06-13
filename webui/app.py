import os
import uuid
import json
import threading
import logging
import pickle
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from pdf_craft import PDFPageExtractor, MarkDownWriter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
app.config['MODEL_DIR'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
app.config['JOBS_FILE'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jobs.pkl')
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['USE_GPU'] = False  # 默认使用CPU，可以在这里修改为True来使用GPU

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)

# Store job status
jobs = {}

# 从文件加载作业状态
def load_jobs():
    global jobs
    if os.path.exists(app.config['JOBS_FILE']):
        try:
            with open(app.config['JOBS_FILE'], 'rb') as f:
                jobs = pickle.load(f)
            logger.info(f"已从文件加载 {len(jobs)} 个作业状态")

            # 检查文件是否存在
            for job_id, job in list(jobs.items()):
                if 'file_path' in job and not os.path.exists(job['file_path']):
                    logger.warning(f"作业 {job_id} 的文件不存在，移除该作业")
                    del jobs[job_id]

                if job['status'] == 'processing':
                    logger.warning(f"作业 {job_id} 处于处理中状态，重置为失败状态")
                    job['status'] = 'failed'
                    job['error'] = '服务器重启，处理被中断'
        except Exception as e:
            logger.error(f"加载作业状态失败: {str(e)}")
            jobs = {}

# 保存作业状态到文件
def save_jobs():
    try:
        with open(app.config['JOBS_FILE'], 'wb') as f:
            pickle.dump(jobs, f)
        logger.info(f"已保存 {len(jobs)} 个作业状态到文件")
    except Exception as e:
        logger.error(f"保存作业状态失败: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class ProgressReporter:
    def __init__(self, job_id):
        self.job_id = job_id
        self.current_page = 0
        self.total_pages = 0

    def report(self, current_page, total_pages):
        self.current_page = current_page
        self.total_pages = total_pages
        jobs[self.job_id]['progress'] = {
            'current_page': current_page,
            'total_pages': total_pages,
            'percentage': int((current_page / total_pages) * 100) if total_pages > 0 else 0
        }
        logger.info(f"Job {self.job_id}: Processing page {current_page}/{total_pages}")
        # 每次更新进度时保存作业状态
        save_jobs()

def process_pdf(job_id, pdf_path, output_dir):
    try:
        # Update job status
        jobs[job_id]['status'] = 'processing'
        save_jobs()  # 保存状态变更

        # Create a unique folder for this job's results
        job_result_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
        os.makedirs(job_result_dir, exist_ok=True)

        # Get the filename without extension
        pdf_filename = os.path.basename(pdf_path)
        base_filename = os.path.splitext(pdf_filename)[0]

        # Create markdown output path
        markdown_path = os.path.join(job_result_dir, f"{base_filename}.md")
        images_dir = "images"

        # Create progress reporter
        progress_reporter = ProgressReporter(job_id)

        # 确定是使用GPU还是CPU
        device = "cuda" if app.config['USE_GPU'] else "cpu"
        logger.info(f"Job {job_id}: Using device: {device}")

        # Initialize PDF extractor
        extractor = PDFPageExtractor(
            device=device,
            model_dir_path=app.config['MODEL_DIR']
        )

        # 获取并记录模型信息
        model_files = os.listdir(app.config['MODEL_DIR']) if os.path.exists(app.config['MODEL_DIR']) else []
        logger.info(f"Job {job_id}: Models available: {model_files}")

        # 记录开始处理PDF
        logger.info(f"Job {job_id}: Starting PDF processing with {device} acceleration")

        # Process PDF and write to markdown
        with MarkDownWriter(markdown_path, images_dir, "utf-8") as md:
            for block in extractor.extract(pdf=pdf_path, report_progress=progress_reporter.report):
                md.write(block)

        # Update job status to completed
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = {
            'markdown_path': markdown_path,
            'job_result_dir': job_result_dir,
            'filename': f"{base_filename}.md",
            'device_used': device,
            'model_dir': app.config['MODEL_DIR']
        }
        logger.info(f"Job {job_id}: Completed successfully")
        save_jobs()  # 保存完成状态

    except Exception as e:
        # Update job status to failed
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        logger.error(f"Job {job_id}: Failed with error: {str(e)}")
        save_jobs()  # 保存失败状态

@app.route('/')
def index():
    # 传递设备信息到模板
    return render_template('index.html', use_gpu=app.config['USE_GPU'])

@app.route('/toggle_gpu', methods=['POST'])
def toggle_gpu():
    app.config['USE_GPU'] = not app.config['USE_GPU']
    logger.info(f"GPU acceleration {'enabled' if app.config['USE_GPU'] else 'disabled'}")
    return jsonify({'use_gpu': app.config['USE_GPU']})

@app.route('/system_info')
def system_info():
    # 检查CUDA是否可用
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        cuda_device_name = torch.cuda.get_device_name(0) if cuda_available and cuda_device_count > 0 else "N/A"
    except ImportError:
        cuda_available = False
        cuda_device_count = 0
        cuda_device_name = "PyTorch not installed"

    # 获取模型目录信息
    model_dir = app.config['MODEL_DIR']
    model_files = os.listdir(model_dir) if os.path.exists(model_dir) else []

    return jsonify({
        'use_gpu': app.config['USE_GPU'],
        'cuda_available': cuda_available,
        'cuda_device_count': cuda_device_count,
        'cuda_device_name': cuda_device_name,
        'model_dir': model_dir,
        'model_files': model_files
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('files[]')

    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            # Generate a unique ID for the job
            job_id = str(uuid.uuid4())

            # Secure the filename
            filename = secure_filename(file.filename)

            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.info(f"File uploaded: {filename}, Job ID: {job_id}")

            # Create job entry
            jobs[job_id] = {
                'id': job_id,
                'filename': filename,
                'file_path': file_path,
                'status': 'queued',
                'progress': {
                    'current_page': 0,
                    'total_pages': 0,
                    'percentage': 0
                }
            }

            # 保存作业状态
            save_jobs()

            # Start processing in a separate thread
            thread = threading.Thread(target=process_pdf, args=(job_id, file_path, app.config['RESULTS_FOLDER']))
            thread.start()

            uploaded_files.append({
                'job_id': job_id,
                'filename': filename
            })

    return jsonify({'message': 'Files uploaded successfully', 'jobs': uploaded_files})

@app.route('/jobs')
def get_jobs():
    return jsonify({'jobs': list(jobs.values())})

@app.route('/job/<job_id>')
def get_job(job_id):
    if job_id in jobs:
        return jsonify(jobs[job_id])
    return jsonify({'error': 'Job not found'}), 404

@app.route('/download/<job_id>')
def download_file(job_id):
    if job_id not in jobs or jobs[job_id]['status'] != 'completed':
        return jsonify({'error': 'File not ready for download'}), 404

    job_result_dir = jobs[job_id]['result']['job_result_dir']
    filename = jobs[job_id]['result']['filename']

    # Create a zip file if images are present
    images_dir = os.path.join(job_result_dir, 'images')
    if os.path.exists(images_dir) and os.listdir(images_dir):
        import zipfile
        zip_path = os.path.join(job_result_dir, f"{os.path.splitext(filename)[0]}_with_images.zip")

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add markdown file
            md_path = os.path.join(job_result_dir, filename)
            zipf.write(md_path, arcname=filename)

            # Add all images
            for img_file in os.listdir(images_dir):
                img_path = os.path.join(images_dir, img_file)
                zipf.write(img_path, arcname=os.path.join('images', img_file))

        return send_from_directory(job_result_dir, f"{os.path.splitext(filename)[0]}_with_images.zip", as_attachment=True)

    # If no images, just return the markdown file
    return send_from_directory(job_result_dir, filename, as_attachment=True)

# 应用启动时加载作业状态
load_jobs()

if __name__ == '__main__':
    logger.info(f"Starting application with {'GPU' if app.config['USE_GPU'] else 'CPU'} acceleration")
    logger.info(f"Model directory: {app.config['MODEL_DIR']}")
    app.run(debug=True)