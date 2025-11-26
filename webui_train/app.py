from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from flask_session import Session
import os
import yaml
import tempfile
import zipfile
import tarfile
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非 GUI 后端
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import numpy as np
from loguru import logger
import sys
import torch
from werkzeug.utils import secure_filename
import threading
import time
import json
import socket

# 导入自定义工具模块
from utils.data_processor import DataProcessor
from utils.train_utils import YOLOTrainer
from utils.visualization import TrainingVisualizer

app = Flask(__name__)
app.config.from_pyfile('config.py')

# 配置会话
Session(app)

class YOLOTrainingManager:
    def __init__(self):
        self.setup_logging()
        self.data_processor = DataProcessor()
        self.trainer = YOLOTrainer()
        self.visualizer = TrainingVisualizer()
        self.training_thread = None
        self.training_active = False
        
    def setup_logging(self):
        """配置日志系统"""
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add("logs/training_{time}.log", rotation="10 MB")

training_manager = YOLOTrainingManager()

def allowed_file(filename, allowed_extensions):
    """检查文件扩展名"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    """首页"""
    # 检查系统状态
    system_status = {
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        'ultralytics_available': False,
        'onnx_available': False
    }
    
    try:
        from ultralytics import YOLO
        system_status['ultralytics_available'] = True
    except ImportError:
        pass
        
    try:
        import onnx
        system_status['onnx_available'] = True
    except ImportError:
        pass
    
    return render_template('index.html', system_status=system_status)

@app.route('/data_preparation', methods=['GET', 'POST'])
def data_preparation():
    """数据准备页面"""
    if request.method == 'POST':
        if 'dataset_zip' in request.files:
            # 处理压缩包上传
            file = request.files['dataset_zip']
            if file and allowed_file(file.filename, ['zip', 'tar', 'gz']):
                filename = secure_filename(file.filename)
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                
                # 解压和验证数据集
                dataset_path = training_manager.data_processor.extract_and_validate_dataset(file_path)
                if dataset_path:
                    session['dataset_path'] = dataset_path
                    return jsonify({'success': True, 'message': f'数据集处理完成！路径: {dataset_path}'})
                else:
                    return jsonify({'success': False, 'message': '数据集处理失败'})
        
        elif 'images' in request.files and 'labels' in request.files:
            # 处理原始文件上传
            images = request.files.getlist('images')
            labels = request.files.getlist('labels')
            train_ratio = float(request.form.get('train_ratio', 0.8))
            val_ratio = float(request.form.get('val_ratio', 0.1))
            test_ratio = float(request.form.get('test_ratio', 0.1))
            
            dataset_path = training_manager.data_processor.process_raw_files(
                images, labels, train_ratio, val_ratio, test_ratio
            )
            
            if dataset_path:
                session['dataset_path'] = dataset_path
                return jsonify({'success': True, 'message': f'数据集处理完成！保存路径: {dataset_path}'})
            else:
                return jsonify({'success': False, 'message': '数据集处理失败'})
    
    return render_template('data_preparation.html')

@app.route('/validate_dataset')
def validate_dataset():
    """验证数据集"""
    dataset_path = session.get('dataset_path')
    if not dataset_path:
        return jsonify({'success': False, 'message': '请先上传数据集'})
    
    validation_results = training_manager.data_processor.validate_dataset(dataset_path)
    return jsonify(validation_results)

@app.route('/model_configuration', methods=['GET', 'POST'])
def model_configuration():
    """模型配置页面"""
    if request.method == 'POST':
        # 保存训练配置到session
        training_config = {
            'model': request.form.get('model'),
            'epochs': int(request.form.get('epochs', 100)),
            'img_size': int(request.form.get('img_size', 640)),
            'batch_size': int(request.form.get('batch_size', 16)),
            'learning_rate': float(request.form.get('learning_rate', 0.01)),
            'optimizer': request.form.get('optimizer', 'AdamW'),
            'weight_decay': float(request.form.get('weight_decay', 0.0005)),
            'momentum': float(request.form.get('momentum', 0.937)),
            'device': request.form.get('device', 'CPU'),
            'save_best': request.form.get('save_best') == 'true',
            'patience': int(request.form.get('patience', 50)),
            'workers': int(request.form.get('workers', 4))
        }
        
        session['training_config'] = training_config
        
        # 处理自定义模型上传
        if 'custom_model' in request.files:
            custom_model = request.files['custom_model']
            if custom_model and allowed_file(custom_model.filename, ['pt']):
                model_dir = 'uploads/models'
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, secure_filename(custom_model.filename))
                custom_model.save(model_path)
                session['custom_model_path'] = model_path
                training_config['model'] = model_path
        
        return jsonify({'success': True, 'message': '配置已保存'})
    
    # 预训练模型列表
    pretrained_models = {
        "YOLOv8n": "yolov8n.pt",
        "YOLOv8s": "yolov8s.pt", 
        "YOLOv8m": "yolov8m.pt",
        "YOLOv8l": "yolov8l.pt",
        "YOLOv8x": "yolov8x.pt"
    }
    
    # 设备选项
    device_options = ["CPU"]
    if torch.cuda.is_available():
        device_options.extend([f"GPU {i}" for i in range(torch.cuda.device_count())])
    
    return render_template('model_configuration.html', 
                         pretrained_models=pretrained_models,
                         device_options=device_options)

# 在 app.py 中添加这些路由

@app.route('/training_execution')
def training_execution():
    """训练执行页面"""
    if not session.get('training_config'):
        return render_template('training_execution.html', has_config=False)
    
    return render_template('training_execution.html', 
                         has_config=True,
                         training_config=session.get('training_config'))

@app.route('/start_training', methods=['POST'])
def start_training():
    """开始训练"""
    if training_manager.training_active:
        return jsonify({'success': False, 'message': '训练正在进行中'})
    
    dataset_path = session.get('dataset_path')
    training_config = session.get('training_config')
    
    if not dataset_path or not training_config:
        return jsonify({'success': False, 'message': '请先完成数据准备和模型配置'})
    
    # 启动训练线程
    training_manager.training_active = True
    
    def training_thread():
        try:
            # 这里调用实际的训练逻辑
            result = training_manager.trainer.start_training(
                dataset_path,
                training_config['model'],
                training_config
            )
            training_manager.training_active = False
            if result['success']:
                session['training_results'] = result
                logger.info("训练完成")
            else:
                logger.error(f"训练失败: {result.get('error', '未知错误')}")
        except Exception as e:
            logger.error(f"训练线程错误: {e}")
            training_manager.training_active = False
    
    training_manager.training_thread = threading.Thread(target=training_thread)
    training_manager.training_thread.daemon = True
    training_manager.training_thread.start()
    
    return jsonify({'success': True, 'message': '训练已开始'})

@app.route('/training_status')
def training_status():
    """获取训练状态"""
    # 模拟训练进度和日志
    import random
    progress = random.randint(0, 100) if training_manager.training_active else 0
    
    training_logs = []
    if training_manager.training_active:
        sample_logs = [
            "开始训练...",
            f"使用设备: {session.get('training_config', {}).get('device', 'CPU')}",
            "加载数据集...",
            "初始化模型...",
            "开始训练循环...",
            f"进度: {progress}%",
            "优化损失函数...",
            "更新模型权重..."
        ]
        training_logs = sample_logs[:random.randint(3, 8)]
    
    return jsonify({
        'active': training_manager.training_active,
        'logs': training_logs,
        'progress': progress
    })

@app.route('/stop_training', methods=['POST'])
def stop_training():
    """停止训练"""
    training_manager.training_active = False
    if hasattr(training_manager.trainer, 'stop_training'):
        training_manager.trainer.stop_training()
    
    return jsonify({'success': True, 'message': '训练已停止'})

@app.route('/results_view')
def results_view():
    """结果查看页面"""
    training_results = session.get('training_results')
    if not training_results:
        return render_template('results_view.html', has_results=False)
    
    # 获取训练指标
    metrics = training_manager.visualizer.get_training_metrics(training_results.get('results_path', ''))
    
    # 获取预测图像列表
    pred_images = []
    results_path = training_results.get('results_path')
    if results_path:
        val_pred_dir = os.path.join(results_path, 'val_preds')
        if os.path.exists(val_pred_dir):
            pred_images = [f for f in os.listdir(val_pred_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    return render_template('results_view.html', 
                         has_results=True,
                         metrics=metrics,
                         pred_images=pred_images,
                         results_path=results_path)

@app.route('/get_prediction_image/<filename>')
def get_prediction_image(filename):
    """获取预测图像"""
    results_path = session.get('training_results', {}).get('results_path')
    if not results_path:
        return "结果不存在", 404
    
    image_path = os.path.join(results_path, 'val_preds', filename)
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    else:
        return "图像不存在", 404

@app.route('/model_export', methods=['GET', 'POST'])
def model_export():
    """模型导出页面"""
    if request.method == 'POST':
        export_format = request.form.get('export_format')
        img_size = int(request.form.get('img_size', 640))
        
        training_results = session.get('training_results')
        if not training_results:
            return jsonify({'success': False, 'message': '请先完成训练'})
        
        export_config = {
            'format': export_format,
            'imgsz': img_size
        }
        
        # 添加格式特定配置
        if export_format == 'onnx':
            export_config['opset'] = int(request.form.get('opset_version', 12))
        elif export_format == 'engine':
            export_config['half'] = request.form.get('precision') in ['FP16', 'INT8']
        
        try:
            export_result = training_manager.trainer.export_model(
                training_results['best_model'],
                export_config
            )
            
            if export_result['success']:
                session['exported_model_path'] = export_result['exported_path']
                return jsonify({
                    'success': True, 
                    'message': f'模型导出成功！格式: {export_format}',
                    'download_url': f"/download_exported_model"
                })
            else:
                return jsonify({'success': False, 'message': f'模型导出失败: {export_result["error"]}'})
                
        except Exception as e:
            return jsonify({'success': False, 'message': f'导出错误: {str(e)}'})
    
    return render_template('model_export.html')

@app.route('/download_exported_model')
def download_exported_model():
    """下载导出的模型"""
    exported_path = session.get('exported_model_path')
    if not exported_path or not os.path.exists(exported_path):
        return "文件不存在", 404
    
    return send_file(exported_path, as_attachment=True)

@app.route('/download_file/<file_type>')
def download_file(file_type):
    """下载训练文件"""
    training_results = session.get('training_results')
    if not training_results:
        return "训练结果不存在", 404
    
    results_path = training_results.get('results_path')
    if not results_path:
        return "结果路径不存在", 404
    
    file_mapping = {
        'log': ('train_log.txt', 'text/plain'),
        'curves': ('training_curves.png', 'image/png'),
        'config': ('args.yaml', 'text/yaml')
    }
    
    if file_type not in file_mapping:
        return "文件类型不支持", 400
    
    filename, mime_type = file_mapping[file_type]
    file_path = os.path.join(results_path, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, mimetype=mime_type)
    else:
        return "文件不存在", 404

def get_local_ip():
    """获取本机IP地址"""
    try:
        # 连接一个外部地址但不发送数据
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('uploads/models', exist_ok=True)
    os.makedirs('training_results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    local_ip = get_local_ip()
    print(f"YOLO模型训练系统启动成功！")
    print(f"本地访问: http://127.0.0.1:8002")
    print(f"局域网访问: http://{local_ip}:8002")
    
    app.run(debug=True, host='0.0.0.0', port=8002)