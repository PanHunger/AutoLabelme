import os
import yaml
import glob
import json
import time
import shutil
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import torch
from pathlib import Path
import threading
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# 创建必要的目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# 全局变量
available_models = {}
current_model = None
yolo_model = None
model_lock = threading.Lock()

import yaml
import json

def safe_load_yaml(file_path):
    """安全地读取YAML文件，支持多种格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 尝试标准YAML解析
        try:
            return yaml.safe_load(content)
        except:
            # 如果标准解析失败，尝试处理可能的问题
            # 移除可能的BOM字符
            if content.startswith('\ufeff'):
                content = content[1:]
            
            # 尝试不同的编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
                return yaml.safe_load(content)
            except:
                pass
            
            # 如果是JSON格式
            try:
                return json.loads(content)
            except:
                pass
            
            # 最后尝试逐行解析
            lines = content.split('\n')
            parsed_data = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    # 尝试转换数值类型
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except:
                        pass
                    parsed_data[key] = value
            
            return parsed_data if parsed_data else {'info': '文件格式无法解析'}
            
    except Exception as e:
        return {'error': f'读取文件失败: {str(e)}'}

class YOLOModelManager:
    def __init__(self):
        self.models_base_path = '../../yolo_weights'
        self.current_model = None
        self.model_instance = None
        
    def scan_models(self, keyword=None):
        """扫描可用的YOLO模型"""
        models = {}
        if not os.path.exists(self.models_base_path):
            os.makedirs(self.models_base_path)
            return models
            
        for model_dir in os.listdir(self.models_base_path):
            model_path = os.path.join(self.models_base_path, model_dir)
            if os.path.isdir(model_path):
                # 检查关键字筛选
                if keyword and keyword.lower() not in model_dir.lower():
                    continue
                    
                # 检查是否存在best.pt文件
                pt_files = glob.glob(os.path.join(model_path, '**', 'best.pt'), recursive=True)
                if not pt_files:
                    # 如果没有best.pt，尝试查找其他.pt文件
                    pt_files = glob.glob(os.path.join(model_path, '**', '*.pt'), recursive=True)
                    
                if pt_files:
                    model_info = {
                        'name': model_dir,
                        'path': model_path,
                        'pt_file': pt_files[0],
                        'args_file': os.path.join(model_path, 'args.yaml')
                    }
                    
                    # 尝试多种可能的参数文件名
                    if not os.path.exists(model_info['args_file']):
                        # 尝试其他可能的参数文件名
                        possible_args_files = [
                            os.path.join(model_path, 'opt.yaml'),
                            os.path.join(model_path, 'config.yaml'),
                            os.path.join(model_path, 'hyp.yaml'),
                        ]
                        for args_file in possible_args_files:
                            if os.path.exists(args_file):
                                model_info['args_file'] = args_file
                                break
                    
                    # 读取参数文件
                    if os.path.exists(model_info['args_file']):
                        try:
                            with open(model_info['args_file'], 'r', encoding='utf-8') as f:
                                model_info['args'] = safe_load_yaml(model_info['args_file'])
                        except Exception as e:
                            logger.warning(f"读取参数文件失败 {model_info['args_file']}: {str(e)}")
                            # 尝试其他编码方式
                            try:
                                with open(model_info['args_file'], 'r', encoding='gbk') as f:
                                    model_info['args'] = safe_load_yaml(model_info['args_file'])
                            except:
                                model_info['args'] = {'error': '无法读取参数文件'}
                    else:
                        model_info['args'] = {'info': '未找到参数文件'}
                    
                    models[model_dir] = model_info
                    
        return models
    
    def load_model(self, model_name):
        """加载指定的YOLO模型"""
        global yolo_model
        
        if model_name not in available_models:
            return False, "模型不存在"
            
        model_info = available_models[model_name]
        
        try:
            # 使用线程锁确保线程安全
            with model_lock:
                if self.model_instance is not None:
                    del self.model_instance
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # 动态导入ultralytics
                try:
                    from ultralytics import YOLO
                except ImportError:
                    return False, "请安装ultralytics: pip install ultralytics"
                
                # 加载模型
                self.model_instance = YOLO(model_info['pt_file'])
                self.current_model = model_name
                yolo_model = self.model_instance
                
            return True, "模型加载成功"
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False, f"加载模型失败: {str(e)}"
    
    def get_model_info(self, model_name):
        """获取模型详细信息"""
        if model_name not in available_models:
            return None
            
        model_info = available_models[model_name].copy()
        
        # 获取评估指标图像
        model_path = model_info['path']
        metrics_images = {}
        
        # 改进的图像查找逻辑
        image_extensions = ['.png', '.jpg', '.jpeg']
        
        # 定义评估图像类型和可能的文件名模式
        metric_patterns = {
            'confusion_matrix': ['confusion_matrix', 'confusion', 'cm'],
            'precision_recall': ['precision_recall', 'pr_curve', 'precision-recall'],
            'f1_curve': ['f1_curve', 'f1_curve', 'f1-score'],
            'results': ['results', 'metrics', 'training_results'],
            'labels': ['labels', 'detections'],
            'val_batch_pred': ['val_batch_pred', 'validation_predictions'],
            'val_batch_labels': ['val_batch_labels', 'validation_labels']
        }
        
        # 递归查找所有图像文件
        all_image_files = []
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    all_image_files.append(os.path.join(root, file))
        
        # 匹配评估图像
        for metric_name, patterns in metric_patterns.items():
            for pattern in patterns:
                matched_files = [f for f in all_image_files if pattern in os.path.basename(f).lower()]
                if matched_files:
                    # 选择最匹配的文件（通常是最大的文件）
                    matched_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                    metrics_images[metric_name] = matched_files[0]
                    break
        
        model_info['metrics_images'] = metrics_images
        
        return model_info

    # 添加图像服务路由
    @app.route('/api/model_image/<model_name>/<image_type>')
    def get_model_image(model_name, image_type):
        """提供模型评估图像"""
        try:
            if model_name not in available_models:
                return jsonify({'error': '模型不存在'}), 404
            
            model_info = model_manager.get_model_info(model_name)
            if not model_info or 'metrics_images' not in model_info:
                return jsonify({'error': '模型信息不完整'}), 404
            
            if image_type not in model_info['metrics_images']:
                # 尝试查找相近的图像
                all_images = []
                for root, dirs, files in os.walk(available_models[model_name]['path']):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            all_images.append(os.path.join(root, file))
                
                # 查找包含image_type关键词的图像
                matched_images = [img for img in all_images if image_type in os.path.basename(img).lower()]
                if not matched_images:
                    return jsonify({'error': '未找到对应的评估图像'}), 404
                
                # 使用第一个匹配的图像
                image_path = matched_images[0]
            else:
                image_path = model_info['metrics_images'][image_type]
            
            if not os.path.exists(image_path):
                return jsonify({'error': '图像文件不存在'}), 404
            
            # 返回图像文件
            return send_file(image_path)
            
        except Exception as e:
            logger.error(f"获取模型图像失败: {str(e)}")
            return jsonify({'error': '服务器内部错误'}), 500

model_manager = YOLOModelManager()

def allowed_file(filename):
    """检查文件类型"""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/api/models')
def get_models():
    """获取模型列表"""
    keyword = request.args.get('keyword', '')
    global available_models
    available_models = model_manager.scan_models(keyword)
    
    models_list = [{
        'name': name,
        'has_args': 'args' in info and info['args']
    } for name, info in available_models.items()]
    
    return jsonify({'models': models_list})

@app.route('/api/model/<model_name>')
def get_model_details(model_name):
    """获取模型详细信息"""
    model_info = model_manager.get_model_info(model_name)
    if not model_info:
        return jsonify({'error': '模型不存在'}), 404
    
    return jsonify(model_info)

@app.route('/api/model/<model_name>/load', methods=['POST'])
def load_model(model_name):
    """加载模型"""
    success, message = model_manager.load_model(model_name)
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'error': message}), 400

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """上传文件"""
    if 'files[]' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    
    files = request.files.getlist('files[]')
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # 获取图像信息
            img = cv2.imread(file_path)
            if img is not None:
                height, width = img.shape[:2]
                file_info = {
                    'filename': unique_filename,
                    'original_name': filename,
                    'path': file_path,
                    'url': f"/static/uploads/{unique_filename}",
                    'size': os.path.getsize(file_path),
                    'dimensions': f"{width}x{height}"
                }
                uploaded_files.append(file_info)
    
    return jsonify({'files': uploaded_files})

@app.route('/api/sample_images/<model_name>')
def get_sample_images(model_name):
    """获取示例图像"""
    if model_name not in available_models:
        return jsonify({'error': '模型不存在'}), 404
    
    model_info = available_models[model_name]
    sample_images = []
    
    # 从args.yaml中获取数据路径
    data_path = None
    if 'args' in model_info and model_info['args']:
        data_path = model_info['args'].get('data')
    
    if data_path and os.path.exists(data_path):
        # 查找图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        for ext in image_extensions:
            image_files = glob.glob(os.path.join(data_path, '**', ext), recursive=True)
            image_files.extend(glob.glob(os.path.join(data_path, '**', ext.upper()), recursive=True))
            
            for img_path in image_files[:50]:  # 限制数量
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        # 复制到静态目录
                        filename = os.path.basename(img_path)
                        dest_path = os.path.join('static', 'samples', model_name, filename)
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        
                        if not os.path.exists(dest_path):
                            shutil.copy2(img_path, dest_path)
                        
                        sample_images.append({
                            'filename': filename,
                            'path': dest_path,
                            'url': f"/{dest_path}",
                            'dimensions': f"{width}x{height}"
                        })
                except Exception as e:
                    logger.error(f"处理样本图像失败 {img_path}: {str(e)}")
    
    return jsonify({'images': sample_images})

@app.route('/api/infer', methods=['POST'])
def infer_image():
    """执行推理"""
    if yolo_model is None:
        return jsonify({'error': '请先加载模型'}), 400
    
    data = request.json
    image_path = data.get('image_path')
    parameters = data.get('parameters', {})
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': '图像文件不存在'}), 400
    
    try:
        # 设置推理参数
        conf = parameters.get('conf', 0.25)
        iou = parameters.get('iou', 0.45)
        imgsz = parameters.get('imgsz', 640)
        
        # 执行推理
        start_time = time.time()
        results = yolo_model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=False
        )
        inference_time = time.time() - start_time
        
        # 处理结果
        if results and len(results) > 0:
            result = results[0]
            
            # 保存结果图像
            timestamp = int(time.time())
            result_filename = f"result_{timestamp}.jpg"
            result_path = os.path.join('static', 'results', result_filename)
            
            # 绘制结果
            result_img = result.plot()
            cv2.imwrite(result_path, result_img)
            
            # 收集检测信息
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    detections.append({
                        'class': yolo_model.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist()
                    })
            
            response = {
                'success': True,
                'result_image': f"/static/results/{result_filename}",
                'inference_time': round(inference_time, 3),
                'image_size': f"{result.orig_shape[1]}x{result.orig_shape[0]}",
                'detections': detections,
                'detection_count': len(detections)
            }
            
            return jsonify(response)
        else:
            return jsonify({'error': '推理失败'}), 500
            
    except Exception as e:
        logger.error(f"推理失败: {str(e)}")
        return jsonify({'error': f'推理失败: {str(e)}'}), 500

@app.route('/api/save_image', methods=['POST'])
def save_image():
    """保存图像到本地"""
    data = request.json
    image_url = data.get('image_url')
    filename = data.get('filename', 'saved_image.jpg')
    
    if not image_url:
        return jsonify({'error': '没有提供图像URL'}), 400
    
    try:
        # 从URL获取实际路径
        if image_url.startswith('/static/'):
            source_path = image_url[1:]  # 移除开头的斜杠
        else:
            source_path = os.path.join('static', 'results', os.path.basename(image_url))
        
        if not os.path.exists(source_path):
            return jsonify({'error': '源文件不存在'}), 404
        
        # 确保文件名安全
        filename = secure_filename(filename)
        save_path = os.path.join('static', 'downloads', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        shutil.copy2(source_path, save_path)
        
        return jsonify({
            'success': True,
            'message': '图像保存成功',
            'download_url': f"/static/downloads/{filename}"
        })
        
    except Exception as e:
        logger.error(f"保存图像失败: {str(e)}")
        return jsonify({'error': f'保存失败: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '文件太大'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"服务器错误: {str(e)}")
    return jsonify({'error': '内部服务器错误'}), 500

if __name__ == '__main__':
    # 初始化扫描模型
    available_models = model_manager.scan_models()
    
    # 获取本机IP地址
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"服务器启动在: http://{local_ip}:5000")
    print("局域网内其他用户可以通过此地址访问")
    
    app.run(host='0.0.0.0', port=5000, debug=True)