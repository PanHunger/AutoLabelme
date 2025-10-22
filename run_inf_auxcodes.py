import os
import re  # 引入正则表达式模块
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
        self.models_base_path = 'yolo_weights'
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
                    # 按文件大小排序，选择最大的.pt文件（通常是最新的权重）
                    pt_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                    
                if pt_files:
                    model_info = {
                        'name': model_dir,
                        'path': model_path,
                        'pt_file': pt_files[0],
                    }
                    
                    # 查找参数文件 - 更全面的搜索策略
                    args_file = None
                    possible_args_files = [
                        'args.yaml', 'opt.yaml', 'hyp.yaml', 'config.yaml',
                        'hyp.yaml', 'data.yaml', 'train_args.yaml'
                    ]
                    
                    # 首先在模型根目录查找
                    for args_filename in possible_args_files:
                        test_path = os.path.join(model_path, args_filename)
                        if os.path.exists(test_path):
                            args_file = test_path
                            break
                    
                    # 如果没有找到，递归搜索整个目录
                    if not args_file:
                        for root, dirs, files in os.walk(model_path):
                            for file in files:
                                if any(file.lower().endswith(ext) for ext in ['.yaml', '.yml']):
                                    # 检查文件名是否包含参数相关的关键词
                                    param_keywords = ['arg', 'opt', 'hyp', 'config', 'data', 'train']
                                    if any(keyword in file.lower() for keyword in param_keywords):
                                        args_file = os.path.join(root, file)
                                        break
                            if args_file:
                                break
                    
                    # 如果还没有找到，使用第一个YAML文件
                    if not args_file:
                        yaml_files = glob.glob(os.path.join(model_path, '**', '*.yaml'), recursive=True)
                        yaml_files.extend(glob.glob(os.path.join(model_path, '**', '*.yml'), recursive=True))
                        if yaml_files:
                            args_file = yaml_files[0]
                    
                    model_info['args_file'] = args_file
                    
                    # 读取参数文件
                    if model_info['args_file'] and os.path.exists(model_info['args_file']):
                        try:
                            model_info['args'] = self.read_yaml_file(model_info['args_file'])
                        except Exception as e:
                            logger.warning(f"读取参数文件失败 {model_info['args_file']}: {str(e)}")
                            model_info['args'] = {'error': f'读取失败: {str(e)}'}
                    else:
                        # 尝试从模型文件夹名和文件结构中推断一些信息
                        inferred_info = self.infer_model_info(model_path, pt_files[0])
                        model_info['args'] = inferred_info
                    
                    models[model_dir] = model_info
                    
        return models

    def read_yaml_file(self, file_path):
        """读取YAML文件，支持多种格式和编码"""
        try:
            # 尝试UTF-8编码
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # 尝试GBK编码（中文Windows常用）
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 尝试latin-1编码
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
        
        # 清理内容：移除BOM字符和其他可能的问题字符
        if content.startswith('\ufeff'):
            content = content[1:]  # 移除BOM
        
        # 尝试解析YAML
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            logger.warning(f"YAML解析错误，尝试修复格式: {e}")
            # 尝试修复常见的YAML格式问题
            try:
                # 处理可能的分隔符问题
                content = content.replace('：', ':')  # 中文冒号替换为英文冒号
                # 尝试逐行解析
                lines = content.split('\n')
                parsed_data = {}
                for line in lines:
                    line = line.strip()
                    if line and ':' in line and not line.startswith('#'):
                        parts = line.split(':', 1)
                        key = parts[0].strip()
                        value = parts[1].strip()
                        # 尝试转换数值类型
                        try:
                            if value.isdigit():
                                value = int(value)
                            elif self.is_float(value):
                                value = float(value)
                            elif value.lower() in ['true', 'false']:
                                value = value.lower() == 'true'
                        except:
                            pass
                        parsed_data[key] = value
                return parsed_data if parsed_data else {'info': '自动解析的参数'}
            except Exception as e2:
                logger.error(f"自动解析也失败: {e2}")
                return {'error': f'文件格式无法解析: {str(e)}'}

    def is_float(self, value):
        """检查字符串是否可以转换为浮点数"""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def infer_model_info(self, model_path, pt_file):
        """从文件夹结构和文件信息推断模型信息"""
        info = {}
        
        # 从文件夹名推断
        folder_name = os.path.basename(model_path)
        info['model_name'] = folder_name
        
        # 从pt文件推断
        pt_size = os.path.getsize(pt_file)
        info['model_size'] = f"{pt_size / (1024*1024):.2f} MB"
        
        # 查找可能的类别信息
        data_files = glob.glob(os.path.join(model_path, '**', '*.yaml'), recursive=True)
        for data_file in data_files:
            if 'data' in os.path.basename(data_file).lower():
                try:
                    with open(data_file, 'r', encoding='utf-8') as f:
                        data_content = f.read()
                        if 'names' in data_content or 'nc' in data_content:
                            info['data_config'] = os.path.basename(data_file)
                            break
                except:
                    pass
        
        # 查找训练日志
        log_files = glob.glob(os.path.join(model_path, '**', '*.log'), recursive=True)
        if log_files:
            info['has_training_logs'] = True
        
        # 查找评估结果图像
        result_images = glob.glob(os.path.join(model_path, '**', '*.png'), recursive=True)
        if result_images:
            info['has_evaluation_results'] = True
            info['result_image_count'] = len(result_images)
        
        if not info:
            info['info'] = '从文件结构推断的基本信息'
        
        return info
    
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
            'confusion_matrix': ['confusion_matrix_normalized'],
            'precision_recall': ['pr_curve'],
            # 'f1_curve': ['F1_curve'],
            'results': ['results'],
            'labels': ['labels'],
            'val_batch_pred': ['val_batch0_pred'],
            'val_batch_labels': ['val_batch0_labels']
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
                # 使用正则表达式进行严格匹配
                regex = re.compile(rf"^{pattern}(\.[a-zA-Z0-9]+)?$", re.IGNORECASE)
                matched_files = [f for f in all_image_files if regex.match(os.path.basename(f))]
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
        # 尝试多种可能的键名
        possible_keys = ['data', 'data_path', 'dataset', 'train', 'val']
        for key in possible_keys:
            if key in model_info['args']:
                data_path = model_info['args'][key]
                break
    
    # 如果从args中没找到，尝试常见的数据路径
    if not data_path:
        # 在模型文件夹内查找常见的数据目录
        common_data_dirs = ['data', 'images', 'val', 'test', 'train', 'dataset']
        for dir_name in common_data_dirs:
            test_path = os.path.join(model_info['path'], dir_name)
            if os.path.exists(test_path):
                data_path = test_path
                break
    
    # 如果还是没找到，在模型文件夹内递归查找图像文件
    if not data_path:
        data_path = model_info['path']
    
    logger.info(f"搜索示例图像在路径: {data_path}")
    
    # 查找图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    # 递归搜索所有图像文件
    for ext in image_extensions:
        pattern = os.path.join(data_path, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    # 如果没有找到，尝试非递归搜索
    if not image_files:
        for ext in image_extensions:
            pattern = os.path.join(data_path, ext)
            image_files.extend(glob.glob(pattern))
    
    # 限制数量并处理
    for img_path in image_files[:100]:  # 限制前100个
        try:
            # 检查文件是否有效
            if os.path.getsize(img_path) == 0:
                continue
                
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                
                # 创建安全的文件名
                filename = secure_filename(os.path.basename(img_path))
                # 使用哈希避免文件名冲突
                file_hash = hash(img_path) % 10000
                unique_filename = f"{file_hash}_{filename}"
                
                # 目标路径
                dest_dir = os.path.join('static', 'samples', model_name)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, unique_filename)
                
                # 复制文件（如果不存在）
                if not os.path.exists(dest_path):
                    try:
                        shutil.copy2(img_path, dest_path)
                    except Exception as e:
                        logger.warning(f"复制图像失败 {img_path} -> {dest_path}: {e}")
                        # 如果复制失败，直接使用原路径（如果可访问）
                        sample_images.append({
                            'filename': filename,
                            'path': img_path,
                            'url': f"/{img_path}" if img_path.startswith('static/') else f"/static/samples/{model_name}/{unique_filename}",
                            'dimensions': f"{width}x{height}",
                            'size': os.path.getsize(img_path)
                        })
                        continue
                
                sample_images.append({
                    'filename': filename,
                    'path': dest_path,
                    'url': f"/static/samples/{model_name}/{unique_filename}",
                    'dimensions': f"{width}x{height}",
                    'size': os.path.getsize(img_path)
                })
                
        except Exception as e:
            logger.error(f"处理样本图像失败 {img_path}: {str(e)}")
            continue
    
    logger.info(f"找到 {len(sample_images)} 张示例图像")
    
    # 如果没有找到图像，提供一些默认示例
    if not sample_images:
        logger.warning(f"在 {data_path} 中没有找到图像文件")
        # 可以在这里添加一些默认示例图像
    
    return jsonify({'images': sample_images})

@app.route('/api/infer', methods=['POST'])
def infer_image():
    """执行推理"""
    if yolo_model is None:
        return jsonify({'error': '请先加载模型'}), 400
    
    # 支持多种Content-Type
    if request.is_json:
        data = request.get_json()
    else:
        # 尝试解析表单数据或其他格式
        try:
            if request.content_type.startswith('application/x-www-form-urlencoded'):
                data = {
                    'image_path': request.form.get('image_path'),
                    'parameters': {
                        'conf': float(request.form.get('conf', 0.25)),
                        'iou': float(request.form.get('iou', 0.45)),
                        'imgsz': int(request.form.get('imgsz', 640))
                    }
                }
            else:
                # 尝试直接解析JSON
                try:
                    data = request.get_json(force=True, silent=True)
                    if data is None:
                        # 尝试解析原始数据
                        raw_data = request.get_data(as_text=True)
                        if raw_data:
                            import json
                            data = json.loads(raw_data)
                except:
                    return jsonify({'error': '无法解析请求数据，请使用JSON格式'}), 400
        except Exception as e:
            return jsonify({'error': f'请求数据格式错误: {str(e)}'}), 400
    
    if not data:
        return jsonify({'error': '请求数据为空'}), 400
    
    image_path = data.get('image_path')
    parameters = data.get('parameters', {})
    
    # 验证图像路径
    if not image_path:
        return jsonify({'error': '未提供图像路径'}), 400
    
    # 确保路径正确
    if not os.path.exists(image_path):
        # 尝试在uploads目录查找
        if not image_path.startswith('static/'):
            alt_path = os.path.join('static', 'uploads', os.path.basename(image_path))
            if os.path.exists(alt_path):
                image_path = alt_path
            else:
                return jsonify({'error': f'图像文件不存在: {image_path}'}), 400
    
    try:
        # 设置推理参数
        conf = parameters.get('conf', 0.25)
        iou = parameters.get('iou', 0.45)
        imgsz = parameters.get('imgsz', 640)
        
        # 验证参数范围
        conf = max(0.01, min(0.99, conf))
        iou = max(0.01, min(0.99, iou))
        imgsz = max(320, min(1280, imgsz))
        
        logger.info(f"开始推理: {image_path}, 参数: conf={conf}, iou={iou}, imgsz={imgsz}")
        
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
                for i, box in enumerate(result.boxes):
                    # 确保索引有效
                    cls_index = int(box.cls) if box.cls is not None else 0
                    class_name = yolo_model.names[cls_index] if cls_index < len(yolo_model.names) else f"class_{cls_index}"
                    
                    detections.append({
                        'class': class_name,
                        'confidence': float(box.conf) if box.conf is not None else 0.0,
                        'bbox': box.xyxy[0].tolist() if box.xyxy is not None else [0, 0, 0, 0]
                    })
            
            response = {
                'success': True,
                'result_image': f"/static/results/{result_filename}",
                'inference_time': round(inference_time, 3),
                'image_size': f"{result.orig_shape[1]}x{result.orig_shape[0]}" if hasattr(result, 'orig_shape') else '未知',
                'detections': detections,
                'detection_count': len(detections),
                'model_name': getattr(yolo_model, 'model_name', '未知模型')
            }
            
            return jsonify(response)
        else:
            return jsonify({'error': '推理失败，未返回结果'}), 500
            
    except Exception as e:
        logger.error(f"推理失败: {str(e)}", exc_info=True)
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
    
@app.route('/static/samples/<model_name>/<filename>')
def serve_sample_image(model_name, filename):
    """提供示例图像"""
    try:
        file_path = os.path.join('static', 'samples', model_name, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return jsonify({'error': '图像不存在'}), 404
    except Exception as e:
        logger.error(f"提供示例图像失败: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500

# 添加通用静态文件服务（用于直接访问模型文件夹中的图像）
@app.route('/model_files/<path:filename>')
def serve_model_file(filename):
    """提供模型文件夹中的文件"""
    try:
        # 安全检查：确保文件路径在模型文件夹内
        safe_path = os.path.normpath(filename)
        if '..' in safe_path or safe_path.startswith('/'):
            return jsonify({'error': '无效的文件路径'}), 400
            
        file_path = os.path.join('yolo_weights', safe_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            # 根据文件类型设置MIME类型
            ext = os.path.splitext(file_path)[1].lower()
            mime_types = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg', 
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.yaml': 'application/x-yaml',
                '.yml': 'application/x-yaml'
            }
            mime_type = mime_types.get(ext, 'application/octet-stream')
            return send_file(file_path, mimetype=mime_type)
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        logger.error(f"提供模型文件失败: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '文件太大'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"服务器错误: {str(e)}")
    return jsonify({'error': '内部服务器错误'}), 500

@app.before_request
def before_request():
    """在请求前执行，用于日志记录"""
    logger.info(f"请求: {request.method} {request.path} - Content-Type: {request.content_type}")

@app.after_request
def after_request(response):
    """在请求后执行，用于CORS和日志记录"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': '请求格式错误'}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '资源未找到'}), 404

@app.errorhandler(415)
def unsupported_media_type(error):
    return jsonify({'error': '不支持的媒体类型，请使用application/json'}), 415

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"服务器内部错误: {str(error)}")
    return jsonify({'error': '服务器内部错误'}), 500

@app.route('/api/test_json', methods=['POST'])
def test_json():
    """测试JSON解析端点"""
    try:
        if request.is_json:
            data = request.get_json()
            return jsonify({
                'success': True,
                'message': 'JSON解析成功',
                'received_data': data,
                'content_type': request.content_type
            })
        else:
            # 尝试强制解析
            data = request.get_json(force=True, silent=True)
            if data:
                return jsonify({
                    'success': True,
                    'message': 'JSON强制解析成功',
                    'received_data': data,
                    'content_type': request.content_type
                })
            else:
                return jsonify({
                    'success': False,
                    'message': '无法解析JSON',
                    'content_type': request.content_type,
                    'raw_data': request.get_data(as_text=True)[:500]  # 前500字符
                }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'content_type': request.content_type
        }), 400

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