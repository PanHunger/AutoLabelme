import os
from pathlib import Path

class Config:
    # 基础配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # 文件上传配置
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    
    # 模型配置
    MODELS_BASE_PATH = 'yolo_weights'
    
    # 推理默认参数
    DEFAULT_CONF = 0.25
    DEFAULT_IOU = 0.45
    DEFAULT_IMGSZ = 640
    
    @staticmethod
    def init_app(app):
        # 创建必要的目录
        Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
        Path('static/results').mkdir(exist_ok=True)
        Path('static/downloads').mkdir(exist_ok=True)
        Path('static/samples').mkdir(exist_ok=True)