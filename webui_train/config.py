import os

# 基础配置
SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'

# 上传配置
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB最大文件大小

# 会话配置
SESSION_TYPE = 'filesystem'
SESSION_PERMANENT = False

# 允许的文件扩展名
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_LABEL_EXTENSIONS = {'txt'}
ALLOWED_MODEL_EXTENSIONS = {'pt'}
ALLOWED_DATASET_EXTENSIONS = {'zip', 'tar', 'gz'}