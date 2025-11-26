import os
import yaml
from ultralytics import YOLO
import torch
from loguru import logger

class YOLOTrainer:
    def __init__(self):
        self.training_active = False
        self.current_progress = 0
        self.training_logs = []
    
    def start_training(self, dataset_path, model_path, config):
        """开始训练"""
        try:
            self.training_active = True
            self.current_progress = 0
            self.training_logs = []
            
            # 加载模型
            model = YOLO(model_path)
            
            # 训练配置
            train_config = {
                'data': os.path.join(dataset_path, 'data.yaml'),
                'epochs': config['epochs'],
                'imgsz': config['img_size'],
                'batch': config['batch_size'],
                'lr0': config['learning_rate'],
                'optimizer': config['optimizer'],
                'weight_decay': config['weight_decay'],
                'momentum': config['momentum'],
                'device': '0' if config['device'].startswith('GPU') else 'cpu',
                'save': True,
                'exist_ok': True,
                'patience': config['patience'],
                'workers': config['workers']
            }
            
            # 开始训练
            results = model.train(**train_config)
            
            # 训练完成后的处理
            best_model_path = results.save_dir  # 这里需要根据实际结果调整
            
            return {
                'success': True,
                'results_path': str(results.save_dir),
                'best_model': best_model_path,
                'metrics': results.results_dict if hasattr(results, 'results_dict') else {}
            }
            
        except Exception as e:
            logger.error(f"训练错误: {e}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            self.training_active = False
    
    def stop_training(self):
        """停止训练"""
        self.training_active = False
    
    def get_training_progress(self):
        """获取训练进度"""
        return self.current_progress
    
    def get_training_logs(self):
        """获取训练日志"""
        return self.training_logs
    
    def export_model(self, model_path, export_config):
        """导出模型"""
        try:
            model = YOLO(model_path)
            
            # 执行导出
            exported_path = model.export(**export_config)
            
            return {
                'success': True,
                'exported_path': exported_path
            }
        except Exception as e:
            logger.error(f"导出错误: {e}")
            return {
                'success': False,
                'error': str(e)
            }