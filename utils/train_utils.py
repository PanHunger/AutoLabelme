import os
import yaml
import torch
from ultralytics import YOLO
from loguru import logger
import threading
import queue
import time

class YOLOTrainer:
    def __init__(self):
        self.training_thread = None
        self.training_queue = queue.Queue()
        self.is_training = False
        self.current_training = None

    def start_training(self, dataset_path, model_name, training_config):
        """开始训练任务"""
        try:
            logger.info(f"开始训练: 数据集={dataset_path}, 模型={model_name}")
            
            # 准备训练参数
            train_args = self._prepare_training_args(dataset_path, model_name, training_config)
            
            # 在后台线程中运行训练
            self.is_training = True
            self.training_thread = threading.Thread(
                target=self._run_training,
                args=(train_args,)
            )
            self.training_thread.daemon = True
            self.training_thread.start()
            
            # 等待训练开始
            time.sleep(2)
            
            return {
                'success': True,
                'message': '训练已开始',
                'results_path': train_args.get('project', 'runs/detect/train')
            }
            
        except Exception as e:
            logger.error(f"启动训练时出错: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _prepare_training_args(self, dataset_path, model_name, config):
        """准备训练参数"""
        # 构建data.yaml路径
        data_yaml_path = os.path.join(dataset_path, 'data.yaml')
        
        # 设备配置
        device = 'cpu'
        if config['device'] != 'CPU':
            device = config['device'].split()[-1]  # 提取GPU编号
        
        # 训练参数
        args = {
            'data': data_yaml_path,
            'epochs': config['epochs'],
            'imgsz': config['img_size'],
            'batch': config['batch_size'],
            'lr0': config['learning_rate'],
            'optimizer': config['optimizer'].lower(),
            'weight_decay': config['weight_decay'],
            'momentum': config['momentum'],
            'device': device,
            'workers': config['workers'],
            'patience': config['patience'],
            'save': config['save_best'],
            'exist_ok': True,  # 允许覆盖现有运行
            'project': 'runs/detect',
            'name': f'train_{int(time.time())}'
        }
        
        # 断点续训
        if config['resume']:
            args['resume'] = True
        
        return args

    def _run_training(self, train_args):
        """在后台线程中运行训练"""
        try:
            logger.info("训练线程启动")
            
            # 加载模型
            model = YOLO(train_args['model'] if 'model' in train_args else train_args.get('pretrained', 'yolov8n.pt'))
            
            # 开始训练
            results = model.train(**{k: v for k, v in train_args.items() if k != 'model'})
            
            # 保存训练结果
            self.training_queue.put({
                'success': True,
                'results': results,
                'best_model': results.save_dir if hasattr(results, 'save_dir') else train_args['project']
            })
            
            logger.info("训练完成")
            
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            self.training_queue.put({
                'success': False,
                'error': str(e)
            })
        
        finally:
            self.is_training = False

    def get_training_status(self):
        """获取训练状态"""
        try:
            return self.training_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_training(self):
        """停止训练"""
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            # 这里需要更优雅的方式停止训练
            pass

    def export_model(self, model_path, export_config):
        """导出模型到指定格式"""
        try:
            logger.info(f"开始导出模型: {model_path} -> {export_config['format']}")
            
            # 加载训练好的模型
            if isinstance(model_path, str):
                model = YOLO(model_path)
            else:
                model = model_path
            
            # 执行导出
            export_args = {'imgsz': export_config['imgsz']}
            
            if export_config['format'] == 'onnx' and 'opset' in export_config:
                export_args['opset'] = export_config['opset']
            
            if export_config['format'] == 'engine' and 'half' in export_config:
                export_args['half'] = export_config['half']
            
            exported_path = model.export(**export_args)
            
            return {
                'success': True,
                'exported_path': exported_path,
                'format': export_config['format']
            }
            
        except Exception as e:
            logger.error(f"导出模型时出错: {e}")
            return {
                'success': False,
                'error': str(e)
            }