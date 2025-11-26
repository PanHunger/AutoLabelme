import os
import shutil
import yaml
import zipfile
import tarfile
import tempfile
from pathlib import Path
import random
from loguru import logger
import cv2
import numpy as np
from PIL import Image
import filetype

class DataProcessor:
    def __init__(self):
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)

    def process_raw_files(self, uploaded_images, uploaded_labels, train_ratio, val_ratio, test_ratio):
        """处理原始上传的文件并自动分割数据集"""
        try:
            # 创建临时目录
            dataset_name = f"dataset_{random.randint(1000, 9999)}"
            dataset_path = os.path.join(self.temp_dir, dataset_name)
            
            # 创建目录结构
            dirs = ['images/train', 'images/val', 'images/test', 
                   'labels/train', 'labels/val', 'labels/test']
            for dir_path in dirs:
                os.makedirs(os.path.join(dataset_path, dir_path), exist_ok=True)
            
            # 构建文件名映射
            image_files = {img.name: img for img in uploaded_images}
            label_files = {label.name.replace('.txt', ''): label for label in uploaded_labels}
            
            # 验证文件匹配
            common_files = set(image_files.keys()).intersection(set(label_files.keys()))
            if len(common_files) != len(uploaded_images):
                logger.warning("部分图片没有对应的标注文件")
            
            # 随机分割数据集
            file_list = list(common_files)
            random.shuffle(file_list)
            
            n_total = len(file_list)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_files = file_list[:n_train]
            val_files = file_list[n_train:n_train + n_val]
            test_files = file_list[n_train + n_val:]
            
            # 保存文件到对应目录
            self._save_files_to_split(image_files, label_files, train_files, dataset_path, 'train')
            self._save_files_to_split(image_files, label_files, val_files, dataset_path, 'val')
            self._save_files_to_split(image_files, label_files, test_files, dataset_path, 'test')
            
            # 生成data.yaml
            class_names = self._extract_classes_from_labels(label_files, common_files)
            self._create_data_yaml(dataset_path, class_names)
            
            logger.info(f"数据集处理完成: {dataset_path}")
            return dataset_path
            
        except Exception as e:
            logger.error(f"处理原始文件时出错: {e}")
            return None

    def _save_files_to_split(self, image_files, label_files, file_list, dataset_path, split):
        """保存文件到指定的分割目录"""
        for filename in file_list:
            # 保存图片
            img_data = image_files[filename].getvalue()
            img_ext = Path(image_files[filename].name).suffix
            img_path = os.path.join(dataset_path, 'images', split, f"{filename}{img_ext}")
            with open(img_path, 'wb') as f:
                f.write(img_data)
            
            # 保存标注
            label_data = label_files[filename].getvalue()
            label_path = os.path.join(dataset_path, 'labels', split, f"{filename}.txt")
            with open(label_path, 'wb') as f:
                f.write(label_data)

    def _extract_classes_from_labels(self, label_files, common_files):
        """从标注文件中提取类别信息"""
        class_ids = set()
        for filename in common_files:
            try:
                label_content = label_files[filename].getvalue().decode('utf-8')
                for line in label_content.strip().split('\n'):
                    if line:
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
            except Exception as e:
                logger.warning(f"解析标注文件 {filename} 时出错: {e}")
        
        # 创建默认类别名称
        class_names = [f"class_{i}" for i in range(len(class_ids))]
        return class_names

    def _create_data_yaml(self, dataset_path, class_names):
        """创建data.yaml配置文件"""
        data_config = {
            'path': os.path.abspath(dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = os.path.join(dataset_path, 'data.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    def extract_and_validate_dataset(self, uploaded_archive):
        """解压并验证数据集压缩包"""
        try:
            # 创建临时目录
            dataset_name = f"uploaded_{random.randint(1000, 9999)}"
            extract_path = os.path.join(self.temp_dir, dataset_name)
            os.makedirs(extract_path, exist_ok=True)
            
            # 解压文件
            if uploaded_archive.name.endswith('.zip'):
                with zipfile.ZipFile(uploaded_archive, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            elif uploaded_archive.name.endswith(('.tar', '.tar.gz', '.tgz')):
                with tarfile.open(fileobj=uploaded_archive, mode='r:*') as tar_ref:
                    tar_ref.extractall(extract_path)
            
            # 验证目录结构
            validation_result = self.validate_dataset_structure(extract_path)
            
            if validation_result['is_valid']:
                logger.info(f"数据集验证成功: {extract_path}")
                return extract_path
            else:
                logger.error(f"数据集验证失败: {validation_result['issues']}")
                # 清理无效数据集
                shutil.rmtree(extract_path)
                return None
                
        except Exception as e:
            logger.error(f"解压数据集时出错: {e}")
            return None

    def validate_dataset_structure(self, dataset_path):
        """验证数据集目录结构"""
        issues = []
        
        # 检查必要目录
        required_dirs = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test'
        ]
        
        for dir_path in required_dirs:
            full_path = os.path.join(dataset_path, dir_path)
            if not os.path.exists(full_path):
                issues.append(f"缺少目录: {dir_path}")
        
        # 检查data.yaml
        data_yaml_path = os.path.join(dataset_path, 'data.yaml')
        if not os.path.exists(data_yaml_path):
            issues.append("缺少 data.yaml 文件")
        else:
            try:
                with open(data_yaml_path, 'r') as f:
                    data_config = yaml.safe_load(f)
                
                required_keys = ['nc', 'names', 'train', 'val', 'test']
                for key in required_keys:
                    if key not in data_config:
                        issues.append(f"data.yaml 中缺少必要字段: {key}")
            except Exception as e:
                issues.append(f"解析 data.yaml 时出错: {e}")
        
        # 检查图片和标注文件匹配
        for split in ['train', 'val', 'test']:
            img_dir = os.path.join(dataset_path, 'images', split)
            label_dir = os.path.join(dataset_path, 'labels', split)
            
            if os.path.exists(img_dir) and os.path.exists(label_dir):
                img_files = set([f.split('.')[0] for f in os.listdir(img_dir) 
                               if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                label_files = set([f.split('.')[0] for f in os.listdir(label_dir) 
                                 if f.endswith('.txt')])
                
                missing_labels = img_files - label_files
                missing_images = label_files - img_files
                
                if missing_labels:
                    issues.append(f"{split}分割中 {len(missing_labels)} 张图片缺少标注文件")
                if missing_images:
                    issues.append(f"{split}分割中 {len(missing_images)} 个标注文件缺少对应图片")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }

    def validate_dataset(self, dataset_path):
        """全面验证数据集"""
        try:
            # 基本结构验证
            structure_validation = self.validate_dataset_structure(dataset_path)
            
            if not structure_validation['is_valid']:
                return {
                    'is_valid': False,
                    'issues': structure_validation['issues'],
                    'train_images': 0,
                    'train_labels': 0,
                    'val_images': 0,
                    'val_labels': 0,
                    'test_images': 0,
                    'test_labels': 0
                }
            
            # 统计文件数量
            stats = {}
            for split in ['train', 'val', 'test']:
                img_dir = os.path.join(dataset_path, 'images', split)
                label_dir = os.path.join(dataset_path, 'labels', split)
                
                if os.path.exists(img_dir):
                    stats[f'{split}_images'] = len([f for f in os.listdir(img_dir) 
                                                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                else:
                    stats[f'{split}_images'] = 0
                    
                if os.path.exists(label_dir):
                    stats[f'{split}_labels'] = len([f for f in os.listdir(label_dir) 
                                                  if f.endswith('.txt')])
                else:
                    stats[f'{split}_labels'] = 0
            
            # 标注格式验证
            format_issues = self._validate_label_format(dataset_path)
            
            return {
                'is_valid': len(format_issues) == 0,
                'issues': structure_validation['issues'] + format_issues,
                **stats
            }
            
        except Exception as e:
            logger.error(f"验证数据集时出错: {e}")
            return {
                'is_valid': False,
                'issues': [f"验证过程中发生错误: {str(e)}"],
                'train_images': 0,
                'train_labels': 0,
                'val_images': 0,
                'val_labels': 0,
                'test_images': 0,
                'test_labels': 0
            }

    def _validate_label_format(self, dataset_path):
        """验证标注文件格式"""
        issues = []
        
        try:
            # 读取data.yaml获取类别信息
            data_yaml_path = os.path.join(dataset_path, 'data.yaml')
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            num_classes = data_config.get('nc', 0)
            
            # 检查每个分割的标注文件
            for split in ['train', 'val', 'test']:
                label_dir = os.path.join(dataset_path, 'labels', split)
                if not os.path.exists(label_dir):
                    continue
                
                for label_file in os.listdir(label_dir):
                    if not label_file.endswith('.txt'):
                        continue
                    
                    label_path = os.path.join(label_dir, label_file)
                    try:
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                        
                        for line_num, line in enumerate(lines, 1):
                            parts = line.strip().split()
                            if not parts:
                                continue
                            
                            # 检查字段数量
                            if len(parts) != 5:
                                issues.append(f"{label_path}:第{line_num}行 字段数量错误")
                                continue
                            
                            # 检查类别ID
                            try:
                                class_id = int(parts[0])
                                if class_id < 0 or class_id >= num_classes:
                                    issues.append(f"{label_path}:第{line_num}行 类别ID {class_id} 超出范围 (0-{num_classes-1})")
                            except ValueError:
                                issues.append(f"{label_path}:第{line_num}行 类别ID格式错误")
                            
                            # 检查坐标值
                            for i in range(1, 5):
                                try:
                                    coord = float(parts[i])
                                    if coord < 0 or coord > 1:
                                        issues.append(f"{label_path}:第{line_num}行 坐标值超出范围 [0,1]")
                                except ValueError:
                                    issues.append(f"{label_path}:第{line_num}行 坐标值格式错误")
                                    
                    except Exception as e:
                        issues.append(f"读取标注文件 {label_path} 时出错: {e}")
        
        except Exception as e:
            issues.append(f"验证标注格式时发生错误: {e}")
        
        return issues

    def update_class_info(self, dataset_path, class_names):
        """更新类别信息"""
        try:
            data_yaml_path = os.path.join(dataset_path, 'data.yaml')
            
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # 更新类别信息
            data_config['nc'] = len(class_names)
            data_config['names'] = class_names
            
            # 保存更新后的配置
            with open(data_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"成功更新类别信息: {len(class_names)} 个类别")
            return True
            
        except Exception as e:
            logger.error(f"更新类别信息时出错: {e}")
            return False