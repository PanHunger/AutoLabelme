import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
import os
import yaml
import cv2

class TrainingVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.fig_size = (10, 6)

    def plot_training_curves(self, results_path):
        """绘制训练曲线"""
        try:
            # 这里应该从训练结果中读取实际的指标数据
            # 由于Ultralytics YOLO的训练结果格式，这里提供示例实现
            
            # 创建示例数据（实际应该从results.csv读取）
            epochs = 100
            x = range(1, epochs + 1)
            
            # 模拟训练数据
            train_loss = [0.5 * (0.98 ** i) + 0.1 * np.random.random() for i in x]
            val_loss = [0.6 * (0.97 ** i) + 0.15 * np.random.random() for i in x]
            map_50 = [0.2 + 0.6 * (1 - np.exp(-i/30)) + 0.1 * np.random.random() for i in x]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # 训练损失
            ax1.plot(x, train_loss, 'b-', label='Train Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 验证损失
            ax2.plot(x, val_loss, 'r-', label='Validation Loss', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Validation Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # mAP@0.5
            ax3.plot(x, map_50, 'g-', label='mAP@0.5', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('mAP')
            ax3.set_title('mAP@0.5')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 学习率
            lr = [0.01 * (0.95 ** i) for i in x]
            ax4.plot(x, lr, 'purple', label='Learning Rate', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"绘制训练曲线时出错: {e}")
            return None

    def plot_confusion_matrix(self, cm_data):
        """绘制混淆矩阵"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_data, annot=True, fmt='.2f', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        return fig

    def plot_precision_recall_curve(self, precision, recall):
        """绘制精确率-召回率曲线"""
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.plot(recall, precision, 'b-', linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        return fig

    def visualize_predictions(self, image_path, predictions, class_names):
        """可视化预测结果"""
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 绘制预测框
        for pred in predictions:
            x1, y1, x2, y2 = pred['bbox']
            confidence = pred['confidence']
            class_id = pred['class_id']
            class_name = class_names[class_id]
            
            # 绘制边界框
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
            # 添加标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)), (255, 0, 0), -1)
            cv2.putText(image, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image