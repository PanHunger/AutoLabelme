import sys
import argparse
from ultralytics import YOLO
import torch
import json
import yaml
import os
import shutil
import random
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QRadioButton,
    QPushButton,
    QSpinBox,
    QGroupBox,
    QGridLayout,
    QComboBox,
    QDialog,
    QTextEdit,
    QMessageBox,
    QProgressBar,
    QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont
import subprocess
import datetime
from glob import glob
from tqdm import tqdm
import numpy as np

from PyQt5.QtGui import QColor, QPalette



def set_dark_theme(app):
    """设置科技感强的深色主题，并统一字体样式"""
    dark_palette = QPalette()
    dark_color = QColor(35, 35, 45)  # 窗口背景颜色
    text_color = QColor(230, 230, 230)  # 文本颜色
    highlight_color = QColor(0, 122, 204)  # 高亮颜色

    # 设置全局调色板
    dark_palette.setColor(QPalette.Window, dark_color)
    dark_palette.setColor(QPalette.WindowText, text_color)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 35))  # 控件背景
    dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 55))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    dark_palette.setColor(QPalette.ToolTipText, text_color)
    dark_palette.setColor(QPalette.Text, text_color)
    dark_palette.setColor(QPalette.Button, QColor(50, 50, 60))  # 按钮颜色
    dark_palette.setColor(QPalette.ButtonText, text_color)
    dark_palette.setColor(QPalette.Highlight, highlight_color)  # 选中颜色
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))  # 高亮文字颜色
    app.setPalette(dark_palette)

    # 设置全局字体
    font = QFont("微软雅黑", 16)
    app.setFont(font)


def apply_stylesheet(app):
    """设置增强的样式表"""
    app.setStyleSheet("""
        QMainWindow {
            background: qlineargradient(
                spread:pad, x1:0, y1:0, x2:1, y2:1,
                stop:0 #1E1E2E, stop:1 #29293F
            );
        }
        QLabel {
            color: #E6E6E6;
            font-size: 16px;
            padding: 5px;
        }
        QCheckBox {
            color: #E6E6E6;
            font-size: 16px;
            padding: 5px;
        }
        QRadioButton{
            color: #E6E6E6;
            font-size: 16px;
            padding: 5px;
        }
        QLineEdit {
            background-color: #2A2A36;
            color: #DCDCDC;
            font-size: 16px;
            border: 1px solid #3A3A4F;
            border-radius: 5px;
            padding: 5px;
        }
        QTextEdit {
            background-color: #2D2D37;
            color: #DCDCDC;
            font-size: 16px;
            border: 1px solid #3A3A4F;
            border-radius: 5px;
            padding: 5px;
        }
        QComboBox {
            background-color: #2A2A36;
            color: #DCDCDC;
            font-size: 16px;
            border: 1px solid #3A3A4F;
            border-radius: 5px;
            padding: 5px;
        }
        QSpinBox {
            background-color: #2A2A36;
            color: #DCDCDC;
            border: 1px solid #3A3A4F;
            border-radius: 5px;
            font-size: 16px;
            padding: 5px;
        }
        QPushButton {
            background-color: #3A3A4F;
            color: #DCDCDC;
            border: 1px solid #4A4A5F;
            border-radius: 8px;
            padding: 8px 12px;
        }
        QPushButton:hover {
            background-color: #4A4A5F;
        }
        QPushButton:pressed {
            background-color: #2A2A36;
        }
        QGroupBox {
            background: #778899;
            border: 1px solid #3A3A4F;
            font-size: 16px;
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        QMessageBox{
            background: #2A2A36;
            color: #DCDCDC;
            font-size: 16px;
            border: 1px solid #4A4A5F;
            border-radius: 8px;
            padding: 8px 12px;
        }
        QScrollBar:vertical {
            background: #2A2A36;
            font-size: 16px;
            width: 10px;
            margin: 0px 3px 0px 3px;
        }
        QScrollBar::handle:vertical {
            background: #4A4A5F;
            font-size: 16px;
            border-radius: 5px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            background: none;
            font-size: 16px;
            border: none;
        }
    """)


def add_shadow_effect(widget):
    """为控件添加阴影效果"""
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(10)
    shadow.setXOffset(3)
    shadow.setYOffset(3)
    shadow.setColor(QColor(0, 0, 0, 120))  # 柔和阴影
    widget.setGraphicsEffect(shadow)


class HelpDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("帮助文档")
        self.resize(800, 600)

        # 布局和文本显示
        layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)  # 设置为只读模式
        help_text.setText(
            """
标注文件夹路径: 标注的xml, json文件，默认和图片在一个文件夹内
图片文件夹路径: 图片文件夹的路径，注意不要使用中文路径
预训练权重路径: 如果有预训练权重文件（如yolov8n.pt），可将此文件路径填入，否则保持为空
模型输入尺寸:   尽量保证是32的倍数，不需要跟你的图像匹配一致；默认为640，适合绝大多数的检测
每批图像数量:   每一次训练时的图片数量，-1表示自动识别，大约使用60%的显存，内存古用大约是显存的3-4倍，数值越大训练越快，请预留一定的空间，过高的值会导致内存不足中断
最大训练次数:   可以设置高一些，如果训练已达到最优，训练会自动停止，并非跑完所有次数
使用GPU训练:    编号填写GPU的索引，若使用多块GPU可填写0,1,2,3这样，注意英文逗号；取消勾选则使用cpu训练
无改善停止数:   当模型已经达到最优效果时，最多可额外尝试的次数，到达该次数发现无明显改善会自动停止训练防止过拟合
加载线程数:     加载数据时的工作线程数。可以使用多个线程并行地加载数据，以提高数据读取速度。具体的最佳值取决于硬件和数据集的大小
标注类型:       矩形标注会生成常规检测数据集，多边形标注会生成分制数据集，矩形+点标注会生成关键点数据集;如果指定了错误的类型，会被过滤掉，不可同时使用矩形和多边形混合标注
模型大小:       预设的模型大小，n-x从小到大，模型越大精度越高，但推理速度会越慢
小目标检测增强:  YOLO对过小的目标检测精度一般，如果训练后发现识制率不够，那么可以启用该选项
"""
        )
        layout.addWidget(help_text)

        close_button = QPushButton("确定")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)


class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setAutoFillBackground(True)
        self.setFixedHeight(60)  # 标题栏高度

        # 设置标题栏背景颜色
        self.setStyleSheet("""
            background-color: #2B2B2B;
            color: #FFFFFF;
        """)

        # 标题栏布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)  # 设置内边距
        self.setLayout(layout)

        # 标题文本
        self.title_label = QLabel("自动训练程序", self)
        self.title_label.setStyleSheet("color: #FFFFFF; font-size: 18px;")
        self.title_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        # 最小化按钮
        self.minimize_button = QPushButton("-", self)
        self.minimize_button.setFixedSize(50, 50)
        self.minimize_button.setStyleSheet("""
            QPushButton {
                background-color: #2B2B2B;
                color: #FFFFFF;
                border: none;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        self.minimize_button.clicked.connect(self.parent.showMinimized)

        # 关闭按钮
        self.close_button = QPushButton("X", self)
        self.close_button.setFixedSize(50, 50)
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #2B2B2B;
                color: #FFFFFF;
                border: none;
            }
            QPushButton:hover {
                background-color: #FF5C5C;
            }
        """)
        self.close_button.clicked.connect(self.parent.close)

        # 将控件添加到布局中
        layout.addWidget(self.title_label)
        layout.addStretch()  # 添加弹性空间
        layout.addWidget(self.minimize_button)
        layout.addWidget(self.close_button)

        # 标志位，用于实现拖动功能
        self.moving = False
        self.offset = QPoint()

    def mousePressEvent(self, event):
        """鼠标按下事件，用于拖动窗口"""
        if event.button() == Qt.LeftButton:
            self.moving = True
            self.offset = event.globalPos() - self.parent.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        """鼠标移动事件，用于拖动窗口"""
        if self.moving:
            self.parent.move(event.globalPos() - self.offset)

    def mouseReleaseEvent(self, event):
        """鼠标释放事件，停止拖动"""
        self.moving = False


class TrainingInterface(QWidget):
    def __init__(self, img_path=None, annotation_path=None, pretrained_path=None):
        super().__init__()
        self.img_path = QLineEdit(img_path)
        self.annotation_path = QLineEdit(annotation_path)
        self.pretrained_path = QLineEdit(pretrained_path)
        self.train_thread = None
        self.weights_path = None
        self.cfg_path = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Train with Labels")
        self.resize(1200, 980)
        # Main Layout
        main_layout = QVBoxLayout()
        
        # 创建自定义标题栏
        # self.title_bar = CustomTitleBar(self)
        # self.main_content = QWidget(self) # 创建主内容区域
        # self.main_content.setStyleSheet("background-color: #282828;")
        # main_layout.setSpacing(20) # 设置布局，将标题栏和主内容区域添加到布局中
        # main_layout.setContentsMargins(20, 20, 20, 20)
        # main_layout.addWidget(self.title_bar)
        # main_layout.addWidget(self.main_content)

        # Training type selection
        train_type_layout = QHBoxLayout()   
        
        train_type_label = QLabel("模型类型")
        train_type_layout.addWidget(train_type_label)
        
        self.model_version_combo = QComboBox()
        self.model_version_combo.addItems(
            [
            "YOLOv8", 
            "YOLOv5", 
            # "YOLOv11"
            ]
        )
        train_type_layout.addWidget(self.model_version_combo)
              
        main_layout.addLayout(train_type_layout)
        
        train_task_layout = QHBoxLayout()   
             
        train_type_label = QLabel("训练任务类型")
        train_task_layout.addWidget(train_type_label)
        
        self.train_type_combo = QComboBox()
        self.train_type_combo.addItems(["多边形Segment训练(分割模型)", "矩形detect训练(检测模型)"])
        train_task_layout.addWidget(self.train_type_combo)
              
        main_layout.addLayout(train_task_layout)

        # File paths
        file_layout = QGridLayout()
        file_layout.addWidget(QLabel("标注文件夹路径："), 0, 0)
        file_layout.addWidget(self.annotation_path, 0, 1)

        file_layout.addWidget(QLabel("图片文件夹路径："), 1, 0)
        file_layout.addWidget(self.img_path, 1, 1)

        file_layout.addWidget(QLabel("预训练权重路径："), 2, 0)
        file_layout.addWidget(self.pretrained_path, 2, 1)

        file_layout.addWidget(QLabel("模型/Yaml保存路径："), 3, 0)
        if self.img_path.text() != "":
            self.save_path = QLineEdit(os.path.join("yolo_weights", self.img_path.text().split("\\")[-3]))
        else:
            self.save_path = QLineEdit("yolo_weights\\")
        file_layout.addWidget(self.save_path, 3, 1)

        main_layout.addLayout(file_layout)

        # Training parameters
        params_group = QGroupBox("训练参数")
        params_layout = QGridLayout()

        params_layout.addWidget(QLabel("模型输入尺寸"), 0, 0)
        self.input_size = QLineEdit("640")
        params_layout.addWidget(self.input_size, 0, 1)

        params_layout.addWidget(QLabel("每批图像数量"), 0, 2)
        self.batch_size = QLineEdit("-1")
        params_layout.addWidget(self.batch_size, 0, 3)

        params_layout.addWidget(QLabel("最大训练次数"), 1, 0)
        self.max_epochs = QLineEdit("100")
        params_layout.addWidget(self.max_epochs, 1, 1)

        params_layout.addWidget(QLabel("无改善停止次数"), 1, 2)
        self.patience = QLineEdit("20")
        params_layout.addWidget(self.patience, 1, 3)

        self.use_gpu = QCheckBox("使用GPU训练")
        self.use_gpu.setChecked(True)
        params_layout.addWidget(self.use_gpu, 2, 0)
        self.gpus = QLineEdit("0")
        params_layout.addWidget(self.patience, 2, 1)

        params_layout.addWidget(QLabel("加载线程数"), 2, 2)
        self.num_threads = QSpinBox()
        self.num_threads.setValue(4)
        params_layout.addWidget(self.num_threads, 2, 3)
        
        params_layout.addWidget(QLabel("模型大小"), 3, 0)
        self.size_n = QRadioButton("n")
        self.size_n.setChecked(True)
        self.size_s = QRadioButton("s")
        self.size_m = QRadioButton("m")
        self.size_l = QRadioButton("l")
        self.size_x = QRadioButton("x")
        size_layout = QHBoxLayout()
        size_layout.addWidget(self.size_n)
        size_layout.addWidget(self.size_s)
        size_layout.addWidget(self.size_m)
        size_layout.addWidget(self.size_l)
        size_layout.addWidget(self.size_x)
        params_layout.addLayout(size_layout, 3, 1, 1, 3)

        self.p2 = QCheckBox("小目标检测P2增强")
        params_layout.addWidget(self.p2, 4, 0)
        
        self.val = QCheckBox("分配10%验证集")
        params_layout.addWidget(self.val, 4, 2)

        # self.auto_correct = QCheckBox("自动修正图像格式")
        # params_layout.addWidget(self.auto_correct, 4, 2)
        
        params_layout.addWidget(QLabel("旋转角度"), 5, 0)
        self.degree = QSpinBox()
        self.degree.setValue(0)
        self.degree.setRange(0, 90)
        params_layout.addWidget(self.degree, 5, 1)

        params_layout.addWidget(QLabel("缩放比例"), 5, 2)
        self.scale = QSpinBox()
        self.scale.setRange(0, 20)
        self.scale.setValue(10)

        params_layout.addWidget(self.scale, 5, 3)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Labeling parameters
        params_group = QGroupBox("标注参数")
        params_layout = QGridLayout()
        
        self.generate_labels = QCheckBox("自动为剩余数据生成标注")
        self.generate_labels.setChecked(True)
        params_layout.addWidget(self.generate_labels, 1, 0)
        
        self.generate_labels = QCheckBox("只使用Verified标注数据")
        self.generate_labels.setChecked(True)
        params_layout.addWidget(self.generate_labels, 1, 1)
        
        self.sam = QCheckBox("使用Segmentation模型优化标注")
        params_layout.addWidget(self.sam, 1, 2)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Progress Bar and Log
        custom_font = QFont("微软雅黑", 10)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(custom_font)
        main_layout.addWidget(self.log_text)
        
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # Buttons        
        button_layout = QHBoxLayout()
        self.train_button = QPushButton("开始训练")
        add_shadow_effect(self.train_button)
        self.train_button.setFont(custom_font)
        # 假设这是一个 QPushButton
        self.train_button.setStyleSheet("""
            QPushButton {
                background-color: #008B45;
                color: #DCDCDC;
                border: 1px solid #4A4A5F;
                border-radius: 8px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #2E8B57;
            }
            QPushButton:pressed {
                background-color: #698B69;
            }
        """)
        
        self.train_button.clicked.connect(self.start_training)
        
        help_button = QPushButton("说明")
        add_shadow_effect(help_button)
        help_button.setFont(custom_font)
        help_button.clicked.connect(self.show_help_dialog)
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(help_button)
        main_layout.addLayout(button_layout)

        # Set main layout
        self.setLayout(main_layout)

    def closeEvent(self, event):
        """在关闭窗口时释放资源"""
        print("TrainingInterface 窗口正在关闭...")
        
        # 清理可能的临时变量或停止进程
        if self.train_thread is not None:
            if self.train_thread.isRunning():
                QMessageBox.critical(self, "错误", "训练过程不可终止，请等待训练完成，程序会自动关闭此窗口！")
                event.ignore()
                return
            else:
                self.train_thread.stop()  # 停止线程
                self.train_thread.quit()
                self.train_thread.wait()  # 等待线程退出
                self.train_thread.deleteLater()  # 删除线程对象
                self.train_thread.terminate()  # 终止线程
                self.train_thread = None
        
        event.accept()
        # 假设有一些训练进程或线程在运行，确保它们停止
        self.cleanup_resources()

        # 调用 deleteLater() 销毁窗口对象
        self.deleteLater()

        # 调用父类的 closeEvent 方法
        super().closeEvent(event)

    def cleanup_resources(self):
        """清理临时变量和资源"""
        # 假设存在某些训练进程、线程或临时数据
        print("清理 TrainingInterface 的临时变量...")
        # 示例：self.training_process.terminate()
        # 示例：del self.some_temp_data

    def show_help_dialog(self):
        help_dialog = HelpDialog()
        help_dialog.exec_()
        
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def update_log(self, message):
        """更新日志文本框"""
        self.log_text.append(message)

    def start_training(self):
        """启动训练任务"""
        if self.annotation_path.text() == "" or self.img_path.text() == "":
            return QMessageBox.critical(self, "错误", "请填写标注文件夹路径和图片文件夹路径！")
        
        # 获取用户输入参数
        version = self.model_version_combo.currentText()[5:]
        pretrained_model = self.pretrained_path.text()
        
        batch_size = int(self.batch_size.text())
        epochs = int(self.max_epochs.text())
        patience = int(self.patience.text())
        workers = self.num_threads.value()
        
        degree = int(self.degree.text())
        scale = float(self.scale.text()) / 10
        
        imgsz = int(self.input_size.text())
        
        if self.use_gpu.isChecked() and torch.cuda.is_available():
            device = self.gpus.text()
        else:
            device = "cpu"
            
        if self.size_n.isChecked():
            model_type = "n"
        elif self.size_s.isChecked():
            model_type = "s"
        elif self.size_m.isChecked():
            model_type = "m"
        elif self.size_l.isChecked():
            model_type = "l"
        elif self.size_x.isChecked():
            model_type = "x"
        else:
            model_type = "n"
        
        p2 = ""
        if self.p2.isChecked():
            p2 = "-p2"
            
        # # 使用 os.path.dirname 获取父目录路径
        # base_path = os.path.dirname(self.save_path.text())
        # # 使用 os.path.basename 获取最后的目录或文件名
        # folder_name = os.path.basename(self.save_path.text())
        
        base_path = self.save_path.text()
        # 获取当前时间，格式化时间为字符串，例如：2023-10-05_14-30-00
        current_time = datetime.datetime.now()
        
        label_img_num = len(glob(os.path.join(self.img_path.text(), "*.json")))
        folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S") + f"_with_{label_img_num}_images"
            
        params = {
            "version": version,
            "model_type": model_type,
            "pretrained_model": pretrained_model,
            "device": device,
            "batch_size": batch_size,
            "epochs": epochs,
            "degree": degree,
            "scale": scale,
            "imgsz": imgsz,
            "workers": workers,
            "patience": patience,
            "p2": p2,
            "train_type_combo": self.train_type_combo.currentText(),
            "base_path": base_path,
            "folder_name": folder_name,
            "annotation_path": self.annotation_path.text(),
            "image_path": self.img_path.text(),
            "val": self.val.isChecked()
        }
        
        self.weights_path = os.path.join(base_path, folder_name, "weights")
        self.cfg_path = os.path.join("./cfgs", f"{os.path.basename(base_path)}.yaml")

        # 启动训练线程
        self.train_thread = TrainThread(params, self.train_button)
        self.train_thread.progress_signal.connect(self.update_progress)
        self.train_thread.log_signal.connect(self.update_log)
        self.train_thread.start()
        self.train_button.setEnabled(False)
        self.train_button.setText("正在训练中...")
        
    def stop_training(self):
        """停止训练线程"""
        self.train_thread.stop()
     
     
def get_gpu_memory():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], encoding='utf-8'
        )
        return int(result.split('\n')[0])  # 返回第一个 GPU 的显存占用
    except Exception:
        return "N/A"

class TrainThread(QThread):
    progress_signal = pyqtSignal(int)  # 信号：用于更新进度条
    log_signal = pyqtSignal(str)  # 信号：用于更新日志

    def __init__(self, params, botton):
        super().__init__()
        self.params = params
        self.botton = botton
        self.running = True  # 标志位控制线程运行

    def stop(self):
        """安全地停止线程"""
        self.running = False
    
    def find_unique_labels_in_directory(self, directory_path):
        """
        在指定文件夹中查找所有独一无二的label标签。
        """
        def extract_labels_from_json(file_path):
            """
            从单个JSON文件中提取所有label字段。
            """
            labels = set()
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'shapes' in data:
                        for shape in data['shapes']:
                            if 'label' in shape:
                                labels.add(shape['label'])
            except json.JSONDecodeError:
                print(f"Error decoding JSON file: {file_path}")
            return labels
        
        unique_labels = set()
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    labels = extract_labels_from_json(file_path)
                    unique_labels.update(labels)
        # 如果需要将结果存储为字典，键和值可以相同
        labels_dict = {label: i for i, label in enumerate(unique_labels)}
        return unique_labels, labels_dict
    
    def convert_json2txt(self, anno_dir, image_dir, class_map):
        sub_folders = os.listdir(anno_dir)
        if len(sub_folders) == 0 or len(sub_folders) >= 100:
            sub_folders = [""]

        for sub_folder in sub_folders:
            source_path = anno_dir + '\\' + sub_folder
            dest_path = image_dir.replace("images", "labels") + "\\" + sub_folder
            # print(dest_path)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
                print(f"标注目录{dest_path}不存在，已创建目录")
                
            filenames = set(os.listdir(source_path))
            filenames = [fn for fn in filenames if fn.endswith(('.json'))]  # 确保只选择 json 文件 
            filenames.sort()

            for i, filename in enumerate(filenames[:]):
                
                if not os.path.exists(os.path.join(source_path, filename.split('.')[0] + '.json')):
                    with open(os.path.join(dest_path, filename.split('.')[0] + '.txt'), 'w') as f:
                        pass
                    continue
                    
                # 读取标注数据  
                with open(os.path.join(source_path, filename.split('.')[0] + '.json'), 'r') as f:  
                    annotation = json.load(f)  
                
                # 获取图片路径  
                image_path = annotation['imagePath']
                # 获取标注形状  
                shapes = annotation['shapes']  
                h = annotation['imageHeight']
                w = annotation['imageWidth']
                wh = np.array([w, h])
                
                with open(os.path.join(dest_path, filename.split('.')[0] + '.txt'), 'w') as f:
                
                    # 遍历所有形状并绘制多边形 
                    for shape in shapes:  
                        if shape['shape_type'] == 'polygon': # 多边形
                            points = np.array(shape['points'], dtype=np.int32)  
                            points = points.reshape((-1, 2))  # 将点转换为可以规范化的格式  
                            normalized_points = points / wh
                            normalized_points = normalized_points.flatten()
                            f.write(f'{class_map[shape["label"]]} ')
                            for p in normalized_points:
                                f.write(f'{p} ')
                            f.write('\n')
                        if shape['shape_type'] == 'rectangle': # 矩形
                            points = np.array(shape['points'], dtype=np.int32)  
                            points = points.reshape((-1, 2))  # 将点转换为可以规范化的格式  
                            normalized_points = points / wh
                            normalized_points = normalized_points.flatten()
                            f.write(f'{class_map[shape["label"]]} ')
                            for p in normalized_points:
                                f.write(f'{p} ')
                            f.write('\n')
                        if shape['shape_type'] == 'circle':  # 圆形
                            points = np.array(shape['points'], dtype=np.int32)  
                            points = points.reshape((-1, 2))  # 将点转换为可以规范化的格式  
                            normalized_points = points / wh
                            normalized_points = normalized_points.flatten()
                            f.write(f'{class_map[shape["label"]]} ')
                            for p in normalized_points:
                                f.write(f'{p} ')
                            f.write('\n')
                        if shape['shape_type'] == 'line':  # 直线
                            points = np.array(shape['points'], dtype=np.int32)  
                            points = points.reshape((-1, 2))  # 将点转换为可以规范化的格式  
                            normalized_points = points / wh
                            normalized_points = normalized_points.flatten()
                            f.write(f'{class_map[shape["label"]]} ')
                            for p in normalized_points:
                                f.write(f'{p} ')
                            f.write('\n')
                        if shape['shape_type'] == 'linestrip': # 折线
                            points = np.array(shape['points'], dtype=np.int32)  
                            points = points.reshape((-1, 2))  # 将点转换为可以规范化的格式  
                            normalized_points = points / wh
                            normalized_points = normalized_points.flatten()
                            f.write(f'{class_map[shape["label"]]} ')
                            for p in normalized_points:
                                f.write(f'{p} ')
                            f.write('\n')

    def generate_dataset_yaml(self, dataset_path, train_folder, val_folder, labels_dict, output_file):
        """
        生成符合指定格式的数据集 YAML 文件。
        
        Args:
            dataset_path (str): 数据集根路径。
            train_folder (str): 训练图片文件夹（相对于 dataset_path 的相对路径）。
            val_folder (str): 验证图片文件夹（相对于 dataset_path 的相对路径）。
            labels_dict (dict): 类别名称字典，键是类别索引，值是类别名称。
            output_file (str): 输出 YAML 文件路径。
        """
        # 构造数据集配置信息
        
        labels_dict_reverse = {v: k for k, v in labels_dict.items()}
        
        dataset_config = {
            "path": dataset_path,
            "train": train_folder,
            "val": val_folder,
            "test": None,
            "names": labels_dict_reverse
        }

        # 自定义的 YAML 转换函数，确保符合您需要的格式
        def yaml_represent_none(self, _):
            return self.represent_scalar('tag:yaml.org,2002:null', '')

        yaml.add_representer(type(None), yaml_represent_none)  # 确保空值写作空字符串

        # 写入 YAML 文件
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(
                dataset_config, 
                f, 
                default_flow_style=False,  # 禁止使用大括号 {}
                allow_unicode=True         # 支持中文等特殊字符
            )

        print(f"数据集 YAML 配置文件已成功保存到 {output_file}！")
        
    def split_train_val(self, data_path, val_ratio=0.1):

        # 定义数据集路径和划分后的保存路径
        root_path = os.path.dirname(os.path.dirname(data_path))  # 原始数据集路径
    
        images_path = data_path
        labels_path = data_path.replace("images", "labels")

        # print("data_path: ", data_path)
        # print("root_path: ", root_path)
        # print("images_path: ", images_path)
        # print("labels_path: ", labels_path)

        # 创建划分后的目录结构
        if os.path.exists(os.path.join(root_path, 'images\\train')):
            shutil.rmtree(os.path.join(root_path, 'images\\train'))
        if os.path.exists(os.path.join(root_path, 'labels\\train')):
            shutil.rmtree(os.path.join(root_path, 'labels\\train'))
        if os.path.exists(os.path.join(root_path, 'images\\val')):
            shutil.rmtree(os.path.join(root_path, 'images\\val'))
        if os.path.exists(os.path.join(root_path, 'labels\\val')):
            shutil.rmtree(os.path.join(root_path, 'labels\\val'))
        os.makedirs(os.path.join(root_path, 'images\\train'), exist_ok=False)
        os.makedirs(os.path.join(root_path, 'labels\\train'), exist_ok=False)
        os.makedirs(os.path.join(root_path, 'images\\val'), exist_ok=False)
        os.makedirs(os.path.join(root_path, 'labels\\val'), exist_ok=False)

        # 获取所有有标注图片文件的文件名
        label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]  # 假设标注文件格式为 txt
        image_files = [f.replace('.txt', '.jpg') for f in label_files]  # 假设图片格式为 jpg
        
        # print("label_files: ", label_files)
        # print("image_files: ", image_files)
        
        # 随机打乱文件列表
        combined = list(zip(image_files, label_files))
        random.shuffle(combined)

        # 定义划分比例（例如，90%用于训练，10%用于验证）
        train_ratio = 1 - val_ratio
        train_size = int(train_ratio * len(combined))

        # 划分数据集
        train_data = combined[:train_size]
        val_data = combined[train_size:]

        # 复制文件到相应的目录
        for img_file, label_file in train_data:
            # print(os.path.join(images_path, img_file), os.path.join(root_path, 'images\\train', img_file))
            shutil.copy(os.path.join(images_path, img_file), os.path.join(root_path, 'images\\train', img_file))
            # print(os.path.join(labels_path, label_file), os.path.join(root_path, 'labels\\train', label_file))
            shutil.copy(os.path.join(labels_path, label_file), os.path.join(root_path, 'labels\\train', label_file))

        for img_file, label_file in val_data:
            print(os.path.join(images_path, img_file), os.path.join(root_path, 'images\\val', img_file))
            shutil.copy(os.path.join(images_path, img_file), os.path.join(root_path, 'images\\val', img_file))
            print(os.path.join(labels_path, label_file), os.path.join(root_path, 'labels\\val', label_file))
            shutil.copy(os.path.join(labels_path, label_file), os.path.join(root_path, 'labels\\val', label_file))
            
        return 'images\\train', 'images\\val'

    def run(self):
        try:
            self.log_signal.emit("开始训练...")
            # 获取参数
            version = self.params['version']
            model_type = self.params['model_type']
            pretrained_model = self.params['pretrained_model']
            device = self.params['device']
            batch_size = self.params['batch_size']
            epochs = self.params['epochs']
            degree = self.params['degree']
            scale = self.params['scale']
            imgsz = self.params['imgsz']
            p2 = self.params['p2']
            workers = self.params['workers']
            patience = self.params['patience']
            train_type_combo = self.params['train_type_combo']
            base_path = self.params['base_path']
            folder_name = self.params['folder_name']
            image_path = self.params['image_path']
                
            print("GPU 可用？ ", torch.cuda.is_available())
            # 加载 YOLO 模型
            self.log_signal.emit("加载模型...")
            if pretrained_model == "":
                if train_type_combo == "多边形Segment训练(分割模型)":
                    # print(f'./training/segment/cfg/yolov{version}{model_type}-seg{p2}.yaml')
                    model = YOLO(f'./yolo_cfgs/segment/yolov{version}{model_type}-seg{p2}.yaml')
                elif train_type_combo == "矩形detect训练(检测模型)":
                    # print(f'./training/detect/cfg/yolov{version}{model_type}-det{p2}.yaml')
                    model = YOLO(f'./yolo_cfgs/segment/yolov{version}{model_type}-det{p2}.yaml')
            else:
                model = YOLO(pretrained_model)
            self.log_signal.emit("成功加载模型...")
                
            self.log_signal.emit("加载数据...")
            # 获取所有标签
            unique_labels, labels_dict = self.find_unique_labels_in_directory(self.params['annotation_path'])
            # 将 json 文件转换为 txt 文件
            self.convert_json2txt(anno_dir=os.path.dirname(self.params['annotation_path']), 
                                  image_dir=os.path.dirname(self.params['image_path']), 
                                  class_map=labels_dict)
            cfg_path = os.path.join("./cfgs", f"{os.path.basename(base_path)}.yaml")
            
            if self.params['val']:
                train_folder, val_folder = self.split_train_val(self.params['image_path'])
            else:
                train_folder = "images\\" + os.path.basename(self.params['image_path'])
                val_folder = train_folder
            
            print(f"train_folder: {train_folder}, val_folder: {val_folder}")
            self.generate_dataset_yaml(dataset_path=os.path.dirname(os.path.dirname(image_path)),
                                        train_folder=train_folder,
                                        val_folder=val_folder,  # 没有验证集
                                        labels_dict=labels_dict,
                                        output_file=cfg_path)
            self.log_signal.emit("成功加载数据...")

            # 自定义训练过程的回调函数，用于更新进度
            def on_train_epoch_start(trainer):
                progress = int((trainer.epoch / trainer.args.epochs) * 100)
                self.progress_signal.emit(progress)
                # 获取训练指标
                metrics = trainer.metrics
                # print(metrics)
                # print(trainer)
                current_epoch = trainer.epoch
                total_epochs = trainer.args.epochs
                current_batch = trainer.args.batch
                box_loss = metrics['val/box_loss'] if 'val/box_loss' in metrics else 'N/A'
                mask_loss = metrics['val/seg_loss'] if 'val/seg_loss' in metrics else 'N/A'
                box_map50 = metrics['metrics/mAP50(B)'] if 'metrics/mAP50(B)' in metrics else 'N/A'
                mask_map50 = metrics['metrics/mAP50(M)'] if 'metrics/mAP50(M)' in metrics else 'N/A'

                # 获取显存占用（单位：MB）
                if torch.cuda.is_available():
                    memory = get_gpu_memory()
                else:
                    memory = "N/A"

                # 更新日志
                log = (f"Epoch: {current_epoch}/{total_epochs}, Memory: {memory}MB, "
                       f"Batch: {current_batch}, "
                       f"Loss(Box): {round(box_loss, 2)}, Loss(Mask): {round(mask_loss, 2)}, "
                       f"mAP50(Box): {round(box_map50, 2)}, mAP50(Mask): {round(mask_map50, 2)}")
                
                self.log_signal.emit(log)

            self.log_signal.emit("启动训练...")
            # 添加回调函数
            model.add_callback("on_train_epoch_start", on_train_epoch_start)
            
            # 开始训练
            results = model.train(
                data=cfg_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                degrees=degree,
                scale=scale,
                device=device,
                workers=workers,
                patience=patience,
                project=base_path,  # 设置输出文件夹的路径
                name=folder_name,  # 设置输出文件夹的名称
            )

            self.progress_signal.emit(100)  # 训练完成，设置进度条为 100%
            self.log_signal.emit("训练完成！")
            self.botton.setText("训练完成！")
            if os.path.exists(train_folder) and self.params['val']:
                os.remove(train_folder)
            if os.path.exists(val_folder) and self.params['val']:
                os.remove(val_folder)

        except Exception as e:
            self.log_signal.emit(f"训练失败：{str(e)}")  
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_dark_theme(app)
    apply_stylesheet(app)
    # 设置全局默认字体
    font = QFont("微软雅黑", 10)  # 字体名称为微软雅黑，大小为16
    app.setFont(font)
    window = TrainingInterface()
    window.show()
    sys.exit(app.exec_())
    