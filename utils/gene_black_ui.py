import sys
import os
import json
import random
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, 
                             QFileDialog, QMessageBox, QScrollArea, QListWidget, QListWidgetItem,
                             QCheckBox)
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor

class Rectangle:
    def __init__(self, x1, y1, x2, y2, is_manual=True):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.selected = False
        self.is_manual = is_manual  # 标记是否为手动绘制的矩形
    
    def contains_point(self, point):
        return (self.x1 <= point.x() <= self.x2 and 
                self.y1 <= point.y() <= self.y2)
    
    def to_dict(self):
        # 转换为JSON格式
        return {
            "label": "bad",
            "points": [
                [self.x1, self.y1],
                [self.x2, self.y2]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None
        }
    
    def get_rect(self):
        return QRect(int(self.x1), int(self.y1), int(self.x2 - self.x1), int(self.y2 - self.y1))
    
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.display_image = None
        self.original_image = None  # 保存原始图像
        self.manual_rectangles = []  # 手动绘制的矩形
        self.annotation_rectangles = []  # 从JSON加载的标注矩形
        self.current_rect = None
        self.drawing = False
        self.start_point = None
        self.selected_rect_index = -1
        self.show_annotations = True  # 是否显示标注矩形
        
        # 图像显示相关
        self.image_scale = 1.0
        self.image_offset = QPoint(0, 0)
        self.scaled_width = 0
        self.scaled_height = 0
        
        self.setMinimumSize(800, 600)
        
    def load_image(self, image_path):
        try:
            self.image = cv2.imread(image_path)
            if self.image is not None and self.image.size > 0:
                self.original_image = self.image.copy()  # 保存原始图像
                self.update_display_image()
                self.manual_rectangles = []  # 只清空手动矩形
                self.selected_rect_index = -1
                self.update()
                return True
            else:
                print(f"无法加载图像: {image_path}")
                return False
        except Exception as e:
            print(f"加载图像时出错: {str(e)}")
            return False
    
    def update_display_image(self):
        if self.image is not None:
            height, width = self.image.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.display_image = QPixmap.fromImage(q_image)
            
            # 计算缩放比例和偏移，使图像居中显示
            self.calculate_image_position()
            self.update()
    
    def calculate_image_position(self):
        if self.display_image is None:
            return
            
        img_width = self.display_image.width()
        img_height = self.display_image.height()
        
        # 计算缩放比例，使图像适应窗口
        scale_x = self.width() / img_width
        scale_y = self.height() / img_height
        self.image_scale = min(scale_x, scale_y, 1.0)  # 不超过原始大小
        
        # 计算缩放后的尺寸
        self.scaled_width = int(img_width * self.image_scale)
        self.scaled_height = int(img_height * self.image_scale)
        
        # 计算偏移使图像居中
        self.image_offset = QPoint(
            (self.width() - self.scaled_width) // 2,
            (self.height() - self.scaled_height) // 2
        )
    
    def resizeEvent(self, event):
        self.calculate_image_position()
        super().resizeEvent(event)
    
    def widget_to_image_coords(self, point):
        """将窗口坐标转换为图像坐标"""
        if self.image_scale == 0:
            return point
            
        x = (point.x() - self.image_offset.x()) / self.image_scale
        y = (point.y() - self.image_offset.y()) / self.image_scale
        
        # 确保坐标在图像范围内
        if self.image is not None:
            x = max(0, min(x, self.image.shape[1] - 1))
            y = max(0, min(y, self.image.shape[0] - 1))
        
        return QPoint(int(x), int(y))
    
    def image_to_widget_coords(self, point):
        """将图像坐标转换为窗口坐标"""
        x = point.x() * self.image_scale + self.image_offset.x()
        y = point.y() * self.image_scale + self.image_offset.y()
        return QPoint(int(x), int(y))
    
    def image_rect_to_widget_rect(self, rect):
        """将图像中的矩形转换为窗口中的矩形"""
        top_left = self.image_to_widget_coords(QPoint(int(rect.x1), int(rect.y1)))
        bottom_right = self.image_to_widget_coords(QPoint(int(rect.x2), int(rect.y2)))
        return QRect(top_left, bottom_right)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), Qt.lightGray)
        
        if self.display_image:
            # 绘制缩放后的图像
            painter.drawPixmap(
                self.image_offset.x(), 
                self.image_offset.y(),
                self.scaled_width,
                self.scaled_height,
                self.display_image
            )
            
            # 绘制手动矩形
            for i, rect in enumerate(self.manual_rectangles):
                if rect.selected:
                    pen = QPen(QColor(255, 0, 0), 3)  # 红色边框表示选中
                else:
                    pen = QPen(QColor(0, 255, 0), 2)  # 绿色边框表示手动矩形
                painter.setPen(pen)
                
                # 将图像坐标转换为窗口坐标
                widget_rect = self.image_rect_to_widget_rect(rect)
                painter.drawRect(widget_rect)
                
                # 显示矩形信息
                info_text = f"手动矩形 {i+1}: ({rect.x1}, {rect.y1}) - ({rect.x2}, {rect.y2})"
                painter.setPen(QPen(Qt.white))
                painter.drawText(widget_rect.topLeft() + QPoint(0, -5), info_text)
            
            # 绘制标注矩形（如果启用显示）
            if self.show_annotations:
                for i, rect in enumerate(self.annotation_rectangles):
                    pen = QPen(QColor(255, 255, 0), 2)  # 黄色边框表示标注矩形
                    painter.setPen(pen)
                    
                    # 将图像坐标转换为窗口坐标
                    widget_rect = self.image_rect_to_widget_rect(rect)
                    painter.drawRect(widget_rect)
                    
                    # 显示矩形信息
                    info_text = f"标注 {i+1}: ({rect.x1}, {rect.y1}) - ({rect.x2}, {rect.y2})"
                    painter.setPen(QPen(Qt.white))
                    painter.drawText(widget_rect.topLeft() + QPoint(0, -10), info_text)
            
            # 绘制当前正在绘制的矩形
            if self.drawing and self.current_rect:
                pen = QPen(QColor(0, 255, 255), 2)  # 青色边框表示正在绘制
                painter.setPen(pen)
                
                # 将图像坐标转换为窗口坐标
                widget_rect = self.image_rect_to_widget_rect(self.current_rect)
                painter.drawRect(widget_rect)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 将窗口坐标转换为图像坐标
            image_pos = self.widget_to_image_coords(event.pos())
            
            # 检查坐标是否在图像范围内
            if self.image is None or (
                image_pos.x() < 0 or image_pos.y() < 0 or 
                image_pos.x() >= self.image.shape[1] or 
                image_pos.y() >= self.image.shape[0]):
                return
            
            # 检查是否点击了已有的手动矩形
            self.selected_rect_index = -1
            for i, rect in enumerate(self.manual_rectangles):
                if rect.contains_point(image_pos):
                    rect.selected = True
                    self.selected_rect_index = i
                else:
                    rect.selected = False
            
            # 如果没有点击到矩形，开始绘制新矩形
            if self.selected_rect_index == -1:
                self.drawing = True
                self.start_point = image_pos
                self.current_rect = Rectangle(
                    image_pos.x(), image_pos.y(), 
                    image_pos.x(), image_pos.y()
                )
            
            self.update()
    
    def mouseMoveEvent(self, event):
        if self.drawing and self.start_point:
            # 将窗口坐标转换为图像坐标
            current_pos = self.widget_to_image_coords(event.pos())
            
            # 限制在图像范围内
            if self.image is not None:
                current_pos.setX(max(0, min(current_pos.x(), self.image.shape[1] - 1)))
                current_pos.setY(max(0, min(current_pos.y(), self.image.shape[0] - 1)))
            
            self.current_rect = Rectangle(
                self.start_point.x(), self.start_point.y(), 
                current_pos.x(), current_pos.y()
            )
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.current_rect and self.current_rect.area() > 3:  # 最小面积限制
                self.manual_rectangles.append(self.current_rect)
                # 选中新添加的矩形
                if self.manual_rectangles:
                    for rect in self.manual_rectangles:
                        rect.selected = False
                    self.manual_rectangles[-1].selected = True
                    self.selected_rect_index = len(self.manual_rectangles) - 1
            self.current_rect = None
            self.update()
    
    def add_dirt_to_rect(self, rect_index, density, min_gray, max_gray):
        if rect_index < 0 or rect_index >= len(self.manual_rectangles) or self.image is None:
            return False
        
        rect = self.manual_rectangles[rect_index]
        x1, y1, x2, y2 = rect.x1, rect.y1, rect.x2, rect.y2
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, self.image.shape[1]))
        y1 = max(0, min(y1, self.image.shape[0]))
        x2 = max(0, min(x2, self.image.shape[1]))
        y2 = max(0, min(y2, self.image.shape[0]))
        
        if x1 >= x2 or y1 >= y2:
            return False
        
        # 计算要添加的黑点数量
        area = (x2 - x1) * (y2 - y1)
        num_dots = int(area * density / 100.0)
        
        for _ in range(num_dots):
            # 随机位置
            x = random.randint(int(x1), int(x2) - 1)
            y = random.randint(int(y1), int(y2) - 1)
            
            # 随机灰度值
            gray_value = random.randint(min_gray, max_gray)
            
            # 添加黑点（小矩形模拟脏污）
            size = random.randint(1, 3)
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.image.shape[1] and 0 <= ny < self.image.shape[0] and
                        x1 <= nx <= x2 and y1 <= ny <= y2):
                        if random.random() < 0.7:  # 70%的概率设置像素
                            self.image[ny, nx] = [gray_value, gray_value, gray_value]
        
        self.update_display_image()
        return True
    
    def add_dirt_to_all_rects(self, density, min_gray, max_gray):
        if self.image is None:
            return False
        
        success = True
        for i in range(len(self.manual_rectangles)):
            if not self.add_dirt_to_rect(i, density, min_gray, max_gray):
                success = False
        
        return success
    
    def save_image(self, file_path):
        if self.image is not None:
            cv2.imwrite(file_path, self.image)
            return True
        return False
    
    def delete_selected_rect(self):
        if self.selected_rect_index >= 0 and self.selected_rect_index < len(self.manual_rectangles):
            del self.manual_rectangles[self.selected_rect_index]
            self.selected_rect_index = -1
            self.update()
            return True
        return False
    
    def toggle_annotations(self, show):
        self.show_annotations = show
        self.update()

class DirtGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_index = 0
        self.image_files = []
        self.annotation_files = {}
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("黑色脏污生成工具")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)
        
        # 文件夹选择区域
        folder_group = QGroupBox("文件夹设置")
        folder_layout = QVBoxLayout(folder_group)
        
        self.input_folder_btn = QPushButton("选择图片文件夹")
        self.input_folder_btn.clicked.connect(self.select_input_folder)
        folder_layout.addWidget(self.input_folder_btn)
        
        self.annotation_folder_btn = QPushButton("选择标注文件夹")
        self.annotation_folder_btn.clicked.connect(self.select_annotation_folder)
        folder_layout.addWidget(self.annotation_folder_btn)
        
        self.output_folder_btn = QPushButton("选择输出文件夹")
        self.output_folder_btn.clicked.connect(self.select_output_folder)
        folder_layout.addWidget(self.output_folder_btn)
        
        control_layout.addWidget(folder_group)
        
        # 图片导航
        nav_group = QGroupBox("图片导航")
        nav_layout = QHBoxLayout(nav_group)
        
        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self.previous_image)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self.next_image)
        nav_layout.addWidget(self.next_btn)
        
        control_layout.addWidget(nav_group)
        
        # 参数设置
        param_group = QGroupBox("参数设置")
        param_layout = QVBoxLayout(param_group)
        
        # 密度设置
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("密度(%):"))
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.1, 50.0)
        self.density_spin.setValue(20.0)
        self.density_spin.setSingleStep(0.5)
        density_layout.addWidget(self.density_spin)
        param_layout.addLayout(density_layout)
        
        # 灰度范围设置
        gray_layout1 = QHBoxLayout()
        gray_layout1.addWidget(QLabel("最小灰度:"))
        self.min_gray_spin = QSpinBox()
        self.min_gray_spin.setRange(0, 255)
        self.min_gray_spin.setValue(0)
        gray_layout1.addWidget(self.min_gray_spin)
        param_layout.addLayout(gray_layout1)
        
        gray_layout2 = QHBoxLayout()
        gray_layout2.addWidget(QLabel("最大灰度:"))
        self.max_gray_spin = QSpinBox()
        self.max_gray_spin.setRange(0, 255)
        self.max_gray_spin.setValue(128)
        gray_layout2.addWidget(self.max_gray_spin)
        param_layout.addLayout(gray_layout2)
        
        control_layout.addWidget(param_group)
        
        # 显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout(display_group)
        
        self.show_annotations_check = QCheckBox("显示JSON标注")
        self.show_annotations_check.setChecked(True)
        self.show_annotations_check.stateChanged.connect(self.toggle_annotations_display)
        display_layout.addWidget(self.show_annotations_check)
        
        control_layout.addWidget(display_group)
        
        # 操作按钮
        self.single_generate_btn = QPushButton("单个生成")
        self.single_generate_btn.clicked.connect(self.single_generate)
        control_layout.addWidget(self.single_generate_btn)
        
        self.all_generate_btn = QPushButton("全部生成")
        self.all_generate_btn.clicked.connect(self.all_generate)
        control_layout.addWidget(self.all_generate_btn)
        
        self.delete_rect_btn = QPushButton("删除选中矩形")
        self.delete_rect_btn.clicked.connect(self.delete_selected_rect)
        control_layout.addWidget(self.delete_rect_btn)
        
        self.save_btn = QPushButton("保存图片和标注")
        self.save_btn.clicked.connect(self.save_image_and_annotation)
        control_layout.addWidget(self.save_btn)
        
        # 矩形列表
        rect_group = QGroupBox("手动矩形区域")
        rect_layout = QVBoxLayout(rect_group)
        self.rect_list = QListWidget()
        self.rect_list.itemClicked.connect(self.on_rect_selected)
        rect_layout.addWidget(self.rect_list)
        control_layout.addWidget(rect_group)
        
        control_layout.addStretch()
        
        main_layout.addWidget(control_panel)
        
        # 右侧图片显示区域
        self.image_viewer = ImageViewer()
        main_layout.addWidget(self.image_viewer)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
        # 初始化文件夹路径
        self.input_folder = ""
        self.annotation_folder = ""
        self.output_folder = ""
    
    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.input_folder = folder
            self.load_image_files()
            self.statusBar().showMessage(f"图片文件夹: {folder}")
    
    def select_annotation_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择标注文件夹")
        if folder:
            self.annotation_folder = folder
            self.statusBar().showMessage(f"标注文件夹: {folder}")
    
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_folder = folder
            self.statusBar().showMessage(f"输出文件夹: {folder}")
    
    def load_image_files(self):
        if not self.input_folder:
            return
        
        self.image_files = []
        for file in os.listdir(self.input_folder):
            if file.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png', '.tiff')):
                self.image_files.append(file)
        
        self.image_files.sort()
        self.current_image_index = 0
        
        if self.image_files:
            self.load_current_image()
        else:
            self.statusBar().showMessage("输入文件夹中没有找到图片文件")
    
    def load_current_image(self):
        if not self.image_files or self.current_image_index >= len(self.image_files):
            return
        
        image_path = os.path.join(self.input_folder, self.image_files[self.current_image_index])
        
        # 检查图像是否成功加载
        if not self.image_viewer.load_image(image_path):
            QMessageBox.warning(self, "警告", f"无法加载图像: {self.image_files[self.current_image_index]}")
            return
        
        # 加载对应的标注文件（仅用于显示，不用于编辑）
        self.load_annotation_for_display()
        
        # 更新矩形列表
        self.update_rect_list()
        
        self.statusBar().showMessage(f"当前图片: {self.image_files[self.current_image_index]} ({self.current_image_index + 1}/{len(self.image_files)})")
    
    def load_annotation_for_display(self):
        """加载标注文件，仅用于显示，不用于编辑"""
        if not self.annotation_folder or not self.image_files:
            return
        
        image_file = self.image_files[self.current_image_index]
        base_name = os.path.splitext(image_file)[0]
        json_file = base_name + '.json'
        json_path = os.path.join(self.annotation_folder, json_file)
        
        self.image_viewer.annotation_rectangles = []
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                
                for shape in annotation_data.get('shapes', []):
                    if shape.get('shape_type') == 'rectangle':
                        points = shape.get('points', [])
                        if len(points) == 2:
                            x1, y1 = points[0]
                            x2, y2 = points[1]
                            rect = Rectangle(int(x1), int(y1), int(x2), int(y2), is_manual=False)
                            self.image_viewer.annotation_rectangles.append(rect)
                
                self.image_viewer.update()
                
            except Exception as e:
                print(f"加载标注文件失败: {str(e)}")
    
    def save_annotation(self):
        """保存标注文件，只包含手动绘制的矩形"""
        if not self.annotation_folder or not self.image_files or not self.output_folder:
            return False
        
        image_file = self.image_files[self.current_image_index]
        base_name = os.path.splitext(image_file)[0]
        json_file = base_name + '.json'
        json_path = os.path.join(self.output_folder, json_file)
        
        # 创建基础标注结构
        if self.image_viewer.image is not None and hasattr(self.image_viewer.image, 'shape'):
            image_height = self.image_viewer.image.shape[0]
            image_width = self.image_viewer.image.shape[1]
        else:
            image_height = 480
            image_width = 640
        
        annotation_data = {
            "version": "5.6.0a0",
            "flags": {},
            "shapes": [],
            "imagePath": image_file,
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width,
            "verified": False
        }
        
        # 添加所有手动矩形
        for rect in self.image_viewer.manual_rectangles:
            annotation_data["shapes"].append(rect.to_dict())
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            QMessageBox.warning(self, "警告", f"保存标注文件失败: {str(e)}")
            return False
    
    def update_rect_list(self):
        self.rect_list.clear()
        for i, rect in enumerate(self.image_viewer.manual_rectangles):
            item = QListWidgetItem(f"矩形 {i+1}: ({rect.x1}, {rect.y1}) - ({rect.x2}, {rect.y2})")
            if rect.selected:
                item.setBackground(QColor(200, 200, 255))
            self.rect_list.addItem(item)
    
    def on_rect_selected(self, item):
        index = self.rect_list.row(item)
        for i, rect in enumerate(self.image_viewer.manual_rectangles):
            rect.selected = (i == index)
        self.image_viewer.selected_rect_index = index
        self.image_viewer.update()
        self.update_rect_list()
    
    def toggle_annotations_display(self, state):
        show = state == Qt.Checked
        self.image_viewer.toggle_annotations(show)
    
    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
    
    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
    
    def single_generate(self):
        if self.image_viewer.selected_rect_index == -1:
            QMessageBox.warning(self, "警告", "请先选择一个矩形区域")
            return
        
        density = self.density_spin.value() / 100.0
        min_gray = self.min_gray_spin.value()
        max_gray = self.max_gray_spin.value()
        
        if min_gray > max_gray:
            QMessageBox.warning(self, "警告", "最小灰度不能大于最大灰度")
            return
        
        success = self.image_viewer.add_dirt_to_rect(
            self.image_viewer.selected_rect_index, 
            density, min_gray, max_gray
        )
        
        if success:
            self.statusBar().showMessage("脏污生成成功")
        else:
            QMessageBox.warning(self, "警告", "脏污生成失败")
    
    def all_generate(self):
        if not self.image_viewer.manual_rectangles:
            QMessageBox.warning(self, "警告", "请先创建矩形区域")
            return
        
        density = self.density_spin.value() / 100.0
        min_gray = self.min_gray_spin.value()
        max_gray = self.max_gray_spin.value()
        
        if min_gray > max_gray:
            QMessageBox.warning(self, "警告", "最小灰度不能大于最大灰度")
            return
        
        success = self.image_viewer.add_dirt_to_all_rects(density, min_gray, max_gray)
        
        if success:
            self.statusBar().showMessage("所有矩形区域脏污生成成功")
        else:
            QMessageBox.warning(self, "警告", "脏污生成失败")
    
    def delete_selected_rect(self):
        if self.image_viewer.delete_selected_rect():
            self.update_rect_list()
            self.statusBar().showMessage("已删除选中矩形")
        else:
            QMessageBox.warning(self, "警告", "请先选择一个矩形")
    
    def save_image_and_annotation(self):
        if not self.output_folder:
            QMessageBox.warning(self, "警告", "请先选择输出文件夹")
            return
        
        if not self.image_files:
            QMessageBox.warning(self, "警告", "没有可保存的图片")
            return
        
        # 保存图片
        image_file = self.image_files[self.current_image_index]
        output_path = os.path.join(self.output_folder, image_file)
        
        if self.image_viewer.save_image(output_path):
            # 保存标注
            if self.save_annotation():
                self.statusBar().showMessage(f"图片和标注已保存到: {output_path}")
            else:
                self.statusBar().showMessage(f"图片已保存，但标注保存失败: {output_path}")
        else:
            QMessageBox.warning(self, "警告", "图片保存失败")

def main():
    app = QApplication(sys.argv)
    window = DirtGenerator()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()