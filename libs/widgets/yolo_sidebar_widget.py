from qtpy import QtWidgets
from qtpy.QtCore import Qt
import yaml

class YoloSidebarWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # YOLO模型路径（带文件选择按钮）
        model_path_layout = QtWidgets.QHBoxLayout()
        self.model_path_edit = QtWidgets.QLineEdit()
        self.model_path_edit.setPlaceholderText('YOLO模型路径')
        self.model_path_btn = QtWidgets.QPushButton('...')
        self.model_path_btn.setFixedWidth(30)
        self.model_path_btn.clicked.connect(self.choose_model_path)
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(self.model_path_btn)
        layout.addWidget(QtWidgets.QLabel('YOLO模型路径:'))
        layout.addLayout(model_path_layout)

        # 数据集yaml文件路径（带文件选择按钮）
        yaml_path_layout = QtWidgets.QHBoxLayout()
        self.yaml_path_edit = QtWidgets.QLineEdit()
        self.yaml_path_edit.setPlaceholderText('数据集yaml文件路径')
        self.yaml_path_btn = QtWidgets.QPushButton('...')
        self.yaml_path_btn.setFixedWidth(30)
        self.yaml_path_btn.clicked.connect(self.choose_yaml_path)
        yaml_path_layout.addWidget(self.yaml_path_edit)
        yaml_path_layout.addWidget(self.yaml_path_btn)
        layout.addWidget(QtWidgets.QLabel('数据集yaml文件路径:'))
        layout.addLayout(yaml_path_layout)
        
        # 展示yaml文件中的类别
        self.class_list = QtWidgets.QListWidget()
        layout.addWidget(QtWidgets.QLabel('类别列表:'))
        layout.addWidget(self.class_list)

        # 加载yaml按钮
        self.load_yaml_btn = QtWidgets.QPushButton('加载yaml')
        self.load_yaml_btn.clicked.connect(self.load_yaml)
        layout.addWidget(self.load_yaml_btn)
        
        # imgsz参数
        self.imgsz_spin = QtWidgets.QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        layout.addWidget(QtWidgets.QLabel('imgsz（输入尺寸）:'))
        layout.addWidget(self.imgsz_spin)

        # conf参数
        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0, 1)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.25)
        layout.addWidget(QtWidgets.QLabel('conf阈值:'))
        layout.addWidget(self.conf_spin)

        # iou参数
        self.iou_spin = QtWidgets.QDoubleSpinBox()
        self.iou_spin.setRange(0, 1)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setValue(0.45)
        layout.addWidget(QtWidgets.QLabel('iou阈值:'))
        layout.addWidget(self.iou_spin)
        

        # 是否使用Simplify
        simplify_layout = QtWidgets.QHBoxLayout()
        self.simplify_checkbox = QtWidgets.QCheckBox('使用Simplify稀疏多边形')
        self.simplify_checkbox.setChecked(False)
        simplify_layout.addWidget(self.simplify_checkbox)
        layout.addLayout(simplify_layout)

        # Simplify容差值
        tolerance_layout = QtWidgets.QHBoxLayout()
        self.tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.tolerance_spin.setRange(0.01, 100.0)
        self.tolerance_spin.setSingleStep(0.01)
        self.tolerance_spin.setValue(0.5)
        self.tolerance_spin.setDecimals(3)
        self.tolerance_spin.setEnabled(False)
        tolerance_layout.addWidget(QtWidgets.QLabel('Simplify容差值:'))
        tolerance_layout.addWidget(self.tolerance_spin)
        layout.addLayout(tolerance_layout)

        # 复选框控制容差输入框启用
        self.simplify_checkbox.stateChanged.connect(lambda state: self.tolerance_spin.setEnabled(state == Qt.Checked))


        self.setLayout(layout)

    def get_simplify(self) -> bool:
        """是否使用Simplify"""
        return self.simplify_checkbox.isChecked()

    def get_tolerance(self) -> float:
        """获取Simplify容差值"""
        return float(self.tolerance_spin.value())

    def get_imgsz(self) -> int:
        """获取imgsz参数"""
        return int(self.imgsz_spin.value())
        
    def choose_model_path(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择YOLO模型文件', '', 'PyTorch模型 (*.pt);;所有文件 (*)')
        if path:
            self.model_path_edit.setText(path)

    def choose_yaml_path(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择数据集yaml文件', '', 'YAML文件 (*.yaml *.yml);;所有文件 (*)')
        if path:
            self.yaml_path_edit.setText(path)

    def load_yaml(self):
        yaml_path = self.yaml_path_edit.text()
        self.class_list.clear()
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                names = data.get('names') or data.get('class_names') or []
                if isinstance(names, dict):
                    names = list(names.values())
                for name in names:
                    item = QtWidgets.QListWidgetItem(str(name))
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    self.class_list.addItem(item)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, '加载失败', f'无法加载yaml文件: {e}')

    def get_selected_class(self):
        """获取选中的类别标签列表"""
        selected = []
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected

    def get_iou_threshold(self) -> float:
        """返回当前设置的iou阈值"""
        return float(self.iou_spin.value())

    def get_score_threshold(self) -> float:
        """返回当前设置的conf阈值"""
        return float(self.conf_spin.value())

