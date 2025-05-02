# -*- coding: utf-8 -*-

import functools
import html
import math
import os
import os.path as osp
import re
import sys
import webbrowser

import imgviz
import natsort
import numpy as np
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import QChart, QChartView, QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis

import cv2
import yaml
import torch
import random
from copy import deepcopy
import xml.etree.ElementTree as ET
from skimage import exposure
from strsimpy.jaro_winkler import JaroWinkler
from ultralytics import YOLO

import json
import base64
from PIL import Image
from io import BytesIO
from shapely.geometry import Polygon


def newIcon(icon):
    return QIcon(':/' + icon)

sys.path.append('libs/')
from libs import PY2
from libs import __appname__
from libs import ai
from libs.ai import MODELS
from libs.config import get_config
from libs.label_file import LabelFile
from libs.label_file import LabelFileError
from libs.logger import logger
from libs.shape import Shape
from libs.widgets import AiPromptWidget
from libs.widgets import BrightnessContrastDialog
from libs.widgets import Canvas
from libs.widgets import FileDialogPreview
from libs.widgets import LabelDialog
from libs.widgets import LabelListWidget
from libs.widgets import LabelListWidgetItem
from libs.widgets import ToolBar
from libs.widgets import UniqueLabelQListWidget
from libs.widgets import ZoomWidget

from libs import utils
from libs.utils.datasets import *
from libs.utils import torch_utils
from app_train import TrainingInterface, set_dark_theme, apply_stylesheet, MultiChoiceDialog

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".


LABEL_COLORMAP = imgviz.label_colormap()

def ustr(x):
    '''py2/py3 unicode helper'''

    if sys.version_info < (3, 0, 0):
        from PyQt4.QtCore import QString
        if type(x) == str:
            return x.decode(DEFAULT_ENCODING)
        if type(x) == QString:
            #https://blog.csdn.net/friendan/article/details/51088476
            #https://blog.csdn.net/xxm524/article/details/74937308
            return unicode(x.toUtf8(), DEFAULT_ENCODING, 'ignore')
        return x
    else:
        return x
       
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    
class DataViewer(QWidget):
    def __init__(self, result, parent=None):
        super(DataViewer, self).__init__(parent)
        self.initUI(result)
 
    def initUI(self, result):
        self.setWindowTitle('Data Information and Histogram')
        
        # Create a vertical layout
        layout = QVBoxLayout()
 
        # Create a label for the text information
        info_text = 'label | pic_num | box_num \n'
        info_text += '----------------------------\n'
        for key in result.keys():
            info_text += '{}:  {}\n'.format(key, result[key])
        info_label = QLabel(info_text)
        layout.addWidget(info_label)
 
        # Create a canvas for the histogram
        self.canvas = FigureCanvas(plt.figure())
        self.ax = self.canvas.figure.add_subplot(111)
        
        # Prepare data for the histogram
        classes, pic_nums, box_nums = zip(*[(key, val[0], val[1]) for key, val in result.items() if key != 'total'])
        self.ax.bar(classes, box_nums, label='Box Count')
        self.ax.set_xlabel('Class Label')
        self.ax.set_ylabel('Count')
        self.ax.set_title('Box Distribution per Class')
        self.ax.legend()
 
        # Add the canvas to the layout
        layout.addWidget(self.canvas)
 
        # Add a button to close the window
        close_button = QPushButton('Close')
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
 
        # Set the layout for the window
        self.setLayout(layout)
 
        # Draw the plot
        self.canvas.draw()

class MainWindow(QtWidgets.QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        if output is not None:
            logger.warning("argument output is deprecated, use output_file instead")
            if output_file is None:
                output_file = output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False

        self._copied_shapes = None

        # For loading all image under a directory
        self.mImgList = []

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )

        self.labelList = LabelListWidget()
        self.lastOpenDir = None

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self.loadFlags({k: False for k in config["flags"]})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)

        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self._edit_label)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Polygon Labels"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr(
                "Select label to start annotating for it. " "Press 'Esc' to deselect."
            )
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(self.uniqLabelList)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.mouseMoved.connect(
            lambda pos: self.status(f"Mouse is at: x={pos.x()}, y={pos.y()}")
        )

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scrollArea)

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)

        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        
        # 训练模型 action：
        # 选择任务、选择模型、选择数据集 yaml 文件、设置训练集\验证集比例、设置 batch_size、是否从头训练
        train_with_labels = action('train_with_labels', self.train_with_labels, 'Alt+F1', 'train model with already labeled data')
        analyse_result = action('analyse_result', self.analyse_result, 'Alt+F2', 'analyse training result')
        delete_unchecked = action('delete_unchecked', self.delete_unchecked, 'Alt+F1', 'delect all unchecked labels')
        
        # 自动标注 action：
        search_system = action('Search_System', self.search_actions_info, None, 'zoom-in')
        batch_rename_img = action('batch_rename_img', self.batch_rename_img, 'Ctrl+1', 'edit')
        rename_img_xml = action('rename_img_xml', self.rename_img_json, 'Ctrl+Alt+1', 'edit')
        duplicate_xml = action('duplicate_xml', self.make_duplicate_xml, 'Ctrl+2', 'copy')
        batch_duplicate = action('batch_duplicate', self.batch_duplicate_xml, 'Ctrl+Alt+2', 'copy')
        label_pruning = action('label_pruning', self.prune_useless_label, 'Ctrl+3', 'delete')
        file_pruning = action('file_pruning', self.remove_extra_img_xml, 'Ctrl+Alt+3', 'delete')
        change_label = action('change_label', self.change_label_name, 'Ctrl+4', 'color_line')
        fix_property = action('fix_property', self.fix_xml_property, 'Ctrl+5', 'color_line')
        auto_labeling = action('auto_labeling', self.auto_labeling, 'Ctrl+6', 'new')
        data_augment = action('data_augment', self.data_auto_augment, 'Ctrl+7', 'copy')
        sam_optim = action('sam_optim', self.sam_optim, 'Ctrl+8', 'sam')
        folder_info = action('folder_info', self.show_folder_infor, 'Alt+1', 'help')
        label_info = action('label_info', self.show_label_info, 'Alt+2', 'help')
        
        # 处理视频 action：
        extract_video=action('extract_video', self.extract_video,'Shift+1', 'new')
        extract_stream=action('extract_stream', self.extract_stream,'Shift+2', 'new')
        batch_resize_img=action('batch_resize_img', self.batch_resize_img,'Shift+3', 'fit-window')
        merge_video=action('merge_video', self.merge_video,'Shift+4', 'open')
        annotation_video=action('annotation_video', self.annotation_video,'Shift+5', 'new')
        
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            "quit",
            self.tr("Quit application"),
        )
        open_ = action(
            self.tr("&Open\n"),
            self.openFile,
            shortcuts["open"],
            "open",
            self.tr("Open image or label file"),
        )
        opendir = action(
            self.tr("Open Dir"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "open",
            self.tr("Open Dir"),
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        verify = action(
            self.tr("&Verify"), 
            self.verifyImg,
            shortcuts["verify"], 
            'verify', 
            self.tr('verifyImgDetail')
            )
        openPrevImg = action(
            self.tr("&Prev Image"),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        save = action(
            self.tr("&Save\n"),
            self.saveFile,
            shortcuts["save"],
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.saveAuto.setChecked(x),
            icon="save",
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text=self.tr("Save With Image Data"),
            slot=self.enableSaveImageWithData,
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            self.tr("&Close"),
            self.closeFile,
            shortcuts["close"],
            "close",
            self.tr("Close current file"),
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep previous annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        createMode = action(
            self.tr("Create Polygons"),
            lambda: self.toggleDrawMode(False, createMode="polygon"),
            shortcuts["create_polygon"],
            "objects",
            self.tr("Start drawing polygons"),
            enabled=False,
        )
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "objects",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self.toggleDrawMode(False, createMode="circle"),
            shortcuts["create_circle"],
            "objects",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self.toggleDrawMode(False, createMode="line"),
            shortcuts["create_line"],
            "objects",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            shortcuts["create_point"],
            "objects",
            self.tr("Start drawing points"),
            enabled=False,
        )
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggleDrawMode(False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "objects",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode = action(
            self.tr("Create AI-Polygon"),
            lambda: self.toggleDrawMode(False, createMode="ai_polygon"),
            None,
            "objects",
            self.tr("Start drawing ai_polygon. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode.changed.connect(
            lambda: self.canvas.initializeAiModel(
                name=self._selectAiModelComboBox.currentText()
            )
            if self.canvas.createMode == "ai_polygon"
            else None
        )
        createAiMaskMode = action(
            self.tr("Create AI-Mask"),
            lambda: self.toggleDrawMode(False, createMode="ai_mask"),
            None,
            "objects",
            self.tr("Start drawing ai_mask. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiMaskMode.changed.connect(
            lambda: self.canvas.initializeAiModel(
                name=self._selectAiModelComboBox.currentText()
            )
            if self.canvas.createMode == "ai_mask"
            else None
        )
        editMode = action(
            self.tr("Edit Polygons"),
            self.setEditMode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Polygons"),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "copy_clipboard",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Polygons"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        removePoint = action(
            text=self.tr("Remove Selected Point"),
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip=self.tr("Remove selected point from polygon"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            shortcuts["hide_all_polygons"],
            icon="eye",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            shortcuts["show_all_polygons"],
            icon="eye",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )
        toggleAll = action(
            self.tr("&Toggle\nPolygons"),
            functools.partial(self.togglePolygons, None),
            shortcuts["toggle_all_polygons"],
            icon="eye",
            tip=self.tr("Toggle all polygons"),
            enabled=False,
        )

        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="help",
            tip=self.tr("Show tutorial page"),
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoomBoxLayout = QtWidgets.QVBoxLayout()
        zoomLabel = QtWidgets.QLabel(self.tr("Zoom"))
        zoomLabel.setAlignment(Qt.AlignCenter)
        zoomBoxLayout.addWidget(zoomLabel)
        zoomBoxLayout.addWidget(self.zoomWidget)
        zoom.setDefaultWidget(QtWidgets.QWidget())
        zoom.defaultWidget().setLayout(zoomBoxLayout)
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(
                    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            functools.partial(self.addZoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            functools.partial(self.addZoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            functools.partial(self.setZoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            self.tr("&Brightness Contrast"),
            self.brightnessContrast,
            None,
            "color",
            self.tr("Adjust brightness and contrast"),
            enabled=False,
        )
        # Group zoom controls into a list for easier toggling.
        zoomActions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("&Edit Label"),
            self._edit_label,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()

        # Label list context menu.
        labelMenu = QtWidgets.QMenu()
               
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = utils.struct(
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            createAiPolygonMode=createAiPolygonMode,
            createAiMaskMode=createAiMaskMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            zoomActions=zoomActions,
            openNextImg=openNextImg,
            verify=verify,
            openPrevImg=openPrevImg,
            fileMenuActions=(open_, opendir, save, saveAs, close, quit),
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                duplicate,
                copy,
                paste,
                delete,
                None,
                undo,
                undoLastPoint,
                None,
                removePoint,
                None,
                toggle_keep_prev_mode,
            ),
            # menu shown at right click
            menu=(
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                createAiMaskMode,
                editMode,
                edit,
                duplicate,
                copy,
                paste,
                delete,
                undo,
                undoLastPoint,
                removePoint,
            ),
            onLoadActive=(
                close,
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                createAiMaskMode,
                editMode,
                brightnessContrast,
            ),
            onShapesPresent=(saveAs, hideAll, showAll, toggleAll),
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.menus = utils.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            training=self.menu(self.tr('Training-Tools')),
            annotate=self.menu(self.tr('&Annotate-Tools')),
            video=self.menu(self.tr('&Video-Tools')),
            help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
        )

        # 设置菜单栏
        utils.addActions(self.menus.training,
            (
               train_with_labels,
               analyse_result,
               delete_unchecked  
            )
        )
        utils.addActions(self.menus.annotate,
            (
                batch_rename_img, rename_img_xml,
                None,
                duplicate_xml, batch_duplicate,
                None,
                label_pruning, file_pruning, change_label, fix_property,
                None,
                auto_labeling, data_augment, sam_optim,
                None,
                folder_info, label_info
            )
        )
        utils.addActions(self.menus.video,
            (
                extract_video, extract_stream, 
                None, 
                batch_resize_img, merge_video, 
                None, 
                annotation_video
            )
        )

        utils.addActions(self.menus.file,
            (
                open_,
                openNextImg,
                openPrevImg,
                verify,
                opendir,
                self.menus.recentFiles,
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, 
                         (help,))
        utils.addActions(self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                toggleAll,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], 
                         self.actions.menu)
        utils.addActions(self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        selectAiModel = QtWidgets.QWidgetAction(self)
        selectAiModel.setDefaultWidget(QtWidgets.QWidget())
        selectAiModel.defaultWidget().setLayout(QtWidgets.QVBoxLayout())
        #
        selectAiModelLabel = QtWidgets.QLabel(self.tr("AI Mask Model"))
        selectAiModelLabel.setAlignment(QtCore.Qt.AlignCenter)
        selectAiModel.defaultWidget().layout().addWidget(selectAiModelLabel)
        #
        self._selectAiModelComboBox = QtWidgets.QComboBox()
        selectAiModel.defaultWidget().layout().addWidget(self._selectAiModelComboBox)
        model_names = [model.name for model in MODELS]
        self._selectAiModelComboBox.addItems(model_names)
        if self._config["ai"]["default"] in model_names:
            model_index = model_names.index(self._config["ai"]["default"])
        else:
            logger.warning(
                "Default AI model is not found: %r",
                self._config["ai"]["default"],
            )
            model_index = 0
        self._selectAiModelComboBox.setCurrentIndex(model_index)
        self._selectAiModelComboBox.currentIndexChanged.connect(
            lambda: self.canvas.initializeAiModel(
                name=self._selectAiModelComboBox.currentText()
            )
            if self.canvas.createMode in ["ai_polygon", "ai_mask"]
            else None
        )

        self._ai_prompt_widget: QtWidgets.QWidget = AiPromptWidget(
            on_submit=self._submit_ai_prompt, parent=self
        )
        ai_prompt_action = QtWidgets.QWidgetAction(self)
        ai_prompt_action.setDefaultWidget(self._ai_prompt_widget)

        self.tools = self.toolbar("Tools")
        self.actions.tool = (
            open_,
            opendir,
            openPrevImg,
            openNextImg,
            verify,
            save,
            deleteFile,
            None,
            createMode,
            editMode,
            duplicate,
            delete,
            undo,
            brightnessContrast,
            None,
            fitWindow,
            zoom,
            None,
            selectAiModel,
            None,
            ai_prompt_action,
        )

        self.statusBar().showMessage(str(self.tr("%s started.")) % __appname__)
        self.statusBar().show()

        if output_file is not None and self._config["auto_save"]:
            logger.warn(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.recentFiles = []
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightnessContrast_values = {}
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(state)

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()
        self.showMaximized()
        
        self.filePath = None
        self.defaultSaveDir = None
        self.training_window = None
                
        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar

    # Support Functions

    def noShapes(self):
        return not len(self.labelList)

    def populateModeActions(self):
        tool, menu = self.actions.tool, self.actions.menu
        self.tools.clear()
        utils.addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createCircleMode,
            self.actions.createLineMode,
            self.actions.createPointMode,
            self.actions.createLineStripMode,
            self.actions.createAiPolygonMode,
            self.actions.createAiMaskMode,
            self.actions.editMode,
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.saveLabels(label_file)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}*".format(title, self.filename)
        self.setWindowTitle(title)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createCircleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        self.actions.createAiPolygonMode.setEnabled(True)
        self.actions.createAiMaskMode.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}".format(title, self.filename)
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def _submit_ai_prompt(self, _) -> None:
        texts = self._ai_prompt_widget.get_text_prompt().split(",")
        boxes, scores, labels = ai.get_rectangles_from_texts(
            model="yoloworld",
            image=utils.img_qt_to_arr(self.image)[:, :, :3],
            texts=texts,
        )

        for shape in self.canvas.shapes:
            if shape.shape_type != "rectangle" or shape.label not in texts:
                continue
            box = np.array(
                [
                    shape.points[0].x(),
                    shape.points[0].y(),
                    shape.points[1].x(),
                    shape.points[1].y(),
                ],
                dtype=np.float32,
            )
            boxes = np.r_[boxes, [box]]
            scores = np.r_[scores, [1.01]]
            labels = np.r_[labels, [texts.index(shape.label)]]

        boxes, scores, labels = ai.non_maximum_suppression(
            boxes=boxes,
            scores=scores,
            labels=labels,
            iou_threshold=self._ai_prompt_widget.get_iou_threshold(),
            score_threshold=self._ai_prompt_widget.get_score_threshold(),
            max_num_detections=100,
        )

        keep = scores != 1.01
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        shape_dicts: list[dict] = ai.get_shapes_from_annotations(
            boxes=boxes,
            scores=scores,
            labels=labels,
            texts=texts,
        )

        shapes: list[Shape] = []
        for shape_dict in shape_dicts:
            shape = Shape(
                label=shape_dict["label"],
                shape_type=shape_dict["shape_type"],
                description=shape_dict["description"],
            )
            for point in shape_dict["points"]:
                shape.addPoint(QtCore.QPointF(*point))
            shapes.append(shape)

        self.canvas.storeShapes()
        self.loadShapes(shapes, replace=False)
        self.setDirty()

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self.otherData = None
        self.filePath = None
        self.canvas.resetState()
        
    def find_files(self, directory, extensions):
        # 获取目录中的所有文件和子目录
        files = os.listdir(directory)
        
        # 筛选符合后缀的文件
        image_files = [f for f in files if os.path.splitext(f)[1][1:].lower() in extensions]
        
        return image_files

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def tutorial(self):
        url = "https://github.com/labelmeai/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        draw_actions = {
            "polygon": self.actions.createMode,
            "rectangle": self.actions.createRectangleMode,
            "circle": self.actions.createCircleMode,
            "point": self.actions.createPointMode,
            "line": self.actions.createLineMode,
            "linestrip": self.actions.createLineStripMode,
            "ai_polygon": self.actions.createAiPolygonMode,
            "ai_mask": self.actions.createAiMaskMode,
        }

        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            for draw_action in draw_actions.values():
                draw_action.setEnabled(True)
        else:
            for draw_mode, draw_action in draw_actions.items():
                draw_action.setEnabled(createMode != draw_mode)
        self.actions.editMode.setEnabled(not edit)

    def setEditMode(self):
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, "&%d %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def _edit_label(self, value=None):
        if not self.canvas.editing():
            return

        items = self.labelList.selectedItems()
        if not items:
            logger.warning("No label is selected, so cannot edit label.")
            return

        shape = items[0].shape()

        if len(items) == 1:
            edit_text = True
            edit_flags = True
            edit_group_id = True
            edit_description = True
        else:
            edit_text = all(item.shape().label == shape.label for item in items[1:])
            edit_flags = all(item.shape().flags == shape.flags for item in items[1:])
            edit_group_id = all(
                item.shape().group_id == shape.group_id for item in items[1:]
            )
            edit_description = all(
                item.shape().description == shape.description for item in items[1:]
            )

        if not edit_text:
            self.labelDialog.edit.setDisabled(True)
            self.labelDialog.labelList.setDisabled(True)
        if not edit_flags:
            for i in range(self.labelDialog.flagsLayout.count()):
                self.labelDialog.flagsLayout.itemAt(i).setDisabled(True)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(True)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(True)

        text, flags, group_id, description = self.labelDialog.popUp(
            text=shape.label if edit_text else "",
            flags=shape.flags if edit_flags else None,
            group_id=shape.group_id if edit_group_id else None,
            description=shape.description if edit_description else None,
        )

        if not edit_text:
            self.labelDialog.edit.setDisabled(False)
            self.labelDialog.labelList.setDisabled(False)
        if not edit_flags:
            for i in range(self.labelDialog.flagsLayout.count()):
                self.labelDialog.flagsLayout.itemAt(i).setDisabled(False)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(False)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(False)

        if text is None:
            assert flags is None
            assert group_id is None
            assert description is None
            return

        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        self.canvas.storeShapes()
        for item in items:
            shape: Shape = item.shape()

            if edit_text:
                shape.label = text
            if edit_flags:
                shape.flags = flags
            if edit_group_id:
                shape.group_id = group_id
            if edit_description:
                shape.description = description

            self._update_shape_color(shape)
            if shape.group_id is None:
                item.setText(
                    '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                        html.escape(shape.label), *shape.fill_color.getRgb()[:3]
                    )
                )
            else:
                item.setText("{} ({})".format(shape.label, shape.group_id))
            self.setDirty()
            if self.uniqLabelList.findItemByLabel(shape.label) is None:
                item = self.uniqLabelList.createItemFromLabel(shape.label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(shape.label)
                self.uniqLabelList.setItemLabel(item, shape.label, rgb)

    def fileSearchChanged(self):
        self.importDirImages(
            self.lastOpenDir,
            pattern=self.fileSearch.text(),
            load=False,
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.mayContinue():
            return

        currIndex = self.imageList.index(str(item.text()))
        if currIndex < len(self.imageList):
            filename = self.imageList[currIndex]
            if filename:
                self.loadFile(filename)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected)

    def addLabel(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), *shape.fill_color.getRgb()[:3]
            )
        )

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label):
        if self._config["shape_color"] == "auto":
            item = self.uniqLabelList.findItemByLabel(label)
            if item is None:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
            label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            label_id += self._config["shift_auto_shape_color"]
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        elif (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            return self._config["label_colors"][label]
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            description = shape.get("description", "")
            group_id = shape["group_id"]
            other_data = shape["other_data"]

            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
                mask=shape["mask"],
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)
            shape.other_data = other_data

            s.append(shape)
        self.loadShapes(s)

    def loadFlags(self, flags):
        self.flag_widget.clear()
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)

    def saveLabels(self, filename):
        lf = LabelFile()
        verified_temp = self.canvas.verified

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                verified=verified_temp,
                otherData=self.otherData,
                flags=flags,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(self.imagePath, Qt.MatchExactly)
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def duplicateSelectedShape(self):
        self.copySelectedShape()
        self.pasteSelectedShape()

    def pasteSelectedShape(self):
        self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        description = ""
        if self._config["display_label_popup"] or not text:
            previous_text = self.labelDialog.edit.text()
            text, flags, group_id, description = self.labelDialog.popUp(text)
            if not text:
                self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        if text:
            self.labelList.clearSelection()
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            shape.description = description
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightnessContrast(self, value):
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        dialog.exec_()

        brightness = dialog.slider_brightness.value()
        contrast = dialog.slider_contrast.value()
        self.brightnessContrast_values[self.filename] = (brightness, contrast)

    def togglePolygons(self, value):
        flag = value
        for item in self.labelList:
            if value is None:
                flag = item.checkState() == Qt.Unchecked
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)

    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if filename in self.imageList and (
            self.fileListWidget.currentRow() != self.imageList.index(filename)
        ):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        # assumes same name, but json extension
        self.status(str(self.tr("Loading %s...")) % osp.basename(str(filename)))
        label_file = osp.splitext(filename)[0] + ".json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            self.imageData = self.labelFile.imageData
            self.imagePath = osp.join(
                osp.dirname(label_file),
                self.labelFile.imagePath,
            )
            self.otherData = self.labelFile.otherData
            self.canvas.verified = self.labelFile.verified
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
            self.canvas.verified = False
        image = QtGui.QImage.fromData(self.imageData)

        if image.isNull():
            formats = [
                "*.{}".format(fmt.data().decode())
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        flags = {k: False for k in self._config["flags"] or []}
        if self.labelFile:
            self.loadLabels(self.labelFile.shapes)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
        self.loadFlags(flags)
        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        # set brightness contrast values
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if self._config["keep_prev_brightness"] and self.recentFiles:
            brightness, _ = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if self._config["keep_prev_contrast"] and self.recentFiles:
            _, contrast = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        self.brightnessContrast_values[self.filename] = (brightness, contrast)
        if brightness is not None or contrast is not None:
            dialog.onNewValue(None)
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()
        self.status(str(self.tr("Loaded %s")) % osp.basename(str(filename)))
        self.filePath = filename
        self.defaultSaveDir = osp.dirname(filename)
        return True

    def resizeEvent(self, event):
        if (
            self.canvas
            and not self.image.isNull()
            and self.zoomMode != self.MANUAL_ZOOM
        ):
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def dragEnterEvent(self, event):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.mayContinue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def openPrevImg(self, _value=False):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        if self.filename is None:
            return

        currIndex = self.imageList.index(self.filename)
        if currIndex - 1 >= 0:
            filename = self.imageList[currIndex - 1]
            if filename:
                self.loadFile(filename)

        self._config["keep_prev"] = keep_prev
        
    def verifyImg(self, _value=False):
              
        # Proceding next image without dialog if having any label
        if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.saveFile()
                if self.labelFile != None:
                    self.labelFile.toggleVerify()
                else:
                    return

            self.canvas.verified = self.labelFile.verified
            self.paintCanvas()
            self.actions.save.setEnabled(True)
            self.setDirty()
            # self.saveFile()

    def openNextImg(self, _value=False, load=True):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.imageList[0]
        else:
            currIndex = self.imageList.index(self.filename)
            if currIndex + 1 < len(self.imageList):
                filename = self.imageList[currIndex + 1]
            else:
                filename = self.imageList[-1]
        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

        self._config["keep_prev"] = keep_prev

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + ["*%s" % LabelFile.suffix]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self.loadFile(fileName)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(self.imageList.index(current_filename))
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.labelFile:
            # DL20180323 - overwrite when in directory
            self._saveFile(self.labelFile.filename)
        elif self.output_file:
            self._saveFile(self.output_file)
            self.close()
        else:
            self._saveFile(self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def getLabelFile(self):
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"

        return label_file

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, " "proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info("Label file is removed: {}".format(label_file))

            item = self.fileListWidget.currentItem()
            item.setCheckState(Qt.Unchecked)

            self.resetState()

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(self.filename)
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if not self.canvas.hShape.points:
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)
        self.setDirty()

    def deleteSelectedShape(self):
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            "You are about to permanently delete {} polygons, " "proceed anyway?"
        ).format(len(self.canvas.selectedShapes))
        if yes == QtWidgets.QMessageBox.warning(
            self, self.tr("Attention"), msg, yes | no, yes
        ):
            self.remLabels(self.canvas.deleteSelected())
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)

    def copyShape(self):
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else "."
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = osp.dirname(self.filename) if self.filename else "."

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )
        self.importDirImages(targetDirPath)

    @property
    def imageList(self):
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            lst.append(item.text())
        return lst

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)

        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)

        self.openNextImg()

    def importDirImages(self, dirpath, pattern=None, load=True):
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)

        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.filename = None
        self.fileListWidget.clear()

        filenames = self.scanAllImages(dirpath)
        self.mImgList = filenames
        if pattern:
            try:
                filenames = [f for f in filenames if re.search(pattern, f)]
            except re.error:
                pass
        for filename in filenames:
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)
        self.openNextImg(load=load)

    def scanAllImages(self, folderPath):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.normpath(osp.join(root, file))
                    images.append(relativePath)
        images = natsort.os_sorted(images)
        return images
    
    def train_with_labels(self):
        """train with labels, you can choose model and dataset, then start training.
        """
        if self.training_window is None:
            if self.filePath == None or self.defaultSaveDir == None:
                QMessageBox.information(self, u'Wrong!', u'have no loaded folder yet, please check again.')
                self.training_window = TrainingInterface("", "", "")
            else:
                self.training_window = TrainingInterface(img_path=os.path.dirname(self.filePath), annotation_path=self.defaultSaveDir)
            # set_dark_theme(self.training_window)
            # apply_stylesheet(self.training_window)
            font = QFont("微软雅黑", 10)  # 字体名称为微软雅黑，大小为 16
            self.training_window.setFont(font)
            self.training_window.setWindowModality(Qt.ApplicationModal) # 模态窗口（阻塞主窗口）
            self.training_window.show()
            # 连接子窗口关闭信号到回调函数
            self.training_window.destroyed.connect(self.cleanup_training_window)
         

    def cleanup_training_window(self):
        weight_path = self.training_window.weights_path
        cfg_path = self.training_window.cfg_path
        """子窗口销毁时清理引用"""
        print("清理子窗口引用...")
        self.training_window = None  # 删除对 TrainingInterface 的引用
        print("子窗口引用已清理")
        self.yolo_auto_labeling(
            weight_path=weight_path,
            # cfg_path=cfg_path
            )
        
    
    def analyse_result(self):
        pass

    def delete_unchecked(self):
        # 提醒是否真的要删除没有verified的json文件
        reply = QMessageBox.question(self, 'Message', 'Are you sure to delete all unchecked json files?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            json_root = self.defaultSaveDir
            removed_json = []
            if json_root == None:
                QMessageBox.information(self, u'Wrong!', u'have no loaded folder yet, please check again.')
                return
            for path in natsort.natsorted(os.listdir(json_root)):
                if path.split('.')[-1] == 'json':
                    json_path = osp.join(json_root, path)
                    with open(json_path, "r") as f:
                        data = json.load(f)
                        verified = data.get("verified")
                        if not verified:
                            removed_json.append(json_path)
            for json_path in removed_json:
                os.remove(json_path)
            QMessageBox.information(self, 'Message', 'All unchecked json files have been deleted.')
        else:
            pass


    def search_actions_info(self):
        """this is a action information search system.
        input actions name, ti will tell you what it is and how to use it.
        for example, input 'rename_img_xml' or 'ca1' can read actions <rename_img_xml>'s instruction.
        key_word must be actions in menu bar or its shorcut key for now, more intelligent system will update lately. 
        """
        def find_max_similarity(test_str,str_list,threshold=0.7):
            jarowinkler = JaroWinkler()
            max_simi=threshold
            max_str=None
            for string in str_list:
                simi=jarowinkler.similarity(test_str, string)
                if simi > max_simi:
                    max_str=string
                    max_simi = simi
                    
            return max_str
            
        search_key, ok=QInputDialog.getText(self, 'Text Input Dialog', 
                    "Input your search key word：\n(input nothing for search-system's own instruction)")
        if (not ok):
            return
        action_dict={'batch_rename_img':self.batch_rename_img,  'c1':self.batch_rename_img,
                     'rename_img_xml':self.rename_img_json,     'ca1':self.rename_img_json,
                     'duplicate_xml':self.make_duplicate_xml,   'c2':self.make_duplicate_xml,
                     'batch_duplicate':self.batch_duplicate_xml,'ca2':self.batch_duplicate_xml,
                     'label_pruning':self.prune_useless_label,  'c3':self.prune_useless_label,
                     'file_pruning':self.remove_extra_img_xml,  'ca3':self.remove_extra_img_xml,
                     'change_label':self.change_label_name,     'c4':self.change_label_name,
                     'fix_property':self.fix_xml_property,      'c5':self.fix_xml_property,
                     'auto_labeling':self.auto_labeling,        'c6':self.auto_labeling,
                     'data_augment':self.data_auto_augment,     'c7':self.data_auto_augment,
                     'sam_optim':self.sam_optim,                'c8':self.sam_optim,
                     'folder_info':self.show_folder_infor,      'a1':self.show_folder_infor,
                     'label_info':self.show_label_info,         'a2':self.show_label_info,
                     'extract_video':self.extract_video,        's1':self.extract_video,
                     'extract_stream':self.extract_stream,      's2':self.extract_stream,
                     'batch_resize_img':self.batch_resize_img,  's3':self.batch_resize_img,
                     'merge_video':self.merge_video,            's4':self.merge_video,
                     'annotation_video':self.annotation_video,  's5':self.annotation_video,
                     'Search_System':self.search_actions_info
                     }
        search_key='Search_System' if search_key=='' else search_key
        if search_key in action_dict.keys():
            search_info=action_dict[search_key].__doc__
            search_info=search_info.replace('  ','')
            QMessageBox.information(self,u'Info!',search_info)
        else:
            vague_key=find_max_similarity(search_key,action_dict.keys())
            if vague_key in action_dict.keys():
                search_info=action_dict[vague_key].__doc__
                search_info=search_info.replace('  ','')
                search_info="here is info about '{}' based on your input: '{}'\n\n".format(vague_key,search_key)+search_info
                QMessageBox.information(self,u'Info!',search_info)
            else:
                QMessageBox.information(self,u'Sorry!',
                u'unkown key word, key word must in(or similar to) actions in menu bar or its shotcut key, please try again.')
        
    def batch_rename_img(self):
        """batch rename img name. 
        new name constructed by key_word, index, if_fill three prospoty,for example,'car_1.jpg'(or 'car_001.jpg' if fill to 3 digit). 
        additionally, '_' will not appear when key_word is empty. after this actions, you may need reopen img folder.
        !!makesure your new name not conflict with your old name!!
        """
        if self.filePath == None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return
        try:
            path=os.path.dirname(self.filePath)
            filelist = natsort.natsorted(os.listdir(path))
            key_word,ok=QInputDialog.getText(self, 'Text Input Dialog',"Input key word：")
            if not ok:
                return
            index,ok=QInputDialog.getInt(self, 'Text Input Dialog',"Input index：",value=1)
            if not ok:
                return
            Fill,ok=QInputDialog.getInt(self, 'Text Input Dialog',
                    'Digit Fill\n(fill means 1->001, 0 means no fill)',value=0)
            if not ok:
                return
            if 1 < Fill < len(str(index+len(filelist))):
                QMessageBox.information(self,u'Waring!',
                u"your Fill is smaller than largest index's digit, try larger Fill or input 0 to use no fill")
                return
            key_word = '' if key_word == '' else key_word+'_'
            for item in filelist:
                if item.endswith('.jpg') or item.endswith('.jpeg') or item.endswith('.png'):
                    filepath=os.path.join(os.path.abspath(path), item)
                    if Fill > 1:
                        new_item='{}{}.jpg'.format(key_word,str(index).zfill(Fill))
                    else:
                        new_item='{}{}.jpg'.format(key_word,str(index))
                    new_filepath=os.path.join(os.path.abspath(path), new_item)
                    os.rename(filepath,new_filepath)
                    index+=1
            QMessageBox.information(self,u'Done!',u'Batch rename done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
        
    def rename_img_json(self):
        """batch rename img's and its corresponding json's name. 
        new name constructed by key_word, index, if_fill three prospoty,for example,'car_1.jpg'(or 'car_001.jpg' if fill to 3 digit). 
        additionally, '_' will not appear when key_word is empty.after this actions, you may need reopen img folder.
        !!makesure your new name not conflict with your old name!!
        """
        if self.filePath == None:
            QMessageBox.information(self,u'Wrong!', u'have no loaded folder yet, please check again.')
            return
        try:
            img_folder_path = os.path.dirname(self.filePath)
            xml_folder_path = os.path.dirname(self.filePath)
            imglist = natsort.natsorted(os.listdir(img_folder_path))
            xmllist = natsort.natsorted(os.listdir(xml_folder_path))
            key_word,ok = QInputDialog.getText(self, 'Text Input Dialog', "Input key word：")
            if not ok:
                return
            index, ok = QInputDialog.getInt(self, 'Int Input Dialog', "Input index：", value=1)
            if not ok:
                return
            Fill, ok = QInputDialog.getInt(self, 'Int Input Dialog',
                    'Digit Fill\n(fill means 1->001, 0 means no fill)',value=0)
            if not ok:
                return
            if 1 < Fill < len(str(index+len(imglist))):
                QMessageBox.information(self, u'Waring!',
                u"your Fill is smaller than largest index's digit, try larger Fill or input 0 to use no fill")
                return
            key_word = '' if key_word == '' else key_word + '_'
            for item in xmllist:
                if item.endswith('.json') and (item[0:-4]+'.jpg' in imglist or item[0:-4]+'.JPG' in imglist):
                    xmlPath = os.path.join(os.path.abspath(xml_folder_path), item)
                    imgPath = os.path.join(os.path.abspath(img_folder_path), item[0:-4])+'.jpg'
                    if Fill > 1:
                        new_item = '{}{}'.format(key_word, str(index).zfill(Fill))
                    else:
                        new_item = '{}{}'.format(key_word, str(index))
                    new_xmlPath = os.path.join(os.path.abspath(xml_folder_path), new_item+'.json')
                    new_imgPath  =os.path.join(os.path.abspath(img_folder_path), new_item+'.jpg')
                    os.rename(xmlPath, new_xmlPath)
                    os.rename(imgPath, new_imgPath)
                    index+=1
                else:
                    pass
            QMessageBox.information(self,u'Done!',u'Batch rename done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
            
    def make_duplicate_xml(self):
        """copy last xml file to local img, make sure last xml exist. 
        if local xml exist, you need confirm to overwrite it.
        """
        try:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex - 1 >= 0:
                last_filename = self.mImgList[currIndex - 1]
                imgFileName = os.path.basename(last_filename)
                last_xml = os.path.splitext(imgFileName)[0]
                last_path = os.path.join(os.path.dirname(last_filename), last_xml + '.json')
                
                currfilename = self.mImgList[currIndex]
                imgFileName = os.path.basename(currfilename)
                img = cv2.imread(currfilename)
                height, width, _ = img.shape
                curr_xml = os.path.splitext(imgFileName)[0]
                save_path = os.path.join(os.path.dirname(last_filename), curr_xml + '.json')

                xml_info={'filename':'none', 'path':'none'}
                xml_info['filename'] = curr_xml + '.jpg'
                xml_info['path'] = str(self.filePath)
                if os.path.exists(save_path):
                    if self.question_1():
                        print('over write!')
                        os.remove(save_path)
                        pass
                    else:
                        print('cancled!')
                        return
                
                # 将 JSON 字符串加载为 Python 字典
                with open(last_path, 'r') as f:  
                    data = json.load(f) 
                    # 修改 imagePath 字段的值
                    data['imagePath'] = imgFileName
                    data['imageHeight'] = height
                    data['imageWidth'] = width
                    # 指定输出文件的路径
                    output_file_path = save_path
                    # 将修改后的字典转换回JSON字符串，并写入文件
                    with open(output_file_path, 'w', encoding='utf-8') as output_file:
                        json.dump(data, output_file, ensure_ascii=False, indent=4)
                
                print(f'修改后的JSON数据已保存到 {output_file_path}')
                
            else:
                QMessageBox.information(self, u'Sorry!', u'please ensure the first json file exists.')
                return
        except Exception as e:
            QMessageBox.information(self, u'Sorry!', u'something is wrong. ({})'.format(e))
            
    def question_1(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'current xml exists,procesing anyway?'
        return yes == QMessageBox.warning(self, u'Attention:', msg, yes | no)
        
    def batch_duplicate_xml(self):
        """batch copy xml file, make sure at least the first xml exist.
        this action will not overwrite xml file, if local xml exist, it will jump to next and copy local xml to next one.
        """
        if len(self.mImgList) <= 0:
            QMessageBox.information(self, u'Sorry!', u'something is wrong, try load img/xml path again.')
        else:
            for i in range(len(self.mImgList)):
                currfilename = self.mImgList[i]
                imgFileName = os.path.basename(currfilename)
                curr_xml = os.path.splitext(imgFileName)[0]
                save_path = os.path.join(ustr(self.defaultSaveDir),curr_xml+'.xml')
                if i == 0:
                    if os.path.exists(save_path):
                        pass
                    else:
                        QMessageBox.information(self, u'Sorry!', u'please ensure the first xml file exists.')
                        return
                else:
                    last_filename = self.mImgList[i - 1]
                    imgFileName = os.path.basename(last_filename)
                    last_xml = os.path.splitext(imgFileName)[0]
                    last_path = os.path.join(ustr(self.defaultSaveDir),last_xml+'.xml')
                    if os.path.exists(save_path):
                        pass
                    else:
                        xml_info={'filename':'none','path':'none'}
                        xml_info['filename'] = curr_xml+'.jpg'
                        xml_info['path'] = str(self.filePath)
                        tree = ET.ElementTree(file=last_path)
                        root=tree.getroot()
                        for key in xml_info.keys():
                            root.find(key).text=xml_info[key]
                        tree.write(save_path)          
            QMessageBox.information(self, u'Done!', u'batch duplicate xml file succeed, you can procesing other job now.')
        
    def prune_useless_label(self):
        """delete useless label.
        input label name you want to keep, others will be deleted.
        a img whose xml's object(label) deleted completly, this img and its xml will be deleted.
        after this actions, you may need reopen img folder.
        """
        if self.filePath == None:
            QMessageBox.information(self, u'Wrong!', u'have no loaded folder yet, please check again.')
            return
        prune_list=[]
        text, ok=QInputDialog.getText(self, 'Text Input Dialog', "Input labels witch you want keep(split by ',')：")
        text=text.replace(" ", "")
        if ok and text:
            for item in text.split(','):
                prune_list.append(item)
            print(prune_list)
        else:
            QMessageBox.information(self, u'Wrong!', u'get empty list, please try again.')
            return
        if self.question_2(prune_list):
            try:
                print(self.defaultSaveDir, os.path.dirname(self.filePath))  #当前实时的img路径和xml路径
                imglist = os.listdir(os.path.dirname(self.filePath))
                xmllist = os.listdir(self.defaultSaveDir)
                if len(imglist)!=len(xmllist):
                    QMessageBox.information(self, u'Wrong!', u'file list length are different({0}/{1}), please check.'.format(len(imglist), len(xmllist)))
                else:
                    for item in xmllist:
                        xmlPath=os.path.join(os.path.abspath(self.defaultSaveDir), item)
                        tree = ET.ElementTree(file=xmlPath)
                        root=tree.getroot()
                        keep=False
                        for obj in root.findall('object'):
                            if obj.find('name').text in prune_list:
                                keep=True
                                pass
                            else:
                                root.remove(obj)
                        tree.write(xmlPath)  
                        if not keep:
                            os.remove(os.path.join(os.path.abspath(self.defaultSaveDir), item))
                            os.remove(os.path.join(os.path.abspath(os.path.dirname(self.filePath)), item[0:-4])+'.jpg')
                    QMessageBox.information(self,u'Done!',u'label pruning done.')
            except:
                return

    def question_2(self,ls):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'these {0} labels will remain ['.format(len(ls))
        for i in range(len(ls)):
            msg=msg+str(ls[i])+'  '
        msg=msg[0:-2]+'], others will be deleted, sure to continue?(personly advise you back up xml files)'
        return yes == QMessageBox.warning(self, u'Attention:', msg, yes | no)
    
    def remove_extra_img_xml(self):
        """remove img who has no corresponding xml or xml who has no corresponding img.
        after this actions, you may need reopen img folder.
        """
        if self.filePath == None:
            QMessageBox.information(self,u'Wrong!', u'have no loaded folder yet, please check again.')
            return
        try:
            img_folder_path = os.path.dirname(self.filePath)
            xml_folder_path = self.defaultSaveDir
            imglist = sorted(os.listdir(img_folder_path))
            xmllist = sorted(os.listdir(xml_folder_path))
            for item in imglist:
                if item.endswith('.jpg') and (item[0:-4]+'.xml') in xmllist:
                    pass
                else:
                    os.remove(os.path.join(os.path.abspath(img_folder_path), item))

            imglist = os.listdir(img_folder_path)
            xmllist = os.listdir(xml_folder_path)
            for item in xmllist:
                if item.endswith('.xml') and (item[0:-4]+'.jpg') in imglist:
                    pass
                else:
                    os.remove(os.path.join(os.path.abspath(xml_folder_path), item))
            imglist = os.listdir(img_folder_path)
            xmllist = os.listdir(xml_folder_path)
            QMessageBox.information(self,u'Info!',u'done, now have {} imgs, and {} xmls'.format(len(imglist),len(xmllist)))
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))

    def show_folder_infor(self):
        """show current img folder path and xml folder path and img's number and xml's number.
        usually img's amount should equal to xml's amount.
        """
        try:
            imglist = self.find_files(os.path.dirname(self.filePath), ['jpg', 'png', 'bmp', 'jpeg'])
            xmllist = self.find_files(self.defaultSaveDir, 'xml')
            QMessageBox.information(self, u'Folder Info', 
                                    u'img path: \n{0}\nimg nums: {1} imgs\n\nxml path: \n{2}\nxml nums: {3} xmls'.format(
                                        os.path.dirname(self.filePath),
                                        len(imglist),
                                        self.defaultSaveDir,
                                        len(xmllist))
                                    )
           
        except:
            QMessageBox.information(self, u'Wrong!', u'have no loaded folder yet, please check again.')
    
    def plot_histogram_vertical(self, labels, counts):
        """
        绘制从上到下排列的直方图
        :param file_counts: 文件夹名称和文件数量的字典
        """
    
        # 设置y轴的位置
        y_pos = range(len(counts))
    
        # 绘制直方图
        plt.barh(y_pos, counts, align='center', alpha=0.7)
        plt.yticks(y_pos, labels)
        # plt.ylabel('Class')
        # plt.xlabel('Count')
        plt.title('Labels Distribution per Class')
        plt.legend()
        
        # 反转y轴，使直方图从上到下排列
        plt.gca().invert_yaxis()
    
        plt.show()
            
    def show_label_info(self):
        """show all label's name and it's box amount and img amount. 
        """
        
        def file_name(file_dir):
            L = []
            for root, dirs, files in os.walk(file_dir):
                for file in files:
                    if os.path.splitext(file)[1] == '.xml':
                        L.append(os.path.join(root, file))
            return L
        
        try:
            if self.filePath == None:
                QMessageBox.information(self, u'Wrong!', u'have no loaded folder yet, please check again.')
                return
            xml_dirs = self.find_files(self.defaultSaveDir, 'xml')
            
            total_Box = 0
            total_Pic = 0
            Class = []; box_num=[]; pic_num=[]; flag=[]

            for i in range(0, len(xml_dirs)):
                total_Pic += 1
                annotation_file = open(os.path.join(self.defaultSaveDir, xml_dirs[i])).read()
                root = ET.fromstring(annotation_file)
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    if label not in Class:
                        Class.append(label)
                        box_num.append(0)
                        pic_num.append(0)
                        flag.append(0)
                    for i in range(len(Class)):
                        if label == Class[i]:
                            box_num[i] += 1
                            flag[i] = 1
                            total_Box += 1
                for i in range(len(Class)):
                    if flag[i] == 1:
                        pic_num[i] += 1
                        flag[i] = 0
            result = {}
            for i in range(len(Class)):
                result[Class[i]] = (pic_num[i], box_num[i])
            result['total'] = (total_Pic, total_Box)
            info='label | pic_num | box_num \n'
            info += '----------------------------\n'
            for key in result.keys():
                info += '{}:  {}\n'.format(key, result[key])
            QMessageBox.information(self, u'Info', info)

            # DataViewer(result, self).show()
            # 创建直方图
            classes, pic_nums, box_nums = zip(*[(key, val[0], val[1]) for key, val in result.items() if key != 'total'])
            self.plot_histogram_vertical(classes, box_nums)
            
        except Exception as e:
            QMessageBox.information(self, u'Sorry!', u'something is wrong. ({})'.format(e))
    
    def extract_video(self):
        """extract imgs from video.
        'frame gap' means save img by this frequency(not save every img in video if frame_gap larger than 1).
        img will saved in the same path with video.
        this action may take some time, please don't click mouse too frequently.
        """
        try:
            video_path,_ = QFileDialog.getOpenFileName(self,'choose video file:')
            if not video_path:
                return
            save_path = os.path.join(os.path.dirname(os.path.abspath(video_path)), os.path.realpath(video_path).split('.')[0])
            os.makedirs(save_path, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            frame_gap, ok = QInputDialog.getInt(self, 'Int Input Dialog',
                "Input frame gap, img will extract by this frequency",value=1)
            if not ok:
                return
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    index = int(cap.get(1))
                    if index%frame_gap != 0:
                        continue
                    cv2.imwrite(save_path + '/' + str(int(cap.get(1))) + '.jpg', frame)
                else:
                    break
            cap.release()
            QMessageBox.information(self, u'Done!', u'video extract done.')
        except Exception as e:
            QMessageBox.information(self, u'Sorry!', u'something is wrong. ({})'.format(e))

    def extract_stream(self):
        """extract imgs from stream, 'stream_path' usually start with rtsp or rtmp.
        'frame gap' means save img by this frequency(not save every img in video if frame_gap larger than 1).
        'max save number' means actions will stop after save this amount imgs.
        this action will stop after read stream path failed 3 times.
        this action may take some time, please don't click mouse too frequently.
        """
        try:
            stream_path,ok=QInputDialog.getText(self, 'Text Input Dialog', 
                        "Input steam path(start with rtmp、rtsp...):")
            if not(stream_path and ok):
                return
            save_path = QFileDialog.getExistingDirectory()
            if not save_path:
                return
            frame_gap,ok=QInputDialog.getInt(self, 'Int Input Dialog',
                "Input frame gap, img will extract by this frequency",value=1)
            if not ok:
                return
            max_frame,ok=QInputDialog.getInt(self, 'Int Input Dialog',
                "Input max save number, process will end after save cetain number imgs",value=10)
            if not ok:
                return
            cap = cv2.VideoCapture(stream_path)
            drop_times=0
            while True:
                ret, frame = cap.read()
                if ret:
                    index=int(cap.get(1))
                    if index%frame_gap!=0:
                        continue
                    if index>(max_frame*frame_gap):
                        break
                    cv2.imwrite(save_path+'/'+str(int(cap.get(1)))+'.jpg',frame)
                else:
                    cap.release()
                    drop_times+=1
                    if drop_times>=3:
                        QMessageBox.information(self,u'Wrong!',u'stream path not useable.')
                        break
                    cap = cv2.VideoCapture(stream_path)
            cap.release()
            QMessageBox.information(self,u'Done!',u'stream extract done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
           
    def batch_resize_img(self):
        """input Wdith and Height to resize all img to one shape.
        """
        if self.filePath == None:
            QMessageBox.information(self, u'Wrong!', u'have no loaded folder yet, please check again.')
            return    
        try:
            img_path = os.path.dirname(self.filePath)
            filelist = natsort.natsorted(os.listdir(img_path))
            new_W, ok  =QInputDialog.getInt(self,'Integer input dialog', 'input img wdith :', value=1920)
            if not ok:
                return
            new_H, ok = QInputDialog.getInt(self,'Integer input dialog', 'input img height :', value=1080)
            if not ok:
                return
            for item in filelist:
                img = cv2.imread(os.path.join(img_path, item))
                img = cv2.resize(img, (new_W, new_H))
                cv2.imwrite(os.path.join(img_path, item), img)
                
            QMessageBox.information(self, u'Done!', u'batch resize done.')
        except Exception as e:
            QMessageBox.information(self, u'Sorry!', u'something is wrong. ({})'.format(e))
        
    def merge_video(self):
        """merge all img in one path to one video, video will saved in img's parent path.
        for some restraint, fps must be 25, you can use 'repeat times' to repeat play img if you want slower the video.
        this action may take some time, please don't click mouse too frequently. 
        you can press 'space' if you find bounding box not accurate during auto annotate.
        """
        try:
            img_path = QFileDialog.getExistingDirectory(self,'choose imgs folder:')
            if not img_path:
                return
            filelist = natsort.natsorted(os.listdir(img_path)) #获取该目录下的所有文件名
            img=cv2.imread(img_path+'/'+filelist[0])
            img_size=img.shape
            fps = 25
            repeat_time,ok = QInputDialog.getInt(self, 'Int Input Dialog',
                        "Input each img's repeat times(the bigger, the slower), usually set 1",value=1)
            if not ok:
                return
            file_path = img_path +'_result' + ".avi" #导出路径
            fourcc = cv2.VideoWriter_fourcc('P','I','M','1')
            video = cv2.VideoWriter( file_path, fourcc, fps ,(img_size[1],img_size[0]))
            for item in filelist:
                if item.endswith('.jpg'):   #判断图片后缀是否是.png
                    item = img_path +'/'+item 
                    img = cv2.imread(item)
                    for j in range(repeat_time):
                        video.write(img)        

            video.release()
            QMessageBox.information(self,u'Done!',u'video merge done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))

    def annotation_video(self):
        """ auto annotation video file or local camera.
        select video file, cancle to use local camera.
        img and xml will saved on dir of video path unless you use local camera, and folder will be './' in which case.
        'CSRT' type means more accuracy and low speed(recommend), 'MOSSE' means high speed and low accuracy, 'KCF' is in middle.
        frames are resized for display reason, one better run 'fix_property' after this process.
        press 'space' to re-drawing bounding box during annotation if you find bounding box not accurate.
        """
        try:
            tree = ET.ElementTree(file='./data/origin.xml')
            root=tree.getroot()
            for child in root.findall('object'):
                template_obj=child#保存一个物体的样板
                root.remove(child)
            tree.write('./data/template.xml')
            trackerType_selector={'CSRT':cv2.TrackerCSRT_create,
                                  'BOOSTING':cv2.TrackerBoosting_create,
                                  'MIL':cv2.TrackerMIL_create,
                                  'KCF':cv2.TrackerKCF_create,
                                  'TLD':cv2.TrackerTLD_create,
                                  'MEDIANFLOW':cv2.TrackerMedianFlow_create,
                                  'GOTURN':cv2.TrackerGOTURN_create,
                                  'MOSSE':cv2.TrackerMOSSE_create}
            items=tuple(trackerType_selector)
            trackerType , ok = QInputDialog.getItem(self, "Select",
                "Tracker type, usually 'CSRT' is ok:", items, 0, False)
            if not ok:
                return
            videoPath ,_ = QFileDialog.getOpenFileName(self,"choose video file, cancle to use local camera:")
            if not videoPath:
                videoPath = 0
            save_gap,ok=QInputDialog.getInt(self,'Integer input dialog','input save gap, img will saved by this frenquency :',value=25)
            if not ok:
                return
            img_size,ok=QInputDialog.getInt(self,'Integer input dialog','input img size, img resized ti this shape by height:',value=900)
            if not ok:
                return
            process_shape=(int(1.777*img_size),int(img_size))
            cap = cv2.VideoCapture(videoPath)
            ret, frame = cap.read()
            height_K=frame.shape[0]/img_size
            weight_K=frame.shape[1]/(1.777*img_size)
            if not ret:
                print('Failed to read video')
                sys.exit(1)
            else:
                pass
                frame=cv2.resize(frame,process_shape)
            def init_multiTracker(frame):
                bboxes = []
                colors = []
                labels = []
                while True:
                    # 在对象上绘制边界框selectROI的默认行为是从fromCenter设置为false时从中心开始绘制框，可以从左上角开始绘制框
                    bbox = cv2.selectROI("draw box and press 'SPACE' to affirm, Press 'q' to quit draw box and start tracking-labeling", frame)
                    if min(bbox[2],bbox[3]) >= 10:
                        label_name,ok=QInputDialog.getText(self, 'Text Input Dialog', 
                        "Input label name:")
                        if not(label_name and ok):
                            return
                        labels.append(label_name)
                        bboxes.append(bbox)
                        colors.append((random.randint(30, 240), random.randint(30, 240), random.randint(30, 240)))
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        cv2.rectangle(frame, p1, p2, [10,250,10], 2, 1)
                    else:
                        print("bbox size small than 10, will be abandoned")
                    k=cv2.waitKey(0)
                    print(k)
                    if k==113:
                        break
                print('Selected bounding boxes: {}'.format(bboxes))
                multiTracker = cv2.MultiTracker_create()
                # 初始化多跟踪器
                for bbox in bboxes:
                    tracker=trackerType_selector[trackerType]()
                    multiTracker.add(tracker, frame, bbox)    
                return multiTracker,colors,labels
            multiTracker,colors,labels=init_multiTracker(frame)
            cv2.namedWindow('MultiTracker', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("MultiTracker", process_shape[0], process_shape[1])
            cv2.moveWindow("MultiTracker", 10, 10)
            # 处理视频并跟踪对象
            index=0
            while cap.isOpened():
                ret, origin_frame = cap.read()
                if not ret:
                    break
                frame=cv2.resize(origin_frame,process_shape)
                draw=frame.copy()
                ret, boxes = multiTracker.update(frame)
                # 绘制跟踪的对象
                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(draw, p1, p2, colors[i], 2, 1)
                    info = labels[i]
                    t_size=cv2.getTextSize(info, cv2.FONT_HERSHEY_TRIPLEX, 0.7 , 1)[0]
                    cv2.rectangle(draw, p1, (int(newbox[0]) + t_size[0]+3, int(newbox[1]) + t_size[1]+6), colors[i], -1)
                    cv2.putText(draw, info, (int(newbox[0])+1, int(newbox[1])+t_size[1]+2), cv2.FONT_HERSHEY_TRIPLEX, 0.7, [255,255,255], 1)
                # show frame
                cv2.imshow("MultiTracker, press 'SPACE' to redraw box, press 'q' to quit video labeling", draw)
                # quit on ESC or Q button
                if index%save_gap==0:
                    tree = ET.ElementTree(file='./data/template.xml')
                    root=tree.getroot()
                    for i, newbox in enumerate(boxes):
                        temp_obj=template_obj
                        temp_obj.find('name').text=str(labels[i])
                        temp_obj.find('bndbox').find('xmin').text=str(int(weight_K*newbox[0]))
                        temp_obj.find('bndbox').find('ymin').text=str(int(height_K*newbox[1]))
                        temp_obj.find('bndbox').find('xmax').text=str(int(weight_K*newbox[0]+weight_K*newbox[2]))
                        temp_obj.find('bndbox').find('ymax').text=str(int(height_K*newbox[1]+height_K*newbox[3]))
                        root.append(deepcopy(temp_obj))       #深度复制
                    if videoPath==0:
                        parent_path='./temp'
                    else:
                        parent_path=os.path.dirname(videoPath)
                    os.makedirs(os.path.join(parent_path, 'JPEGImages'), exist_ok=True)
                    os.makedirs(os.path.join(parent_path, 'Annotations'), exist_ok=True)
                    cv2.imwrite(os.path.join(parent_path, 'JPEGImages/','{}.jpg'.format(index)),origin_frame)
                    tree.write(os.path.join(parent_path, 'Annotations/','{}.xml'.format(index)))
                index+=1
                k=cv2.waitKey(1)
                if k==32: #press space to reinit box
                    cv2.destroyAllWindows()
                    multiTracker,colors,labels=init_multiTracker(frame)
                    cv2.namedWindow('MultiTracker', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("MultiTracker", process_shape[0], process_shape[1])
                    cv2.moveWindow("MultiTracker", 10, 10)
                if k== 27 or k == 113: #press q or esc to quit
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            cap.release()
            cv2.destroyAllWindows()
            QMessageBox.information(self,u'Done!',u'video auto annotation done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
    
    def change_label_name(self):
        """change label name. from 'origin' to 'target'
        you can only change one label each time, not support multi-label changing.
        """
        if self.filePath == None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return    
        try:
            label_transform={}
            origin, ok=QInputDialog.getText(self, 'Text Input Dialog', "Input origin label name(only sigle label)：")
            if (not ok) and origin !='':
                return
            origin=origin.replace(" ","")
            target, ok=QInputDialog.getText(self, 'Text Input Dialog', "Input target label name(only sigle label)：")
            if (not ok) and target != '':
                return
            target=target.replace(" ","")
            label_transform[origin]=target
            xml_folder_path=self.defaultSaveDir
            img_folder_path=os.path.dirname(self.filePath)
            imglist = natsort.natsorted(os.listdir(img_folder_path))
            xmllist = natsort.natsorted(os.listdir(xml_folder_path))
            for item in xmllist:
                if item.endswith('.xml'):
                    if (item[0:-4]+'.jpg') in imglist:
                        xmlPath=os.path.join(os.path.abspath(xml_folder_path), item)
                        imgPath=os.path.join(os.path.abspath(img_folder_path), item[0:-4])+'.jpg'
                        tree = ET.ElementTree(file=xmlPath)
                        root=tree.getroot()
                        for obj in root.findall('object'):
                            if obj.find('name').text in label_transform.keys():
                                obj.find('name').text=label_transform[obj.find('name').text]
                        tree.write(xmlPath)
                    else:
                        print(item,'has no corresponding img')
                        os.remove(os.path.join(os.path.abspath(xml_folder_path), item))
                        
            QMessageBox.information(self,u'Done!',u'label name changed!')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
        
    def fix_xml_property(self):
        """fix xml's property such as size, folder, filename, path.
        """
        if self.filePath == None:
            QMessageBox.information(self, u'Wrong!', u'have no loaded folder yet, please check again.')
            return 
        try:
            xml_folder_path = self.defaultSaveDir
            img_folder_path = os.path.dirname(self.filePath)
            xmllist = os.listdir(xml_folder_path)
            folder_info = {'folder':'JPEGImages', 'filename':'none', 'path':'none'}
            for item in xmllist:
                if item.endswith('.xml'):
                    folder_info['filename'] = item[0:-4]+'.jpg'
                    folder_info['path'] = os.path.join(img_folder_path, item[0:-4])+'.jpg'
                    img = cv2.imread(folder_info['path'])
                    size = img.shape
                    xmlPath = os.path.join(os.path.abspath(xml_folder_path), item)
                    tree = ET.ElementTree(file=xmlPath)
                    root = tree.getroot()
                    try:
                        root.find('size').find('width').text=str(size[1])
                        root.find('size').find('height').text=str(size[0])
                        root.find('size').find('depth').text=str(size[2])
                    except:
                        print('xml has no size attribute!')
                    for key in folder_info.keys():
                        try:
                            root.find(key).text = folder_info[key]
                        except:
                            print(item, ': attribute', key, 'not exist!')
                            pass
                    tree.write(xmlPath)
            QMessageBox.information(self, u"Done!", u"fix xml's property done!")
        except Exception as e:
            QMessageBox.information(self, u'Sorry!', u'something is wrong. ({})'.format(e))
     
    def yolo_auto_labeling(self, weight_path=None, cfg_path='cfgs'):
        
        if weight_path == None:
            weight_path = QFileDialog.getExistingDirectory(self, "Choose 'yolo_weights' folder:", 'yolo_weights', QFileDialog.ShowDirsOnly)
            if weight_path == '':
                return
        
        weight_list=[]
        for item in sorted(os.listdir(weight_path)):
            if item.endswith('.h5') or item.endswith('.pt') or item.endswith('.pth'):
                weight_list.append(item)
        if len(weight_list) == 0:
            QMessageBox.information(self, u'Wrong!', u'have no weight file in this folder, please check again.')
            return
        items = tuple(weight_list)
        if len(weight_list) > 0 :
            weights, ok = QInputDialog.getItem(self, "Select",
            f"Model weights file(under {weight_path}):", 
            items, 0, False)
            if not ok:
                return
            else:
                weights = os.path.join(weight_path, weights)
        else:
            weights,_ = QFileDialog.getOpenFileName(self,"'yolo_weights' is empty, choose model weights file:")
            if not (weights.endswith('.pt') or weights.endswith('.pth')):
                QMessageBox.information(self, u'Wrong!', u'weights file must endswith .h5 or .pt or .pth')
                return
        conf_thres = 0.5
        iou_thres = 0.5
        # Initialize
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # half = device.type != 'cpu'  # half precision only supported on CUDA
        half = False  # half precision only supported on CUDA

        # Load model and label name.
        model = YOLO(weights)
        task = model.task
        model.to(device)

        cfg_list = []
        for item in sorted(os.listdir(cfg_path)):
            if item.endswith('.yaml'):
                cfg_list.append(item)
        items = tuple(cfg_list)
        if len(cfg_list) > 0 :
            cfgs, ok = QInputDialog.getItem(self, 
                                            "Select", 
                                            "Configure file(Configure file should under 'cfgs' folder):", 
                                            items, 0, False)
            if not ok:
                return
            else:
                cfgs = os.path.join(cfg_path, cfgs)
        else:
            cfgs,_ = QFileDialog.getOpenFileName(self, "'cfgs' is empty, choose configure file:")
            if not cfgs.endswith('.yaml'):
                QMessageBox.information(self, u'Wrong!', u'configure file must endswith .yaml')
                return

        # 读取 YAML 文件
        with open(cfgs, 'r') as file:
            data = yaml.safe_load(file)
            
        # 提取 names 部分的内容并放入字典中
        names = [value for key, value in data['names'].items()]                     
        if len(names) == 1:
            needed_labels = names
        else:
            msg = "Select labels you want auto-labeling?"
            title = "Select Labels"      
            sorted_names = sorted(names) 
            dialog = MultiChoiceDialog(msg, title, sorted_names)
            if dialog.exec_() == QDialog.Accepted:
                needed_labels = dialog.selected_choices()
                print("Selected labels:", needed_labels)
            else:
                print("Dialog cancelled")
                return
        
        # 询问是否使用Simplify来稀疏多边形
        # 没有取消选项，只有是和否
        # reply = QMessageBox.question(self, 'Message', 'Whether use Simplify to rarefy polygon?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # yes_or_no = True if reply == QMessageBox.Yes else False
        # 如果选择了是，则询问容差值
        items = tuple(["Yes", "No"])
        yes_or_no, ok = QInputDialog.getItem(self, 
                                             "Select",
                                             'Whether use Simplify to rarefy polygon?', 
                                             items, 0, False)
        if not ok:
            return
        else:
            yes_or_no = True if yes_or_no == "Yes" else False
        
        # set imsize
        if yes_or_no:
            tolerance, OK = QInputDialog.getInt(self, 'Simplify Setting', 'tolerance value (default=0.5):', value=5)
            if not OK:
                return
        
        imgsz = 640
        
        # 函数：将图像转换为Base64编码
        def image_to_base64(image_path):
            # 打开图像文件
            with Image.open(image_path) as img:
                # 将图像保存到内存中，以便将其转换为 Base64
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                # 将图像内容转换为Base64编码
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str
        
        # 函数：生成 LabelMe 格式的 JSON
        def generate_labelme_json(image_path, segmentation_masks, cls, names, image_height, image_width, needed_labels):
            labelme_data = {
                "version": "5.5.0",
                "flags": {},
                "shapes": [],
                "imagePath": os.path.basename(image_path),
                # "imageData": image_to_base64(image_path),
                "imageData": None,
                "imageHeight": image_height,
                "imageWidth": image_width,
                "verified": False
            }

            for i, mask in enumerate(segmentation_masks):
                label = names[cls[i]]
                if label not in needed_labels:
                    continue
                
                # 将分割掩码坐标转换为多边形的坐标
                points = np.array(mask).reshape(-1, 2).tolist()
                
                if yes_or_no:
                    # 创建一个Shapely多边形对象
                    polygon = Polygon(points)

                    # 使用simplify方法简化多边形，指定容忍度（误差阈值）
                    simplified_polygon = polygon.simplify(tolerance=tolerance, preserve_topology=True)

                    # 获取简化后的点集
                    points = list(simplified_polygon.exterior.coords)

                # 根据需求判断是矩形还是多边形
                shape_type = "polygon" if task == 'segment' else "rectangle"

                # 添加标注信息
                labelme_data["shapes"].append({
                    "label": label,
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": shape_type,
                    "flags": {},
                    "mask": None  # 如果有需要，也可以通过mask提供图像数据
                })

            return labelme_data
        
        source = os.path.dirname(self.filePath)

        imgsz = imgsz  # check img_size
        if half:
            model.half()  # to FP16
        
        # load img and run inference
        dataset = LoadImages(source, img_size=imgsz)
        progress = QProgressDialog(self)
        progress.setWindowTitle(u"Waiting")
        progress.setLabelText(u"auto-labeling with yolo now, Please wait...")
        progress.setCancelButtonText(u"Cancle it")
        progress.setMinimumDuration(1)
        progress.setWindowModality(Qt.WindowModal)
        progress.setRange(0, 100)
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        index = -1
        need_label_image_num = 0
        success_index = 0
        for path, img, im0s, vid_cap in dataset:
            json_path = path.split('.')[0] + ".json"
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                    verified = data.get("verified")
                    if verified:
                        continue    
            index += 1
            progress.setValue(int(100 * index / len(dataset)))
            if progress.wasCanceled():
                QMessageBox.warning(self, "Attention", "auto-labeling canceled！") 
                return

            # Inference
            result = model(im0s, augment=False)
            cls = result[0].boxes.cls
            cls = cls.cpu().numpy().astype(np.int32).tolist()
            names = result[0].names
                
            if task == "classify":
                pass
            elif task == "detect":
                try:
                    if result[0].boxes is None:
                        continue
                    data = result[0].boxes.xyxy
                except Exception as e:
                    QMessageBox.information(self, u'Sorry!', u'something is wrong. ({})'.format(e))
            elif task == "obb":
                pass
            elif task == "segment":
                try:
                    if result[0].masks is None:
                        continue
                    data = result[0].masks.xy
                except Exception as e:
                    QMessageBox.information(self, u'Sorry!', u'something is wrong. ({})'.format(e))           
            else:
                QMessageBox.information(self, u'Sorry!', f'Unimplemented {task} task.')
                
            height, width, _ = im0s.shape
            # 生成 LabelMe 格式的标注数据
            labelme_json = generate_labelme_json(path, data, cls, names, height, width, needed_labels)

            # 将结果保存为JSON文件
            json_path = path.split('.')[0] + ".json"
            with open(json_path, 'w') as f:
                json.dump(labelme_json, f, indent=4)

            success_index += 1
            print(f"LabelMe格式的标注已生成并保存为 {json_path}")
                
        progress.setValue(100)
        QMessageBox.information(self, u'Done!', f'auto labeling done. {success_index}/{index+1} images have got auto labels. \nplease reload img folder')
            
                    
    def auto_labeling(self):
        """use model to labeling unannotated imgs.
        you should choose model type as well as model weights file and input label name.
        supported model type is 'yolo' or 'Retinanet', more type will updating later.
        'label name' must sorted by class number, if you do not remenber them, just press 'Enter', 
        box will named by its class number and you can change them one by one using action <change_label>
        for a recorde, yolo do not need 'label name', but 'img size' additionally.
        this action may take some time, please don't click mouse too frequently.
        """
        if self.filePath == None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return
        try:
            #=====choose model and input label name=====
            # using yolo autolabeling   
            with torch.no_grad():
                self.yolo_auto_labeling()
            return
        
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
        
    def data_auto_augment(self):
        """data augment, using Affine change, intensity change, contrast change, gama change, Gaussian fillter to augment img data.
        you can select augment multiple(1~4).
        this action may take some time, please don't click mouse too frequently.
        """
        try:
            self.xml_folder_path = self.defaultSaveDir
            self.img_folder_path = os.path.dirname(self.filePath)
            imglist = natsort.natsorted(os.listdir(self.img_folder_path))
            xmllist = natsort.natsorted(os.listdir(self.xml_folder_path))
            print(len(imglist), len(xmllist))
            if len(imglist) != len(xmllist):
                QMessageBox.information(self, u'Wrong!', u'lens of img and xml do not equal.')
                return 
            else:
                magnification, OK = QInputDialog.getInt(self, 'Integer input dialog', 'input augment magnification(1~4):', value=4)
                if OK:
                    if magnification < 1 or magnification > 4:
                        magnification = 4
                    img_Temp = []
                    xml_Temp = []
                    for i in range(magnification):
                        img_Temp.extend(imglist)
                        xml_Temp.extend(xmllist)
                    one_step = int(len(img_Temp)/4)
                    imglist1 = img_Temp[0:one_step]; xmllist1 = xml_Temp[0:one_step]
                    imglist2 = img_Temp[one_step:2*one_step]; xmllist2 = xml_Temp[one_step:2*one_step]
                    imglist3 = img_Temp[2*one_step:3*one_step]; xmllist3 = xml_Temp[2*one_step:3*one_step]
                    imglist4 = img_Temp[3*one_step:]; xmllist4 = xml_Temp[3*one_step:]
                    progress = QProgressDialog(self)
                    progress.setWindowTitle(u"Waiting")  
                    progress.setLabelText(u"Processing now, Please wait...")
                    progress.setCancelButtonText(u"Cancle data augment")
                    progress.setMinimumDuration(1)
                    progress.setWindowModality(Qt.WindowModal)
                    progress.setRange(0,100) 
                    
                    self.augment_A(imglist1, xmllist1, progress)
                    self.augment_B(imglist2, xmllist2, progress)
                    self.augment_C(imglist3, xmllist3, progress)
                    self.augment_D(imglist4, xmllist4, progress)
                    imglist = natsort.natsorted(os.listdir(self.img_folder_path))
                    xmllist = natsort.natsorted(os.listdir(self.xml_folder_path))
                    self.exam_augment(xmllist, progress)
                    
                    progress.setValue(100)
                    QMessageBox.information(self,"Done","data augment scuesseed！")
            
        except Exception as e:
            QMessageBox.information(self, u'Sorry!', u'something is wrong. ({})'.format(e))
    
    def sam_optim(self):
        pass
    
    def augment_A(self, imglist, xmllist, progress):
        print('agmt:', len(imglist))
        shift_info = []
        for i in range(len(imglist)):
            progress.setValue(17*(i/(len(imglist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self, "Attention", "augment failed, please check floder！") 
                break
            item = imglist[i]
            if item.endswith('.jpg'):
                imgPath = os.path.join(os.path.abspath(self.img_folder_path), item)
                img = cv2.imread(imgPath)
                size = img.shape
                new_img1 = cv2.flip(img, 1, dst=None)
                shift_X = np.random.randint(-0.15*size[1], 0.15*size[1])
                shift_Y = np.random.randint(-0.15*size[0], 0.15*size[0])
                shift_info.append([shift_X, shift_Y])
                M = np.float32([[1, 0, shift_X], [0, 1, shift_Y]]) #13
                shifted = cv2.warpAffine(new_img1, M, (new_img1.shape[1], new_img1.shape[0]), borderValue=(99,99,99))
                noise = np.random.randint(-8, 8, size=[size[0],size[1],3])
                new_img = shifted+noise
                save_path = self.img_folder_path + '/agmtA_'+item
                cv2.imwrite(save_path, new_img)

        for i in range(len(xmllist)):
            progress.setValue(17 + 3*(i/(len(xmllist))))
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention", "augment failed, please check floder！") 
                break
            item=xmllist[i]
            if item.endswith('.xml'):
                filePath=os.path.join(os.path.abspath(self.xml_folder_path), item)
                imgPath=os.path.join(os.path.abspath(self.img_folder_path), item[0:-4])+'.jpg'
                img = cv2.imread(imgPath)
                size=img.shape
                tree = ET.ElementTree(file=filePath)
                root=tree.getroot()
                root.find('filename').text='agmtA_' + item[0:-4]+'.jpg'
                root.find('path').text=self.img_folder_path.replace('\\','/') + '/agmtA_' + item[0:-4]+'.jpg'
                for child in root:
                    if child.tag=='object':
                        for gchild in child:
                            if gchild.tag=='bndbox':
                                temp=gchild[0].text
                                gchild[0].text=str(size[1]-int(gchild[2].text)+shift_info[i][0])
                                gchild[1].text=str(int(gchild[1].text)+shift_info[i][1])
                                gchild[2].text=str(size[1]-int(temp)+shift_info[i][0])
                                gchild[3].text=str(int(gchild[3].text)+shift_info[i][1])
                tree.write(self.xml_folder_path+'/agmtA_'+item)
        print('augment_A done!')

    def augment_B(self,imglist,xmllist,progress):
        shift_info=[]
        for i in range(len(imglist)):
            progress.setValue(20 + 17*(i/(len(imglist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self, "Attention", "augment failed, please check floder！") 
                break
            item=imglist[i]
            if item.endswith('.jpg'):
                imgPath=os.path.join(os.path.abspath(self.img_folder_path), item)
                img = cv2.imread(imgPath)
                size=img.shape
                result=99*np.ones(img.shape)
                k=random.uniform(0.5,0.7) #根据实际需求更改范围，小于1为缩小，大于1为放大
                small = cv2.resize(img, (0,0), fx=k, fy=k, interpolation=cv2.INTER_AREA)
                result[0:small.shape[0],0:small.shape[1],:]=small
                shift_X = np.random.randint(-0.1*size[1], 0.3*size[1])
                shift_Y = np.random.randint(-0.1*size[0], 0.3*size[0])
                shift_info.append([k, shift_X, shift_Y])
                M = np.float32([[1, 0, shift_X], [0, 1, shift_Y]]) 
                shifted = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]), borderValue=(99,99,99))
                noise = np.random.randint(-8, 8, size=[size[0],size[1],3])
                new_img = shifted + noise
                save_path = self.img_folder_path + '/agmtB_' + item
                cv2.imwrite(save_path, new_img)

        for i in range(len(xmllist)):
            progress.setValue(37+3*(i/(len(xmllist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","augment failed, please check floder！") 
                break
            item=xmllist[i]
            if item.endswith('.xml'):
                xmlPath=os.path.join(os.path.abspath(self.xml_folder_path), item)
                tree = ET.ElementTree(file=xmlPath)
                root=tree.getroot()
                root.find('filename').text='agmtB_'+item[0:-4]+'.jpg'
                root.find('path').text=self.img_folder_path.replace('\\','/')+'/agmtB_'+item[0:-4]+'.jpg'
                for child in root.findall('object'):
                    ymin=int(child.find('bndbox').find('ymin').text)
                    ymax=int(child.find('bndbox').find('ymax').text)
                    xmin=int(child.find('bndbox').find('xmin').text)
                    xmax=int(child.find('bndbox').find('xmax').text)
                    child.find('bndbox').find('ymin').text=str(int(shift_info[i][0]*ymin+shift_info[i][2]))
                    child.find('bndbox').find('ymax').text=str(int(shift_info[i][0]*ymax+shift_info[i][2]))
                    child.find('bndbox').find('xmin').text=str(int(shift_info[i][0]*xmin+shift_info[i][1]))
                    child.find('bndbox').find('xmax').text=str(int(shift_info[i][0]*xmax+shift_info[i][1]))
                tree.write(self.xml_folder_path+'/agmtB_'+item)
        print('augment_B done!')

    def augment_C(self,imglist,xmllist,progress):
        shift_info=[]
        for i in range(len(imglist)):
            progress.setValue(40+17*(i/(len(imglist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","augment failed, please check floder！") 
                break
            item=imglist[i]
            if item.endswith('.jpg'):
                imgPath=os.path.join(os.path.abspath(self.img_folder_path), item)
                img = cv2.imread(imgPath)
                size=img.shape
                a=int(2*random.randint(1,3)+1)
                b=random.uniform(11,21)
                blur = cv2.GaussianBlur(img,(a,a),b)
                shift_X=np.random.randint(-0.1*size[1], 0.1*size[1])
                shift_Y=np.random.randint(-0.1*size[0], 0.1*size[0])
                shift_info.append([shift_X,shift_Y])
                M = np.float32([[1, 0, shift_X], [0, 1, shift_Y]]) #13
                shifted = cv2.warpAffine(blur, M, (blur.shape[1], blur.shape[0]),borderValue=(99,99,99))
                noise=np.random.randint(-5,5,size=[size[0],size[1],3])
                new_img=shifted+noise
                save_path=self.img_folder_path+'/agmtC_'+item
                cv2.imwrite(save_path,new_img)

        for i in range(len(xmllist)):
            progress.setValue(57+3*(i/(len(xmllist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","augment failed, please check floder！") 
                break
            item=xmllist[i]
            if item.endswith('.xml'):
                filePath=os.path.join(os.path.abspath(self.xml_folder_path), item)
                imgPath=os.path.join(os.path.abspath(self.img_folder_path), item[0:-4])+'.jpg'
                img = cv2.imread(imgPath)
                size=img.shape
                tree = ET.ElementTree(file=filePath)
                root=tree.getroot()
                root.find('filename').text='agmtC_'+item[0:-4]+'.jpg'
                root.find('path').text=self.img_folder_path.replace('\\','/')+'/agmtC_'+item[0:-4]+'.jpg'
                for child in root.findall('object'):
                    ymin=int(child.find('bndbox').find('ymin').text)
                    ymax=int(child.find('bndbox').find('ymax').text)
                    xmin=int(child.find('bndbox').find('xmin').text)
                    xmax=int(child.find('bndbox').find('xmax').text)
                    child.find('bndbox').find('ymin').text=str(int(ymin+shift_info[i][1]))
                    child.find('bndbox').find('ymax').text=str(int(ymax+shift_info[i][1]))
                    child.find('bndbox').find('xmin').text=str(int(xmin+shift_info[i][0]))
                    child.find('bndbox').find('xmax').text=str(int(xmax+shift_info[i][0]))
                tree.write(self.xml_folder_path+'/agmtC_'+item)
        print('augment_C done!')

    def augment_D(self,imglist,xmllist,progress):
        shift_info=[]
        for i in range(len(imglist)):
            progress.setValue(60+27*(i/(len(imglist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","augment failed, please check floder！") 
                break
            item=imglist[i]
            if item.endswith('.jpg'):
                imgPath=os.path.join(os.path.abspath(self.img_folder_path), item)
                img = cv2.imread(imgPath)
                size=img.shape
                k=random.uniform(0.6,1.3)
                gama=exposure.adjust_gamma(img,k)
                shift_X=np.random.randint(-0.1*size[1], 0.1*size[1])
                shift_Y=np.random.randint(-0.1*size[0], 0.1*size[0])
                shift_info.append([shift_X,shift_Y])
                M = np.float32([[1, 0, shift_X], [0, 1, shift_Y]]) #13
                shifted = cv2.warpAffine(gama, M, (gama.shape[1], gama.shape[0]),borderValue=(99,99,99))
                noise=np.random.randint(-5,5,size=[size[0],size[1],3])
                new_img=shifted+noise
                save_path=self.img_folder_path+'/agmtD_'+item
                cv2.imwrite(save_path,new_img)
        for i in range(len(xmllist)):
            progress.setValue(87+3*(i/(len(xmllist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","augment failed, please check floder！") 
                break
            item=xmllist[i]
            if item.endswith('.xml'):
                filePath=os.path.join(os.path.abspath(self.xml_folder_path), item)
                imgPath=os.path.join(os.path.abspath(self.img_folder_path), item[0:-4])+'.jpg'
                img = cv2.imread(imgPath)
                size=img.shape
                tree = ET.ElementTree(file=filePath)
                root=tree.getroot()
                root.find('filename').text='agmtD_'+item[0:-4]+'.jpg'
                root.find('path').text=self.img_folder_path.replace('\\','/')+'/agmtD_'+item[0:-4]+'.jpg'
                for child in root.findall('object'):
                    ymin=int(child.find('bndbox').find('ymin').text)
                    ymax=int(child.find('bndbox').find('ymax').text)
                    xmin=int(child.find('bndbox').find('xmin').text)
                    xmax=int(child.find('bndbox').find('xmax').text)
                    child.find('bndbox').find('ymin').text=str(int(ymin+shift_info[i][1]))
                    child.find('bndbox').find('ymax').text=str(int(ymax+shift_info[i][1]))
                    child.find('bndbox').find('xmin').text=str(int(xmin+shift_info[i][0]))
                    child.find('bndbox').find('xmax').text=str(int(xmax+shift_info[i][0]))
                tree.write(self.xml_folder_path+'/agmtD_'+item)
        print('augment_D done!')

def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    from PyQt5.QtGui import QIcon
    # 设置程序的全局图标
    app.setWindowIcon(QIcon("libs\\icons\\icon_new.png"))  # 替换为你的图标文件路径
    
    # 创建 QTranslator 实例
    translator = QTranslator()
    # 获取当前系统的语言环境
    locale = QLocale.system()
    # 加载 zh_CN.qm 文件
    if locale.language() == QLocale.Chinese and translator.load("libs\\translate\\zh_CN.qm"):
        # 安装翻译器到应用程序
        QCoreApplication.installTranslator(translator)
    
    app.setApplicationName(__appname__)
    # app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    # Usage : labelImg.py image predefClassFile saveDir
    win = MainWindow(argv[1] if len(argv) >= 2 else None,
                     argv[2] if len(argv) >= 3 else None,
                     argv[3] if len(argv) >= 4 else None,
                     argv[4] if len(argv) >= 5 else None,
                     argv[5] if len(argv) >= 6 else None)
    win.show()
    return app, win


def main():
    '''construct main app and run it'''
    app, _win = get_main_app(sys.argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
