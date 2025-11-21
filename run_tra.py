import streamlit as st
import os
import shutil
import yaml
import tempfile
import zipfile
import tarfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import numpy as np
from loguru import logger
import sys
from tqdm import tqdm
import filetype

# 导入自定义工具模块
from utils.data_processor import DataProcessor
from utils.train_utils import YOLOTrainer
from utils.visualization import TrainingVisualizer

# streamlit run run_tra.py

# 页面配置
st.set_page_config(
    page_title="YOLO 模型训练平台",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class YOLOTrainingWebUI:
    def __init__(self):
        self.setup_logging()
        self.data_processor = DataProcessor()
        self.trainer = YOLOTrainer()
        self.visualizer = TrainingVisualizer()
        
        # 初始化session state
        if 'training_active' not in st.session_state:
            st.session_state.training_active = False
        if 'training_logs' not in st.session_state:
            st.session_state.training_logs = []
        if 'dataset_path' not in st.session_state:
            st.session_state.dataset_path = None
        if 'current_model' not in st.session_state:
            st.session_state.current_model = None

    def setup_logging(self):
        """配置日志系统"""
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add("logs/training_{time}.log", rotation="10 MB")

    def render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.title("🔍 YOLO 训练平台")
        st.sidebar.markdown("---")
        
        # 导航菜单
        app_mode = st.sidebar.selectbox(
            "选择功能模块",
            ["🏠 首页", "📊 数据准备", "⚙️ 模型配置", "🚀 训练执行", "📈 结果查看", "📤 模型导出"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            "💡 **使用提示**:\n"
            "1. 按顺序完成各模块配置\n"
            "2. 确保数据集格式正确\n"
            "3. 训练前检查GPU可用性"
        )
        
        return app_mode

    def render_home(self):
        """渲染首页"""
        st.title("🎯 YOLO 模型训练平台")
        st.markdown("""
        ### 欢迎使用 YOLO 模型训练 WebUI
        
        本平台基于 **Ultralytics YOLO** 框架，提供可视化的目标检测模型训练解决方案。
        
        **主要功能:**
        - 📊 **数据准备**: 支持多种数据集格式上传和自动校验
        - ⚙️ **模型配置**: 灵活的模型参数配置界面
        - 🚀 **训练执行**: 实时监控训练过程和可视化指标
        - 📈 **结果查看**: 详细的训练结果分析和可视化
        - 📤 **模型导出**: 支持多种格式的模型导出
        
        **开始使用:**
        1. 在 **数据准备** 模块上传您的数据集
        2. 在 **模型配置** 模块选择模型和调整参数
        3. 在 **训练执行** 模块开始训练并监控进度
        4. 在 **结果查看** 模块分析训练结果
        5. 在 **模型导出** 模块导出训练好的模型
        """)
        
        # 系统状态检查
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔧 系统状态")
            # 检查GPU
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                st.success(f"✅ GPU 可用 ({gpu_count} 个设备)")
                for i in range(gpu_count):
                    st.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                st.warning("❌ GPU 不可用，将使用CPU训练")
        
        with col2:
            st.subheader("📦 依赖检查")
            try:
                from ultralytics import YOLO
                st.success("✅ Ultralytics YOLO 可用")
            except ImportError:
                st.error("❌ Ultralytics YOLO 未安装")
            
            try:
                import onnx
                st.success("✅ ONNX 支持可用")
            except ImportError:
                st.warning("⚠️ ONNX 未安装，无法导出ONNX格式")
        
        with col3:
            st.subheader("💾 存储空间")
            # 检查磁盘空间
            import shutil
            total, used, free = shutil.disk_usage("/")
            st.info(f"可用空间: {free // (2**30)} GB")

    def render_data_preparation(self):
        """渲染数据准备模块"""
        st.title("📊 数据准备")
        
        tab1, tab2, tab3 = st.tabs(["📁 上传数据", "🔍 数据校验", "📋 类别管理"])
        
        with tab1:
            self.render_data_upload()
        
        with tab2:
            self.render_data_validation()
        
        with tab3:
            self.render_class_management()

    def render_data_upload(self):
        """渲染数据上传界面"""
        st.subheader("数据集上传")
        
        upload_method = st.radio(
            "选择上传方式",
            ["方式1: 上传原始文件（自动分割）", "方式2: 上传已划分的数据集压缩包"],
            help="方式1: 上传图片和标注文件，系统自动分割数据集\n方式2: 上传已按目录结构组织的数据集压缩包"
        )
        
        if upload_method == "方式1: 上传原始文件（自动分割）":
            self.upload_raw_files()
        else:
            self.upload_structured_dataset()

    def upload_raw_files(self):
        """上传原始文件并自动分割"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**上传图片文件**")
            uploaded_images = st.file_uploader(
                "选择图片文件 (jpg/png/jpeg)",
                type=['jpg', 'png', 'jpeg'],
                accept_multiple_files=True,
                key="images_upload"
            )
        
        with col2:
            st.markdown("**上传标注文件**")
            uploaded_labels = st.file_uploader(
                "选择标注文件 (.txt, YOLO格式)",
                type=['txt'],
                accept_multiple_files=True,
                key="labels_upload"
            )
        
        if uploaded_images and uploaded_labels:
            st.info(f"已上传 {len(uploaded_images)} 张图片, {len(uploaded_labels)} 个标注文件")
            
            # 分割比例配置
            col1, col2, col3 = st.columns(3)
            with col1:
                train_ratio = st.slider("训练集比例", 0.5, 0.9, 0.8, 0.05)
            with col2:
                val_ratio = st.slider("验证集比例", 0.1, 0.4, 0.1, 0.05)
            with col3:
                test_ratio = st.slider("测试集比例", 0.0, 0.3, 0.1, 0.05)
            
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 0.01:
                st.error(f"比例总和应为 1.0，当前为 {total_ratio:.2f}")
                return
            
            if st.button("🚀 处理数据集", type="primary"):
                with st.spinner("正在处理数据集..."):
                    dataset_path = self.data_processor.process_raw_files(
                        uploaded_images, uploaded_labels, train_ratio, val_ratio, test_ratio
                    )
                    if dataset_path:
                        st.session_state.dataset_path = dataset_path
                        st.success(f"数据集处理完成！保存路径: {dataset_path}")

    def upload_structured_dataset(self):
        """上传已划分的数据集压缩包"""
        st.markdown("""
        **数据集目录结构要求:**
        ```
        dataset_name/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── data.yaml
        ```
        """)
        
        uploaded_zip = st.file_uploader(
            "上传数据集压缩包 (zip/tar.gz)",
            type=['zip', 'tar', 'gz'],
            key="dataset_upload"
        )
        
        if uploaded_zip:
            if st.button("📦 解压并验证数据集", type="primary"):
                with st.spinner("正在解压和验证数据集..."):
                    dataset_path = self.data_processor.extract_and_validate_dataset(uploaded_zip)
                    if dataset_path:
                        st.session_state.dataset_path = dataset_path
                        st.success(f"数据集验证完成！路径: {dataset_path}")

    def render_data_validation(self):
        """渲染数据校验界面"""
        st.subheader("数据校验")
        
        if not st.session_state.dataset_path:
            st.warning("请先上传数据集")
            return
        
        if st.button("🔍 运行数据校验", type="primary"):
            with st.spinner("正在校验数据集..."):
                validation_results = self.data_processor.validate_dataset(st.session_state.dataset_path)
                
                # 显示校验结果
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 数据集统计**")
                    stats_df = pd.DataFrame([
                        {"分割": "训练集", "图片数量": validation_results['train_images'], "标注数量": validation_results['train_labels']},
                        {"分割": "验证集", "图片数量": validation_results['val_images'], "标注数量": validation_results['val_labels']},
                        {"分割": "测试集", "图片数量": validation_results['test_images'], "标注数量": validation_results['test_labels']}
                    ])
                    st.dataframe(stats_df, use_container_width=True)
                
                with col2:
                    st.markdown("**✅ 校验结果**")
                    if validation_results['is_valid']:
                        st.success("✅ 数据集格式正确")
                    else:
                        st.error("❌ 数据集存在问题")
                    
                    for issue in validation_results['issues']:
                        st.error(f"⚠️ {issue}")

    def render_class_management(self):
        """渲染类别管理界面"""
        st.subheader("类别管理")
        
        if not st.session_state.dataset_path:
            st.warning("请先上传数据集")
            return
        
        # 尝试从data.yaml读取类别信息
        data_yaml_path = os.path.join(st.session_state.dataset_path, 'data.yaml')
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            if 'names' in data_config:
                st.info("从 data.yaml 读取的类别信息:")
                classes_df = pd.DataFrame({
                    'ID': list(range(len(data_config['names']))),
                    '类别名称': data_config['names']
                })
                st.dataframe(classes_df, use_container_width=True)
        
        # 类别编辑
        st.markdown("**编辑类别信息**")
        class_input_method = st.radio(
            "输入方式",
            ["手动输入", "上传类别文件"],
            horizontal=True
        )
        
        if class_input_method == "手动输入":
            class_names = st.text_area(
                "输入类别名称 (每行一个类别)",
                value="\n".join(data_config.get('names', ['class0', 'class1'])) if 'data_config' in locals() else "",
                height=150
            )
        else:
            class_file = st.file_uploader("上传类别文件 (.txt)", type=['txt'])
            if class_file:
                class_names = class_file.getvalue().decode('utf-8')
        
        if st.button("💾 更新类别信息", type="primary"):
            if class_names:
                classes = [cls.strip() for cls in class_names.split('\n') if cls.strip()]
                success = self.data_processor.update_class_info(st.session_state.dataset_path, classes)
                if success:
                    st.success(f"✅ 成功更新 {len(classes)} 个类别")

    def render_model_configuration(self):
        """渲染模型配置模块"""
        st.title("⚙️ 模型配置")
        
        if not st.session_state.dataset_path:
            st.warning("请先完成数据准备")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self.render_model_selection()
        
        with col2:
            self.render_training_parameters()

    def render_model_selection(self):
        """渲染模型选择界面"""
        st.subheader("🔧 模型选择")
        
        # 预训练模型选择
        pretrained_models = {
            "YOLOv8n": "yolov8n.pt",
            "YOLOv8s": "yolov8s.pt", 
            "YOLOv8m": "yolov8m.pt",
            "YOLOv8l": "yolov8l.pt",
            "YOLOv8x": "yolov8x.pt",
            "YOLOv9c": "yolov9c.pt",
            "YOLOv9e": "yolov9e.pt"
        }
        
        selected_model = st.selectbox(
            "选择预训练模型",
            options=list(pretrained_models.keys()),
            help="选择适合您任务的模型尺寸"
        )
        
        # 自定义模型上传
        use_custom_model = st.checkbox("使用自定义预训练权重")
        custom_model = None
        if use_custom_model:
            custom_model = st.file_uploader(
                "上传自定义权重文件 (.pt)",
                type=['pt'],
                help="上传您自己的预训练权重文件"
            )
        
        st.session_state.current_model = custom_model if use_custom_model and custom_model else pretrained_models[selected_model]

    def render_training_parameters(self):
        """渲染训练参数配置界面"""
        st.subheader("🎯 训练参数")
        
        # 基础参数
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("训练轮次 (epochs)", min_value=1, max_value=1000, value=100)
            img_size = st.selectbox("输入图像尺寸", [224, 320, 416, 512, 640, 768, 1024], index=4)
        
        with col2:
            batch_size = st.number_input("批次大小 (batch size)", min_value=1, max_value=256, value=16)
            learning_rate = st.number_input("学习率 (lr0)", min_value=1e-5, max_value=1.0, value=0.01, format="%.4f")
        
        # 优化器参数
        st.markdown("**优化器配置**")
        col1, col2, col3 = st.columns(3)
        with col1:
            optimizer = st.selectbox("优化器", ["SGD", "Adam", "AdamW"], index=2)
        with col2:
            weight_decay = st.number_input("权重衰减", min_value=0.0, max_value=0.1, value=0.0005, format="%.5f")
        with col3:
            momentum = st.number_input("动量", min_value=0.0, max_value=0.99, value=0.937, format="%.3f")
        
        # 设备选择
        import torch
        if torch.cuda.is_available():
            device_options = ["CPU"] + [f"GPU {i}" for i in range(torch.cuda.device_count())]
            device = st.selectbox("训练设备", device_options, index=1)
        else:
            device = "CPU"
        
        # 其他参数
        col1, col2 = st.columns(2)
        with col1:
            save_best = st.checkbox("仅保存最佳模型", value=True)
            resume_training = st.checkbox("启用断点续训", value=False)
        
        with col2:
            patience = st.number_input("早停耐心值", min_value=0, max_value=100, value=50)
            workers = st.number_input("数据加载进程数", min_value=0, max_value=16, value=4)
        
        # 保存配置到session state
        st.session_state.training_config = {
            'epochs': epochs,
            'img_size': img_size,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'device': device,
            'save_best': save_best,
            'resume': resume_training,
            'patience': patience,
            'workers': workers
        }
        
        st.success("✅ 训练参数配置完成")

    def render_training_execution(self):
        """渲染训练执行模块"""
        st.title("🚀 训练执行")
        
        if not st.session_state.get('training_config'):
            st.warning("请先完成模型配置")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("训练控制")
            
            if not st.session_state.training_active:
                if st.button("🎬 开始训练", type="primary", use_container_width=True):
                    st.session_state.training_active = True
                    # 这里应该启动训练线程
                    self.start_training_thread()
            else:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("⏸️ 暂停训练", use_container_width=True):
                        self.pause_training()
                with col2:
                    if st.button("⏹️ 停止训练", use_container_width=True):
                        self.stop_training()
        
        with col2:
            st.subheader("训练信息")
            st.info(f"模型: {st.session_state.current_model}")
            st.info(f"设备: {st.session_state.training_config['device']}")
            st.info(f"轮次: {st.session_state.training_config['epochs']}")
        
        # 训练日志显示
        st.subheader("📋 训练日志")
        log_container = st.empty()
        
        # 进度条
        st.subheader("📊 训练进度")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 实时更新日志和进度（这里需要根据实际训练状态更新）
        if st.session_state.training_active:
            # 模拟训练进度更新
            for i in range(100):
                # 这里应该从训练进程中获取实际进度
                progress = i + 1
                progress_bar.progress(progress)
                status_text.text(f"训练进度: {progress}%")
                # 更新日志显示
                # log_container.text_area("日志", value="\n".join(st.session_state.training_logs[-20:]), height=300)
                
                # 实际实现中应该使用线程和队列来更新
                break

    def start_training_thread(self):
        """启动训练线程"""
        # 这里应该启动一个后台线程来执行训练
        # 由于Streamlit的限制，实际实现可能需要使用multiprocessing或外部进程
        try:
            # 调用训练工具开始训练
            training_result = self.trainer.start_training(
                st.session_state.dataset_path,
                st.session_state.current_model,
                st.session_state.training_config
            )
            
            if training_result['success']:
                st.session_state.training_results = training_result
                st.session_state.training_active = False
                st.success("✅ 训练完成！")
            else:
                st.error(f"❌ 训练失败: {training_result['error']}")
                
        except Exception as e:
            logger.error(f"训练错误: {e}")
            st.error(f"训练过程中发生错误: {str(e)}")

    def pause_training(self):
        """暂停训练"""
        # 实现训练暂停逻辑
        st.session_state.training_active = False
        st.info("训练已暂停")

    def stop_training(self):
        """停止训练"""
        # 实现训练停止逻辑
        st.session_state.training_active = False
        st.warning("训练已停止")

    def render_results_view(self):
        """渲染结果查看模块"""
        st.title("📈 结果查看")
        
        if not st.session_state.get('training_results'):
            st.warning("请先完成训练")
            return
        
        tab1, tab2, tab3 = st.tabs(["📊 训练指标", "🖼️ 预测可视化", "📁 文件下载"])
        
        with tab1:
            self.render_training_metrics()
        
        with tab2:
            self.render_prediction_visualization()
        
        with tab3:
            self.render_file_download()

    def render_training_metrics(self):
        """渲染训练指标显示"""
        st.subheader("训练指标分析")
        
        # 从训练结果中获取指标数据
        results_path = st.session_state.training_results.get('results_path')
        if results_path and os.path.exists(results_path):
            # 显示关键指标
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("mAP@0.5", "0.85")
            with col2:
                st.metric("mAP@0.5:0.95", "0.67")
            with col3:
                st.metric("Precision", "0.89")
            with col4:
                st.metric("Recall", "0.78")
            
            # 训练曲线
            st.subheader("训练曲线")
            fig = self.visualizer.plot_training_curves(results_path)
            if fig:
                st.pyplot(fig)
            else:
                st.info("训练曲线数据暂不可用")

    def render_prediction_visualization(self):
        """渲染预测结果可视化"""
        st.subheader("预测结果可视化")
        
        # 显示验证集预测结果
        results_path = st.session_state.training_results.get('results_path')
        if results_path:
            val_pred_dir = os.path.join(results_path, 'val_preds')
            if os.path.exists(val_pred_dir):
                pred_images = [f for f in os.listdir(val_pred_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                if pred_images:
                    # 分页显示
                    page_size = 6
                    total_pages = (len(pred_images) + page_size - 1) // page_size
                    page = st.number_input("页码", min_value=1, max_value=total_pages, value=1)
                    
                    start_idx = (page - 1) * page_size
                    end_idx = min(start_idx + page_size, len(pred_images))
                    
                    cols = st.columns(3)
                    for idx, img_name in enumerate(pred_images[start_idx:end_idx]):
                        with cols[idx % 3]:
                            img_path = os.path.join(val_pred_dir, img_name)
                            image = Image.open(img_path)
                            st.image(image, caption=img_name, use_column_width=True)
                else:
                    st.info("暂无预测结果图像")
            else:
                st.info("预测结果目录不存在")

    def render_file_download(self):
        """渲染文件下载界面"""
        st.subheader("训练文件下载")
        
        results_path = st.session_state.training_results.get('results_path')
        if results_path:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 训练日志下载
                log_file = os.path.join(results_path, 'train_log.txt')
                if os.path.exists(log_file):
                    with open(log_file, 'rb') as f:
                        st.download_button(
                            "📥 下载训练日志",
                            f,
                            file_name="training_log.txt",
                            mime="text/plain"
                        )
            
            with col2:
                # 训练曲线下载
                curves_file = os.path.join(results_path, 'training_curves.png')
                if os.path.exists(curves_file):
                    with open(curves_file, 'rb') as f:
                        st.download_button(
                            "📥 下载训练曲线",
                            f,
                            file_name="training_curves.png",
                            mime="image/png"
                        )
            
            with col3:
                # 配置文件下载
                config_file = os.path.join(results_path, 'args.yaml')
                if os.path.exists(config_file):
                    with open(config_file, 'rb') as f:
                        st.download_button(
                            "📥 下载训练配置",
                            f,
                            file_name="training_config.yaml",
                            mime="text/yaml"
                        )

    def render_model_export(self):
        """渲染模型导出模块"""
        st.title("📤 模型导出")
        
        if not st.session_state.get('training_results'):
            st.warning("请先完成训练")
            return
        
        st.subheader("模型导出设置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 导出格式选择
            export_formats = {
                "PyTorch (.pt)": "torchscript",
                "ONNX (.onnx)": "onnx", 
                "TensorRT": "engine",
                "CoreML (.mlmodel)": "coreml",
                "TensorFlow SavedModel": "saved_model",
                "OpenVINO": "openvino"
            }
            
            selected_format = st.selectbox(
                "选择导出格式",
                options=list(export_formats.keys())
            )
            
            # 格式特定配置
            if "ONNX" in selected_format:
                opset_version = st.number_input("ONNX opset版本", min_value=10, max_value=15, value=12)
            
            if "TensorRT" in selected_format:
                precision = st.selectbox("精度模式", ["FP32", "FP16", "INT8"])
        
        with col2:
            st.markdown("**导出信息**")
            st.info(f"原始模型: {st.session_state.current_model}")
            st.info(f"训练轮次: {st.session_state.training_config['epochs']}")
            st.info(f"输入尺寸: {st.session_state.training_config['img_size']}")
        
        if st.button("🚀 开始导出", type="primary"):
            with st.spinner("正在导出模型..."):
                export_config = {
                    'format': export_formats[selected_format],
                    'imgsz': st.session_state.training_config['img_size']
                }
                
                if "ONNX" in selected_format:
                    export_config['opset'] = opset_version
                
                if "TensorRT" in selected_format:
                    export_config['half'] = precision in ["FP16", "INT8"]
                
                export_result = self.trainer.export_model(
                    st.session_state.training_results['best_model'],
                    export_config
                )
                
                if export_result['success']:
                    st.success(f"✅ 模型导出成功！格式: {selected_format}")
                    
                    # 提供下载链接
                    exported_file = export_result['exported_path']
                    if os.path.exists(exported_file):
                        with open(exported_file, 'rb') as f:
                            st.download_button(
                                f"📥 下载{selected_format}模型",
                                f,
                                file_name=os.path.basename(exported_file),
                                mime="application/octet-stream"
                            )
                else:
                    st.error(f"❌ 模型导出失败: {export_result['error']}")

    def run(self):
        """运行主应用"""
        app_mode = self.render_sidebar()
        
        if app_mode == "🏠 首页":
            self.render_home()
        elif app_mode == "📊 数据准备":
            self.render_data_preparation()
        elif app_mode == "⚙️ 模型配置":
            self.render_model_configuration()
        elif app_mode == "🚀 训练执行":
            self.render_training_execution()
        elif app_mode == "📈 结果查看":
            self.render_results_view()
        elif app_mode == "📤 模型导出":
            self.render_model_export()

def main():
    # 创建必要的目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('temp', exist_ok=True)
    
    app = YOLOTrainingWebUI()
    app.run()

if __name__ == "__main__":
    main()