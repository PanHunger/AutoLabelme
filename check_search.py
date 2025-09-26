# test_model_structure.py
import os
import glob

def check_model_structure(model_path):
    """检查模型文件夹结构"""
    print(f"检查模型文件夹: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ 模型文件夹不存在")
        return
    
    # 检查权重文件
    pt_files = glob.glob(os.path.join(model_path, '**', '*.pt'), recursive=True)
    print(f"权重文件: {len(pt_files)} 个")
    for pt in pt_files[:5]:  # 显示前 5 个
        print(f"  - {os.path.basename(pt)} ({os.path.getsize(pt)/1024/1024:.1f} MB)")
    
    # 检查参数文件
    yaml_files = glob.glob(os.path.join(model_path, '**', '*.yaml'), recursive=True)
    yaml_files.extend(glob.glob(os.path.join(model_path, '**', '*.yml'), recursive=True))
    print(f"参数文件: {len(yaml_files)} 个")
    for yaml_file in yaml_files:
        print(f"  - {yaml_file}")
    
    # 检查图像文件
    image_files = glob.glob(os.path.join(model_path, '**', '*.jpg'), recursive=True)
    image_files.extend(glob.glob(os.path.join(model_path, '**', '*.png'), recursive=True))
    image_files.extend(glob.glob(os.path.join(model_path, '**', '*.jpeg'), recursive=True))
    print(f"图像文件: {len(image_files)} 个")
    
    # 检查评估结果图像
    result_images = []
    for pattern in ['*confusion*', '*precision*', '*recall*', '*f1*', '*result*']:
        result_images.extend(glob.glob(os.path.join(model_path, '**', pattern + '.png'), recursive=True))
    print(f"评估图像: {len(result_images)} 个")
    for img in result_images:
        print(f"  - {os.path.basename(img)}")

if __name__ == '__main__':
    model_base = '../../yolo_weights'
    if os.path.exists(model_base):
        for model_dir in os.listdir(model_base):
            model_path = os.path.join(model_base, model_dir)
            if os.path.isdir(model_path):
                check_model_structure(model_path)
                print("-" * 50)
    else:
        print(f"❌ 模型基础目录不存在: {model_base}")