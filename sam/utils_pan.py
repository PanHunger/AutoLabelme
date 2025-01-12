import numpy as np
import json
import os

def get_box_from_labelme_json(json_path, label_name, shape_type='rectangle'):
    with open(json_path, 'r') as f:
        data = json.load(f)
    shapes = data['shapes']
    for shape in shapes:
        if shape['label'] == label_name and shape['shape_type'] == shape_type:
            points = shape['points']
            x_min, y_min = min(np.array(points)[:, 0]), min(np.array(points)[:, 1])
            x_max, y_max = max(np.array(points)[:, 0]), max(np.array(points)[:, 1])
    return np.array([x_min, y_min, x_max, y_max])


# 读取现有的 JSON 文件内容
def load_json(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        # 如果文件不存在，则返回一个空的基础结构
        return {
            "version": "5.5.0",
            "flags": {},
            "shapes": [],
            "imagePath": "5.bmp",
            "imageData": None,
            "imageHeight": 864,
            "imageWidth": 480
        }

# 保存更新后的 JSON 数据
def save_json(data, json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# 获取多边形的点
def get_polygon_points(polygon):
    return polygon.reshape((-1, 2)).tolist()  # 转换为列表格式

# 更新或添加多边形
def update_json_with_polygon(json_data, label, polygon):
    points = get_polygon_points(polygon)
    # 查找是否已存在相同标签的polygon
    for shape in json_data['shapes']:
        if shape['label'] == label:
            shape['points'] = points  # 更新 points
            break
    else:
        # 如果没有找到相同标签，添加新的形状
        json_data['shapes'].append({
            "label": label,
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        })


if __name__ == '__main__':
    input_box = get_box_from_labelme_json('6.json', 
                                      label_name='pingti', shape_type='polygon')
    print(input_box)
    
    # 定义JSON文件路径
    json_file_path = '6.json'
    # 加载现有的 JSON 数据
    json_data = load_json(json_file_path)
    
    # 假设我们要更新或添加的标签
    label = 'pingti'
    
    # 更新 JSON 数据
    polygon = np.array([[[341.8, 658.4], [339.0, 660.0], [340.3, 665.9], [343.8, 664.7], [345.2, 659.9], [344.8, 658.6]]])
    update_json_with_polygon(json_data, label, polygon)
    
    # 保存更新后的 JSON 文件
    save_json(json_data, json_file_path)
    print(f'更新完成，数据已保存到 {json_file_path}')
