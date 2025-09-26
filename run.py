#!/usr/bin/env python3
import os
import socket
from run_app import app, model_manager

def get_local_ip():
    """获取本机IP地址"""
    try:
        # 连接一个外部地址但不发送数据
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == '__main__':
    # 初始化模型管理器
    model_manager.scan_models()
    
    local_ip = get_local_ip()
    print(f"YOLO模型推理系统启动成功！")
    print(f"本地访问: http://127.0.0.1:5000")
    print(f"局域网访问: http://{local_ip}:5000")
    print("\n使用说明:")
    print("1. 将YOLO模型文件夹放在 'yolo_weights' 目录下")
    print("2. 每个模型文件夹应包含 best.pt 和 args.yaml 文件")
    print("3. 通过网页界面选择模型并进行推理")
    
    app.run(host='0.0.0.0', port=5000, debug=True)