import json
import os
import psutil
import GPUtil
import time
import subprocess
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, redirect, url_for

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 生产环境请更改

# 配置文件路径
CONFIG_FILE = 'config.json'

class ServerMonitor:
    def get_system_info(self):
        """获取系统信息"""
        # CPU信息
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # 内存信息
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # 磁盘信息
        disks = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "total": usage.total / (1024**3),
                    "used": usage.used / (1024**3),
                    "free": usage.free / (1024**3),
                    "usage": usage.percent
                })
            except:
                continue
        
        # GPU信息
        gpu_info = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "temperature": gpu.temperature
                })
        except:
            pass
        
        # 温度信息
        temp_info = {}
        try:
            temps = psutil.sensors_temperatures()
            for name, entries in temps.items():
                for entry in entries:
                    temp_info[f"{name}_{entry.label or 'current'}"] = entry.current
        except:
            temp_info = {"message": "温度信息不可用"}
        
        # 网络信息
        net_io = psutil.net_io_counters()
        
        # 进程信息
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
        
        return {
            "cpu": {
                "usage": cpu_percent,
                "cores": cpu_count,
                "frequency": cpu_freq.current if cpu_freq else "N/A"
            },
            "memory": {
                "total": memory.total / (1024**3),
                "used": memory.used / (1024**3),
                "available": memory.available / (1024**3),
                "usage": memory.percent,
                "swap_total": swap.total / (1024**3),
                "swap_used": swap.used / (1024**3)
            },
            "disks": disks,
            "gpus": gpu_info,
            "temperature": temp_info,
            "network": {
                "bytes_sent": net_io.bytes_sent / (1024**2),
                "bytes_recv": net_io.bytes_recv / (1024**2),
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            },
            "processes": processes[:10]  # 前10个进程
        }

class EnvironmentManager:
    def __init__(self):
        self.config_file = CONFIG_FILE
        self.load_config()
    
    def load_config(self):
        """加载配置"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "virtual_envs": {},
                "scheduled_tasks": [],
                "python_scripts": {},
                "settings": {
                    "refresh_interval": 5,
                    "auto_refresh": True
                }
            }
            self.save_config()
    
    def save_config(self):
        """保存配置"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def add_virtual_env(self, name, path):
        """添加虚拟环境"""
        self.config["virtual_envs"][name] = path
        self.save_config()
        return True
    
    def remove_virtual_env(self, name):
        """移除虚拟环境"""
        if name in self.config["virtual_envs"]:
            del self.config["virtual_envs"][name]
            self.save_config()
            return True
        return False
    
    def run_script_in_env(self, env_name, script_path, args=""):
        """在指定虚拟环境中运行脚本"""
        if env_name not in self.config["virtual_envs"]:
            return False, "虚拟环境不存在"
        
        env_path = self.config["virtual_envs"][env_name]
        python_path = os.path.join(env_path, "bin", "python")
        if not os.path.exists(python_path):
            python_path = os.path.join(env_path, "Scripts", "python.exe")
        
        if not os.path.exists(python_path):
            return False, "找不到Python解释器"
        
        try:
            cmd = [python_path, script_path]
            if args:
                cmd.extend(args.split())
                
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            return True, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "脚本执行超时"
        except Exception as e:
            return False, str(e)

# 初始化管理器
monitor = ServerMonitor()
env_manager = EnvironmentManager()

# 路由定义
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/environment')
def environment():
    return render_template('environment.html')

@app.route('/scripts')
def scripts():
    return render_template('scripts.html')

@app.route('/scheduling')
def scheduling():
    return render_template('scheduling.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

# API路由
@app.route('/api/system_info')
def api_system_info():
    return jsonify(monitor.get_system_info())

@app.route('/api/virtual_envs', methods=['GET', 'POST', 'DELETE'])
def api_virtual_envs():
    if request.method == 'GET':
        return jsonify(env_manager.config["virtual_envs"])
    
    elif request.method == 'POST':
        data = request.json
        name = data.get('name')
        path = data.get('path')
        
        if not name or not path:
            return jsonify({"success": False, "message": "名称和路径不能为空"})
        
        if name in env_manager.config["virtual_envs"]:
            return jsonify({"success": False, "message": "环境名称已存在"})
        
        if env_manager.add_virtual_env(name, path):
            return jsonify({"success": True, "message": "虚拟环境添加成功"})
        else:
            return jsonify({"success": False, "message": "添加虚拟环境失败"})
    
    elif request.method == 'DELETE':
        name = request.args.get('name')
        if env_manager.remove_virtual_env(name):
            return jsonify({"success": True, "message": "虚拟环境删除成功"})
        else:
            return jsonify({"success": False, "message": "虚拟环境不存在"})

@app.route('/api/run_script', methods=['POST'])
def api_run_script():
    data = request.json
    env_name = data.get('env_name')
    script_path = data.get('script_path')
    args = data.get('args', '')
    
    if not env_name or not script_path:
        return jsonify({"success": False, "message": "环境和脚本路径不能为空"})
    
    if not os.path.exists(script_path):
        return jsonify({"success": False, "message": "脚本文件不存在"})
    
    success, output = env_manager.run_script_in_env(env_name, script_path, args)
    
    return jsonify({
        "success": success,
        "message": "脚本执行成功" if success else "脚本执行失败",
        "output": output
    })

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if request.method == 'GET':
        return jsonify(env_manager.config.get("settings", {}))
    
    elif request.method == 'POST':
        data = request.json
        env_manager.config["settings"] = data
        env_manager.save_config()
        return jsonify({"success": True, "message": "设置已保存"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)