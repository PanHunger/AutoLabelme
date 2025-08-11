Linux:
export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890

Windows:
$env:http_proxy="http://127.0.0.1:17890"
$env:https_proxy="http://127.0.0.1:17890"

# 打包环境 （安装者不需要执行）
conda env export -n pytorch_env > pytorch.yml

# conda环境创建
conda env create -f conda_environment.yml