# HW2 Baseline Code 使用指南

欢迎使用HW2 baseline code。请按照以下步骤配置和运行本项目。

## 安装依赖

首先，您需要确保已安装Python 3环境。然后，使用以下命令安装项目所需的所有依赖包：

```bash
pip install -r requirements.txt
```

## 配置ZhipuAPI Key

本项目依赖于ZhipuAPI服务。在使用之前，您需要获取并配置ZhipuAPI的访问密钥。请按照以下步骤进行配置：

1. 前往ZhipuAPI的官方网站，注册并获取您的API Key。
2. 运行`export ZHIPUAI_API_KEY=your_api_key_here`，请将`your_api_key_here`替换为您从ZhipuAPI获取的实际API Key。

## 运行代码

在完成上述安装和配置后，您可以通过以下命令运行项目中的`run.py`脚本：

```bash
python run.py
```

如果一切配置正确，您将看到输出结果。

如有任何疑问，请在小作业讨论群提问。