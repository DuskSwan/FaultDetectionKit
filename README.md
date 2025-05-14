# PyTorch 项目模板 自用

fork自[L1aoXingyu/Deep-Learning-Project-Template](https://github.com/L1aoXingyu/Deep-Learning-Project-Template)

## Requirements

- [loguru](https://github.com/Delgan/loguru) (Python logging made stupidly simple)
- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform)
- [ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)（可选，也不是非要用）
- [lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) (Lightning organizes PyTorch code to remove boilerplate and unlock scalability)（想用这个）

## 说明

目录结构如下（省略了每个目录中的__init__.py）

```shell
├──  train_model.py # 实际运行的脚本
│ 
│ 
├──  config
│    └── defaults.py  - 默认配置文件
│    └── test_config.yml  - 实验配置文件
│ 
├──  datasets      - 多个数据集
│    └── dataset1  - 数据集1
│
├──  data_load     - 加载数据集的脚本
│    └── transforms  - 预处理用的一切函数
│    └── build.py     - 产生data loader
│
│
├──  engine
│   ├── trainer.py     - 定义训练过程
│   └── inference.py   - 定义推理过程
│
│
├── modeling            - 该目录下定义模型
│   └── example_model.py
│
│
├── solver             
│   └── build.py           - 产生求解器
│   └── lr_scheduler.py    - 定义学习率调度器
│   
│ 
└── utils            - 定义实用工具
     ├── logger.py
     └── any_other_utils_you_need
```
