
# TensorRT Quantization Tutorial
## 基础知识
- **视频教程**：[B站链接](https://www.bilibili.com/video/BV18L41197Uz/?spm_id_from=333.788&vd_source=eefa4b6e337f16d87d87c2c357db8ca7)
### 目录

1. **模型量化原理**
   - 1.1 量化的定义及意义
     - 1.1.1 模型权重分析
     - 1.1.2 量化的意义
   - 1.2 对称量化与非对称量化
     - 1.2.1 对称量化的定义
     - 1.2.2 对称量化代码手写
     - 1.2.3 非对称量化的定义
     - 1.2.4 非对称量化代码手写
   - 1.3 动态范围的常用计算方法
     - 1.3.1 Max
     - 1.3.2 Histgram
     - 1.3.3 Entropy
   - 1.4 PTQ 与 QAT 介绍
   - 1.5 手写一个带 op 的量化程序
2. **TensorRT Quantization Library**
   - 2.1 Quantizer 的理解
   - 2.2 InputQuant/MixQuant 的理解
   - 2.3 自动插入 QDQ 节点
   - 2.4 手动插入 QDQ 节点
   - 2.5 如何量化一个自定义层
   - 2.6 敏感层分析
   - 2.7 踩坑实录
