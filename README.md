# 介绍
适用于ncnn的一些demo文件，比如简单的推理、打印输入输出张量等操作。还有ncnn的一些工程存储在公司电脑的桌面和WSL2当中，后续找时间进行整理。

# ncnn模型转化
YOLO模型转换NCNN模型需要先从pt模型转为torch模型，然后使用pnnx进行转换，最终可以由.pt模型直接转换为.bin和.param文件模型。pnnx的工程有待后续整理。

