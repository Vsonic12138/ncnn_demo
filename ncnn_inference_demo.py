import ncnn
import numpy as np

def test_model_inference(param_file, bin_file, input_data):
    # 创建Net对象
    net = ncnn.Net()

    # 加载模型
    if net.load_param(param_file) != 0 or net.load_model(bin_file) != 0:
        print("模型加载失败")
        return

    # 创建Extractor对象
    ex = net.create_extractor()

    # 设置输入
    ex.input("in0", ncnn.Mat(input_data))

    # 进行推理
    ret, output = ex.extract("out0")
    if ret != 0:
        print("推理失败")
        return

    # 将ncnn.Mat转换为NumPy数组
    output_np = np.array(output)

    # 输出结果
    print("推理成功，输出结果:", output_np)

# 示例：假设你的模型输入是一个640x640的图像
input_data = np.random.rand(640, 640, 3).astype(np.float32)
test_model_inference('yolov5s.ncnn.param', 'yolov5s.ncnn.bin', input_data)
