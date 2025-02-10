import ncnn

def get_model_io_names(param_file, bin_file):
    # 创建Net对象
    net = ncnn.Net()

    # 加载模型参数和二进制文件
    net.load_param(param_file)
    net.load_model(bin_file)

    # 获取输入和输出张量名称
    input_names = net.input_names()
    output_names = net.output_names()

    return input_names, output_names

# 示例：假设你的模型文件名为'model.param'和'model.bin'
param_file = 'yolov5s.param'
bin_file = 'yolov5s.bin'

input_names, output_names = get_model_io_names(param_file, bin_file)

print("输入张量名称:", input_names)
print("输出张量名称:", output_names)
