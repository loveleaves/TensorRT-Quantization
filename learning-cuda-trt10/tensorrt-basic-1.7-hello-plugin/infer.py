import onnxruntime as ort
import numpy as np

# 使用非dynamo=True方式导出模型，如果要在onnxruntime导入使用，首先需要再添加onnxruntime自定义算子
# ref: https://onnxruntime.ai/docs/reference/operators/add-custom-op.html
session = ort.InferenceSession("demo.onnx", 
                                providers=['CPUExecutionProvider'])

if __name__ == "__main__":
    input_data = np.array([
        1,   1,   1,
        1,   1,   1,
        1,   1,   1,
        -1,   1,   1,
        1,   0,   1,
        1,   1,   -1], dtype=np.float32).reshape((2,1,1,3,3))
    for item in input_data:
        pred = session.run(["div"], {"x": item})[0]
        print("pred: ", pred)