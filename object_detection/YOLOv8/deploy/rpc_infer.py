import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from functools import partial
import cv2
import sys
import time


def preprocessor(frame):
    x = cv2.resize(frame, (640, 640))
    image_data = np.array(x).astype(np.float32) / 255.0  # Normalize to [0, 1] range
    image_data = np.transpose(image_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
    return image_data

classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog',
                17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
                31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
                48: 'sandwich', 49: 'orange', 50: 'broccoli',
                51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
                58: 'potted plant', 59: 'bed', 60: 'dining table',
                61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
                67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
                71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
                77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
color_palette = np.random.uniform(0, 255, size=(len(classes), 3))

def draw_detections(img, box, score, class_id):
    x1, y1, w, h = box
    color = color_palette[class_id]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    text = f"Class {classes[class_id]}: {score:.2f}"
    cv2.putText(img, text, (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

def clamp(val, min, max):
    if val < min:
        return min
    elif val > max:
        return max
    else:
        return val

def postprocessor(results, frame, confidence):
    # iou-thres, top-k and conf-thres already configured in the model
    # iou-thres=0.65, conf-thres=0.25
    img_height, img_width = frame.shape[:2]
    input_width, input_height = (640, 640)
    x_factor = img_width / input_width
    y_factor = img_height / input_height

    num_dets, bboxes, scores, labels = results
    for i in range(num_dets):
        score = scores[i]
        if score<confidence:
            continue
        x1, y1, x2, y2 = bboxes[i]
        x1 = clamp(x1 * x_factor, 0, img_width)
        y1 = clamp(y1 * y_factor, 0, img_height)
        x2 = clamp(x2 * x_factor, 0, img_width)
        y2 = clamp(y2 * y_factor, 0, img_height)
        box = [x1, y1, x2-x1, y2-y1]
        class_id = labels[i]
        frame = draw_detections(frame, box, score, class_id)
    return frame

def callback(rep_result, result, error):
    if error:
        rep_result.append(error)
    else:
        rep_result.append(result)

def main():
    # Create a rpc client for inference server running in localhost:8000.
    triton_client = grpcclient.InferenceServerClient(
        url="localhost:8001", verbose=False
    )
    
    inputs = []
    outputs = []
    imageData = cv2.imread("workspace/bus.jpg")
    imageData = preprocessor(imageData)
    inputs.append(grpcclient.InferInput('images', imageData.shape, "FP32"))
    inputs[0].set_data_from_numpy(imageData)
    outputs.append(grpcclient.InferRequestedOutput('num_dets'))
    outputs.append(grpcclient.InferRequestedOutput('bboxes'))
    outputs.append(grpcclient.InferRequestedOutput('scores'))
    outputs.append(grpcclient.InferRequestedOutput('labels'))
    
    # list to hold the results of inference.
    rep_result = []
    # Send request to the inference server. Get results for both output tensors.
    try:
        triton_client.async_infer(
            model_name="yolo",
            model_version="1",
            inputs=inputs,
            callback=partial(callback, rep_result),
            outputs=outputs
        )
        
        # Wait until the results are available in rep_result
        time_out = 10
        while (len(rep_result) == 0) and time_out > 0:
            time_out = time_out - 1
            time.sleep(1)

        # Display and validate the available results
        if len(rep_result) == 1:
            # Check for the errors
            if type(rep_result[0]) == InferenceServerException:
                print(rep_result[0])
                sys.exit(1)
            
        num_dets = np.squeeze(rep_result[0].as_numpy('num_dets'))
        bboxes = np.squeeze(rep_result[0].as_numpy('bboxes'))
        scores = np.squeeze(rep_result[0].as_numpy('scores'))
        labels = np.squeeze(rep_result[0].as_numpy('labels'))
        originData = cv2.imread("workspace/bus.jpg")
        result = (num_dets, bboxes, scores, labels)
        imageOutput = postprocessor(result, originData, 0.35)
        cv2.imwrite("workspace/output.jpg", imageOutput)
        print("rpc request succeed!")
        
    except InferenceServerException as e:
        print(e)

if __name__ == '__main__':
    main()
