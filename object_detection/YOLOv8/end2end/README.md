# end to end

## Export End2End ONNX with NMS

You can export your onnx model by `ultralytics` API and add postprocess such as bbox decoder and `NMS` into ONNX model at the same time.

``` shell
python3 export_detection.py \
--weights yolov8s.pt \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--opset 11 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0
```

### Description of all arguments

- `--weights` : The PyTorch model you trained.
- `--iou-thres` : IOU threshold for NMS plugin.
- `--conf-thres` : Confidence threshold for NMS plugin.
- `--topk` : Max number of detection bboxes.
- `--opset` : ONNX opset version, default is 11.
- `--sim` : Whether to simplify your onnx model.
- `--input-shape` : Input shape for you model, should be 4 dimensions.
- `--device` : The CUDA deivce you export engine .

You will get an onnx model whose prefix is the same as input weights.

## ONNX -> TensorRT
same with normal, see [normal](../normal/README.md)

## build and infer
same with normal, see [normal](../normal/README.md)

## Note
The difference between normal and end2end is that:
1. normal can change these parameters(i.e. score_thres, iou_thres, topk, num_labels), but end2end can't after model been converted.
2. end2end removed NMS for speed up.