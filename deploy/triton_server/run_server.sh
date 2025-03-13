# run server
docker run --gpus=1 --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.09-py3 tritonserver --model-repository=/models
# stop server
# docker kill containorId