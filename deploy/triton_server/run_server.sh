# run server
docker run --gpus=all --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.09-py3 tritonserver --model-repository=/models
# stop and remove server
# docker kill containorId
# stop/start server
# docker stop/start containorId