import contextlib
import subprocess
from pathlib import Path

def run(triton_repo_path):
    # Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
    tag = "nvcr.io/nvidia/tritonserver:24.09-py3"  # 8.57 GB

    # Pull the image
    subprocess.call(f"docker pull {tag}", shell=True)

    # Run the Triton server and capture the container ID
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm --gpus all -v {triton_repo_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    print("#" * 30)
    print(f"service running on {container_id}")
    print("#" * 30)

def run_server():
    triton_repo_path = (Path("model_repository")).resolve()
    run(triton_repo_path)

def stop_remove_server(container_id):
    # Kill and remove the container at the end of the test
    subprocess.call(f"docker kill {container_id}", shell=True)

if __name__ == "__main__":
    run_server()
    # stop_remove_server("a53ed411f104")