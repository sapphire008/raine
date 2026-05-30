import subprocess

def build_docker_image(image_uri, docker_file_path="./Dockerfile", platform="linux/amd64"):
    subprocess.call(
        [
            "docker",
            "build",
            "-f",
            docker_file_path,
            "-t",
            image_uri,
            "--platform",
            platform,
            ".",
        ]
    )


def push_docker_image(image_uri):
    subprocess.call(["docker", "push", image_uri])
