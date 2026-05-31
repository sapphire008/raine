"""
Push model artifacts to various cloud services for serving
"""

from kubernetes import client, config
from kubernetes.stream import stream
import tempfile
import os


class KubernetesPusher:
    def __init__(
        self,
        namespace="default",
        pod_name="remote-model-store-pod",
        remote_kube_config_path: str = None,
    ):
        # Load the Kubernetes configuration
        if remote_kube_config_path:
            config.load_kube_config(
                remote_kube_config_path
            )  # For local kubectl config
        else:
            try:
                config.load_kube_config()  # For local kubectl config
            except:
                config.load_incluster_config()  # If running inside a pod

        # Create API client
        self.v1 = client.CoreV1Api()

        # Define the pod and namespace
        self.pod_name = pod_name
        self.namespace = namespace

    def exec_command(self, command):
        return stream(
            self.v1.connect_get_namespaced_pod_exec,
            self.pod_name,
            self.namespace,
            command=command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
        )

    # Function to copy file to pod
    def copy_file_to_pod(self, local_path, pod_path):
        with open(local_path, "rb") as file:
            data = file.read()

        self.exec_command(["mkdir", "-p", os.path.dirname(pod_path)])

        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(data)
            temp_file.flush()

            client.CoreV1Api().read_namespaced_pod(
                name=self.pod_name, namespace=self.namespace
            )

            self.exec_command(
                ["cp", "/dev/stdin", pod_path],
                stdin=open(temp_file.name, "rb"),
            )


class StoragePusher:
    def __init__(self, client):
        pass

    def push(self):
        pass


kubernetes_pusher = KubernetesPusher()
# Create directories
kubernetes_pusher.exec_command(["mkdir", "-p", "/pv/model-store/"])
kubernetes_pusher.exec_command(["mkdir", "-p", "/pv/config/"])

# Copy files
kubernetes_pusher.copy_file_to_pod(
    "squeezenet1_1.mar", "/pv/model-store/squeezenet1_1.mar"
)
kubernetes_pusher.copy_file_to_pod(
    "mnist.mar", "/pv/model-store/mnist.mar"
)
kubernetes_pusher.copy_file_to_pod(
    "config.properties", "/pv/config/config.properties"
)
