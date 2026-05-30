import os
from typing import List, Dict
from dataclasses import dataclass, asdict
import requests
import json
import grpc
from urllib.parse import urlencode

# Importing protobuf
from linke.serving.torchserve.proto import (
    inference_pb2_grpc,
    inference_pb2,
    management_pb2_grpc,
    management_pb2,
)


@dataclass
class ModelRegisterSpec:
    url: str  # remote, or when local, the .mar file must be in model_store/ folder
    model_name: str
    handler: str = None
    runtime: str = "PYTHON"
    batch_size: int = 1
    max_batch_delay: int = 100  # milliseconds
    initial_workers: int = 1
    synchronous: bool = False
    response_timeout: int = 120  # seconds

    def as_dict(self):
        # Return fields ignoring None
        return {k: v for k, v in asdict(self).items() if v is not None}


class TorchServeHttpClient:
    def __init__(
        self,
        url: str,
        model_name: str,
        inference_key: str,
        inference_port: str = "8080",
        management_key: str = None,
        management_port: str = "8081",
    ):
        self.url = url
        self.model_name = model_name
        self.inference_port = inference_port
        self.inference_key = inference_key
        self.management_port = management_port
        self.management_key = management_key

    def ping(self) -> requests.Response:
        url = f"{self.url}:{self.inference_port}/ping"
        headers = {
            "Authorization": f"Bearer {self.inference_key}",
        }
        response = requests.get(url, headers=headers)
        return response

    def predict(
        self, data: Dict, model_name: str = None
    ) -> requests.Response:
        """Inference API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.inference_key}",
        }
        url = f"{self.url}:{self.inference_port}/predictions/{model_name or self.model_name}"
        json_data = json.dumps(data)
        response = requests.post(url, data=json_data, headers=headers)
        return response

    def list_models(self, limit: int = 10) -> requests.Response:
        """Management API"""
        url = f"{self.url}:{self.management_port}/models"
        if limit is not None:
            url = f"{url}?limit={limit}"
        headers = {
            "Authorization": f"Bearer {self.management_key}",
        }
        response = requests.get(url, headers=headers)
        return response

    def register_model(
        self, model: ModelRegisterSpec
    ) -> requests.Response:
        """Management API"""
        params = model.as_dict()
        query_string = urlencode(params)
        url = f"{self.url}:{self.management_port}/models?{query_string}"
        headers = {
            "Authorization": f"Bearer {self.management_key}",
        }
        response = requests.post(url, headers=headers)
        return response

    def unregister_model(
        self, model_name: str, version: str
    ) -> requests.Response:
        """Management API"""
        url = f"{self.url}/{self.management_port}/models/{model_name}/{version}"
        headers = {
            "Authorization": f"Bearer {self.management_key}",
        }
        response = requests.delete(url, headers=headers)
        return response


class TorchServeGrpcClient:
    def __init__(
        self,
        url: str,
        model_name: str,
        inference_key: str,
        inference_port: str = "8080",
        management_key: str = None,
        management_port: str = "8081",
        certificate_file: str = None,
    ):

        self.url = url
        self.model_name = model_name
        self.inference_port = inference_port
        self.inference_key = inference_key
        self.management_port = management_port
        self.management_key = management_key

        self.get_inference_stub(certificate_file)
        self.get_management_stub(certificate_file)

    def _get_channel(self, url: str, certificate_file: str = None):
        if certificate_file and os.path.isfile(certificate_file):
            with open(certificate_file, "rb") as fid:
                credentials = grpc.ssl_channel_credentials(fid.read())
            channel = grpc.secure_channel(url, credentials=credentials)
        else:
            channel = grpc.insuecure_channel(url)
        return channel

    def get_inference_stub(self, certificate_file: str = None):
        url = f"{self.url}:{self.inference_port}"
        channel = self._get_channel(url, certificate_file)
        stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
        self.inference_stub = stub

    def get_management_stub(self, certificate_file: str = None):
        if self.management_key:
            self.management_stub = None
            return
        url = f"{self.url}:{self.management_port}"
        channel = self._get_channel(url, certificate_file)
        stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
        self.management_stub = stub

    def predict(
        self, data: Dict, schema: Dict, model_name: str = None
    ) -> requests.Response:
        headers = {
            "Content-Type": "application/text",
            f"Authorization": "Bearer {self.inference_key}",
        }
        prediction_request = inference_pb2.PredictionRequest(
            model_name=model_name or self.model_name,
        )
        prediction = self.inference_stub.Predictions(prediction_request)
        return prediction

    def list_models(self, limit: int = 10):
        list_model_request_object = management_pb2.ListModelsRequest(
            limit=limit
        )
        return self.management_stub.ListModels(
            list_model_request_object
        )
