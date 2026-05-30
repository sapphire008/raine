import sys
import os
import torch
import shutil
import requests
import json
import subprocess
import time

base_dir = os.path.abspath(os.path.realpath(
    os.path.join(os.path.dirname(__file__), "../..")
))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

sub_dir = "examples/serving_torch_model"
deployment_dir = "deployment"

from raine.serving.torchserve.save_model import export_to_model_archive
from examples.serving_torch_model.model import MaskNet, FeatureSpec


# %% Create the untrained model and save weights
# feature_specs = {
#     "day_of_week": FeatureSpec(
#         type="categorical",
#         embed_size=10,
#         vocab_size=7,
#         padding_idx=0,
#     ),
#     "hour_of_day": FeatureSpec(
#         type="categorical", embed_size=10, vocab_size=24
#     ),
#     "account_tenure": FeatureSpec(
#         type="categorical",
#         embed_size=10,
#         vocab_size=5,
#         padding_idx=0,
#     ),
#     "payment_tier": FeatureSpec(
#         type="categorical",
#         embed_size=10,
#         vocab_size=3,
#         padding_idx=0,
#     ),
#     "watch_history": FeatureSpec(
#         type="categorical",
#         embed_size=64,
#         vocab_size=500,
#         sequence_len=50,
#         padding_idx=0,
#     ),
#     "percent_watched": FeatureSpec(
#         type="numerical", embed_size=10, sequence_len=50
#     ),
# }
# model = MaskNet(feature_specs)
# model.compile()

# torch.save(
#     model.state_dict(), 
#     os.path.join(base_dir, sub_dir, "model.pth")
# )

# # Exporting model configs
# model.to_config(
#     os.path.join(base_dir, sub_dir, "model_config.yaml")
# )



#%% Save .mar file
# This simply creates a .zip file. You can unzip the folder 
# and see the contents of it using unzip model.mar
export_to_model_archive(
    model_name="masknet_recommender", model_version="1.0.0", 
    model_file=os.path.join(base_dir, sub_dir, "model.py"),
    serialized_file=os.path.join(base_dir, sub_dir, "model.pth"),
    handler_file=os.path.join(base_dir, sub_dir, "handler.py"),
    config_file=os.path.join(base_dir, sub_dir, "torchserve_config.yaml"),
    export_path=os.path.join(base_dir, sub_dir),
    extra_files=[os.path.join(base_dir, sub_dir, "model_config.yaml")],
    overwrite=True,
)

# %% Move the compiled files to proper deployment folder
destination = os.path.join(base_dir, sub_dir, deployment_dir, "model-store")
os.makedirs(destination, exist_ok=True)
shutil.copy(
    os.path.join(base_dir, sub_dir, "masknet_recommender.mar"), 
    os.path.join(destination,"masknet_recommender.mar")
)

shutil.copy(os.path.join(base_dir, sub_dir, "config.properties"),
   os.path.join(base_dir, sub_dir, deployment_dir, "config.properties"))

os.makedirs(os.path.join(destination, "wf-store"), exist_ok=True)


# %% Start the torchserve service locally
cmd = [
    "torchserve", "--start",
    "--ncs", # disable snapshot
    "--ts-config", os.path.join(base_dir, sub_dir, "deployment/config.properties"),
    "--model-store", os.path.join(base_dir, sub_dir, deployment_dir, "model-store"),
    "--models", "masknet=masknet_recommender.mar"
]
print(" ".join(cmd))
subprocess.call(cmd)
time.sleep(3.0)
# Open the key file and get the inference key
with open("./key_file.json", "r") as fid:
    INFERENCE_KEY = json.load(fid)["inference"]["key"]

#%% Check status
# Ping if the service is healthy. TorchServe requires a key when making requests
# There is a key_file.json key found in the directory when torchserve is called
# !curl http://0.0.0.0:9090/ping -H "Authorization: Bearer <inference key>"
# Expecting:
# {
#   "status": "Healthy"
# }

url = "http://0.0.0.0:9090/ping"
headers = {
    "Authorization": "Bearer {inference_key}".format(inference_key=INFERENCE_KEY)
}
response = requests.get(url, headers=headers)
print(f"Status Code: {response.status_code}")
print(f"Response Content: {response.text}")

# %% Make inference using Post
#!curl -X POST http://0.0.0.0:9090/predictions/masknet -T sample.jpg
url = "http://0.0.0.0:9090/predictions/masknet"
batch_size = 1
seq_len = 50
data = {
     "day_of_week": [4],
     "hour_of_day": [15],
     "account_tenure": [3],
     "payment_tier": [1],
     "watch_history": [torch.linspace(1, 500, 50).round().long().tolist()],
     "percent_watched": [torch.linspace(0.001, 0.999, 50).tolist()],
 }

# Convert the dictionary to a JSON string
json_data = json.dumps(data)

# Set the content type header
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer {inference_key}".format(inference_key=INFERENCE_KEY),
}

# Send the POST request
response = requests.post(url, data=json_data, headers=headers)

# Print the response
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")

#%%
subprocess.call(["torchserve", "--stop"])
