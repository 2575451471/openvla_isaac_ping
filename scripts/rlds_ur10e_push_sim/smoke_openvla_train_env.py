#!/usr/bin/env python3
"""Quick check that openvla conda env can see GPU + load TF/TFDS/peft."""
import google.protobuf as pb
import peft
import tensorflow as tf
import tensorflow_datasets as tfds
import torch

print("protobuf", pb.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if torch.cuda.is_available():
    print("free_GiB", torch.cuda.mem_get_info()[0] / 1024**3)
print("peft", peft.__version__, "tf", tf.__version__, "tfds", tfds.__version__)
print("SMOKE_OK")
