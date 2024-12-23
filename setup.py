import os
from setuptools import setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"

setup(
    name="sd-dinio",
    author="Haoke Xiao",
    description="ZERO-SHOT CO-SALIENT OBJECT DETECTION FRAMEWORK",
    python_requires=">=3.8",
    py_modules=[],
    install_requires=[
        "loguru>=0.5.3",
        "faiss-cpu>=1.7.1",
        "matplotlib>=3.4.2",
        "Pillow>=9.4.0",
        "tqdm>=4.61.2",
        "pillow==9.4.0",
        "numpy>=1.21.0",
        "gdown>=3.13.0",
        "progress==1.6",
        "thop",
        f"mask2former @ file://localhost/{os.getcwd()}/sd-dino/third_party/Mask2Former/",
        f"pydensecrf @ file://localhost/{os.getcwd()}/A2S-v2/third_party/pydensecrf/",
        f"odise @ file://localhost/{os.getcwd()}/sd-dino/third_party/ODISE/"
    ],
    include_package_data=True,
)

# conda create -n zs-cosod python=3.10
# conda activate zs-cosod
# pip install setuptools==59.5.0
# pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# cd sam2
# pip install -e .
# cd sd-dino
# pip install -e .