import os
import shutil
import tensorrt as trt

from optimum.onnxruntime import ORTModelForCausalLM
from .trt_utils import build_engine, save_engine

if __name__ == "__main__":
    repository_path = "/var/triton/gpt2/repository/"
    model_id = "distilgpt2"
    model = ORTModelForCausalLM.from_pretrained(
        model_id, from_transformers=True
    )
    model.save_pretrained("./", file_name="model.onnx")
    trt_logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(trt_logger)
    engine = build_engine(
        runtime=runtime,
        onnx_file_path="model.onnx",
        logger=trt_logger,
        min_shape=(1, 1),
        optimal_shape=(1, 128),  # num beam, batch size
        max_shape=(1, 384),  # num beam, batch size
        workspace_size=10000 * 1024 ** 2,
        fp16=True,
        int8=False,
    )

    # Save tokenizer for use by triton
    tokenizer_path = repository_path + "encoder/"
    os.makedirs(tokenizer_path, exist_ok=True)
    shutil.copy("triton-config/encoder/config.pbtxt", tokenizer_path + "config.pbtxt")
    os.makedirs(tokenizer_path + "1/", exist_ok=True)
    shutil.copy("triton-config/encoder/model.py", tokenizer_path + "1/model.py")

    # Save model for use by triton
    model_path = repository_path + "model/"
    os.makedirs(model_path + "1/", exist_ok=True)
    shutil.copy("triton-config/model/config.pbtxt", model_path + "config.pbtxt")
    save_engine(engine, "model.plan")
    shutil.copy("model.plan", model_path + "1/config.pbtxt")

    # Save pipeline for use by triton
    transformer_path = repository_path + "transformer/"
    os.makedirs(transformer_path + "1/", exist_ok=True)
    shutil.copy("triton-config/transformer/config.pbtxt", transformer_path + "config.pbtxt")
