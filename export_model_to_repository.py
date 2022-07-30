import os
import shutil

from optimum.onnxruntime import ORTModelForCausalLM

if __name__ == "__main__":
    repository_path = "/var/triton/gpt2/repository/"
    model_id = "distilgpt2"
    model = ORTModelForCausalLM.from_pretrained(
        model_id, from_transformers=True
    )

    # Save tokenizer for use by triton
    tokenizer_path = repository_path + "encoder/"
    os.makedirs(tokenizer_path, exist_ok=True)
    shutil.copy("triton-config/encoder/config.pbtxt", tokenizer_path + "config.pbtxt")
    os.makedirs(tokenizer_path + "1/", exist_ok=True)
    shutil.copy("triton-config/encoder/model.py", tokenizer_path + "1/model.py")

    # Save model for use by triton
    model_path = repository_path + "model/"
    os.makedirs(model_path, exist_ok=True)
    shutil.copy("triton-config/model/config.pbtxt", model_path + "config.pbtxt")
    model.save_pretrained(model_path + "1/", file_name="model.onnx")

    # Save pipeline for use by triton
    transformer_path = repository_path + "transformer/"
    os.makedirs(transformer_path + "1/", exist_ok=True)
    shutil.copy("triton-config/model/config.pbtxt", transformer_path + "1/config.pbtxt")
