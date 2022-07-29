import os
import shutil

from optimum.onnxruntime import ORTModelForCausalLM

if __name__ == "__main__":
    repository_path = "/var/triton/gpt2/repository/"
    # model_id = "distilgpt2"
    # model = ORTModelForCausalLM.from_pretrained(
    #     model_id, from_transformers=True
    # )
    # Save tokenizer for use by triton
    tokenizer_path = repository_path + "encoder/"
    os.makedirs(tokenizer_path, exist_ok=True)
    shutil.copy("config.pbtx", tokenizer_path)
    shutil.copy("python_tokenizer.py", tokenizer_path)
    #
    # # Save model for use by triton
    # model_path = repository_path + "model/"
    # os.makedirs(model_path, exist_ok=True)
    # shutil.copy("model.pbtx", model_path)
    # model.save_pretrained(model_path, file_name="model.onnx")
    #
    # # Save pipeline for use by triton
    # transformer_path = repository_path + "transformer/"
    # os.makedirs(transformer_path, exist_ok=True)
    # shutil.copy("transformer.pbtx", transformer_path)
