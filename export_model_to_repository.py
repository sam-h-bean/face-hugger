import os
import shutil

from optimum.onnxruntime import ORTModelForSequenceClassification


if __name__ == "__main__":
    repository_path = "/var/triton/gpt2/"
    os.mkdir(repository_path)
    model_id = "distilgpt2"
    model = ORTModelForSequenceClassification.from_pretrained(
        model_id, from_transformers=True
    )
    # Save tokenizer for use by triton
    tokenizer_path = repository_path + "encoder"
    os.mkdir(tokenizer_path)
    shutil.copy("encoder.pbtx", tokenizer_path)
    shutil.copy("python_tokenizer.py", tokenizer_path)

    # Save model for use by triton
    model_path = repository_path + "model"
    os.mkdir(model_path)
    shutil.copy("model.pbtx", model_path)
    model.save_pretrained(model_path, file_name="mode.onnx")

    # Save pipeline for use by triton
    transformer_path = repository_path + "transformer"
    os.mkdir(transformer_path)
    shutil.copy("transformer.pbtx", transformer_path)
