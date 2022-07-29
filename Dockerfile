FROM nvcr.io/nvidia/tritonserver:22.06-py3
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["tritonserver", "--model-repository=/var/triton/gpt2/repository"]
