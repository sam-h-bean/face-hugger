# face-hugger
![](./resources/hicks-aliens.gif)

This repository is meant to be a minimal example of converting a HuggingFace model to ONNX then 
hosting it on Triton deployed to Kubernetes.

## ONNX Export
This repository uses [Huggingface Optimum](https://github.com/huggingface/optimum) to convert a transformer model to the ONNX format.
In order to not use the same pod resources for both serving and exporting I have used a Helm Chart hook to save the graph to a persistent volume which is then 
used to load the model for inference in the serving pod.

## TensorRT Conversion (In Progress)
Since TensorRT gives better performance than even level 99 ONNX optimized graph on GPU we will try to
convert the ONNX graph to TensorRT and host that.

[transformer-deploy](https://github.com/ELS-RD/transformer-deploy) has done a lot of interesting work
writing the glue code to make TensorRT and Triton easier to use. We will try to use their utility
methods  to convert our ONNX model to TensorRT.

## Triton
This graph is then hosted for inference on a [Triton server](https://github.com/triton-inference-server/server) deployed in Kubernetes. The server is exposed through a Kubernetes LoadBalancer where outside
requests can communicate with the model in Triton.

I wanted to use Triton to see if it was a better MLOps solution for inference as well as learn more about TensorRT.
