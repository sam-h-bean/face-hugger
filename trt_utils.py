from typing import Callable, Dict, List, OrderedDict, Tuple

import tensorrt as trt
import torch
from tensorrt import ICudaEngine, IExecutionContext
from tensorrt.tensorrt import (
    Builder,
    IBuilderConfig,
    IElementWiseLayer,
    ILayer,
    INetworkDefinition,
    IOptimizationProfile,
    IReduceLayer,
    Logger,
    OnnxParser,
    Runtime,
)


def fix_fp16_network(network_definition: INetworkDefinition) -> INetworkDefinition:
    """
    Mixed precision on TensorRT can generate scores very far from Pytorch because of some operator being saturated.
    Indeed, FP16 can't store very large and very small numbers like FP32.
    Here, we search for some patterns of operators to keep in FP32, in most cases, it is enough to fix the inference
    and don't hurt performances.
    :param network_definition: graph generated by TensorRT after parsing ONNX file (during the model building)
    :return: patched network definition
    """
    # search for patterns which may overflow in FP16 precision, we force FP32 precisions for those nodes
    for layer_index in range(network_definition.num_layers - 1):
        layer: ILayer = network_definition.get_layer(layer_index)
        next_layer: ILayer = network_definition.get_layer(layer_index + 1)
        # POW operation usually followed by mean reduce
        if layer.type == trt.LayerType.ELEMENTWISE and next_layer.type == trt.LayerType.REDUCE:
            # casting to get access to op attribute
            layer.__class__ = IElementWiseLayer
            next_layer.__class__ = IReduceLayer
            if layer.op == trt.ElementWiseOperation.POW:
                layer.precision = trt.DataType.FLOAT
                next_layer.precision = trt.DataType.FLOAT
            layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
            next_layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
    return network_definition


def build_engine(
    runtime: Runtime,
    onnx_file_path: str,
    logger: Logger,
    min_shape: Tuple[int, int],
    optimal_shape: Tuple[int, int],
    max_shape: Tuple[int, int],
    workspace_size: int,
    fp16: bool,
    int8: bool,
) -> ICudaEngine:
    """
    Convert ONNX file to TensorRT engine.
    It supports dynamic shape, however it's advised to keep sequence length fix as it hurts performance otherwise.
    Dynamic batch size don't hurt performance and is highly advised.
    :param runtime: global variable shared accross inference call / model building
    :param onnx_file_path: path to the ONNX file
    :param logger: specific logger to TensorRT
    :param min_shape: the minimal shape of input tensors. It's advised to set first dimension (batch size) to 1
    :param optimal_shape: input tensor shape used for optimizations
    :param max_shape: maximal input tensor shape
    :param workspace_size: GPU memory to use during the building, more is always better. If there is not enough memory,
    some optimization may fail, and the whole conversion process will crash.
    :param fp16: enable FP16 precision, it usually provide a 20-30% boost compared to ONNX Runtime.
    :param int8: enable INT-8 quantization, best performance but model should have been quantized.
    :return: TensorRT engine to use during inference
    """
    with trt.Builder(logger) as builder:  # type: Builder
        with builder.create_network(
            flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network_definition:  # type: INetworkDefinition
            with trt.OnnxParser(network_definition, logger) as parser:  # type: OnnxParser
                builder.max_batch_size = max_shape[0]  # max batch size
                config: IBuilderConfig = builder.create_builder_config()
                config.max_workspace_size = workspace_size
                # to enable complete trt inspector debugging, only for TensorRT >= 8.2
                # config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
                # disable CUDNN optimizations
                config.set_tactic_sources(
                    tactic_sources=1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT)
                )
                if int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                if fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
                # https://github.com/NVIDIA/TensorRT/issues/1196 (sometimes big diff in output when using FP16)
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                with open(onnx_file_path, "rb") as f:
                    parser.parse(f.read())
                profile: IOptimizationProfile = builder.create_optimization_profile()
                for num_input in range(network_definition.num_inputs):
                    profile.set_shape(
                        input=network_definition.get_input(num_input).name,
                        min=min_shape,
                        opt=optimal_shape,
                        max=max_shape,
                    )
                config.add_optimization_profile(profile)
                if fp16:
                    network_definition = fix_fp16_network(network_definition)
                trt_engine = builder.build_serialized_network(network_definition, config)
                engine: ICudaEngine = runtime.deserialize_cuda_engine(trt_engine)
                assert engine is not None, "error during engine generation, check error messages above :-("
                return engine


def save_engine(engine: ICudaEngine, engine_file_path: str) -> None:
    """
    Serialize TensorRT engine to file.
    :param engine: TensorRT engine
    :param engine_file_path: output path
    """
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())