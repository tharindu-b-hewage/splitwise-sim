"""
Utility functions to initialize the Cluster with a starting state.
"""

import logging

from joblib.externals.cloudpickle import instance
from pandas.core.computation.expressions import where

from model import ModelParallelism
from simulator import clock, schedule_event, cancel_event, reschedule_event


def load_start_state(start_state_cfg, **kwargs):
    """
    Load the start state configuration and initialize the cluster.
    """
    state_type = start_state_cfg.state_type
    if state_type == "unallocated":
        pass
    elif state_type == "orca":
        uniform(start_state_cfg, **kwargs)
    elif state_type == "baseline":
        uniform(start_state_cfg, **kwargs)
    elif "splitwise" in state_type:
        splitwise(start_state_cfg, **kwargs)
    else:
        raise ValueError(f"Unknown start state type: {state_type}")


def uniform(start_state_cfg, cluster, applications, **kwargs):
    """
    Initialize all servers with a single instance of the application.
    """
    application = applications[start_state_cfg.application_id]
    allocator = application.allocator
    servers = cluster.servers

    instance_cfg = start_state_cfg.instance
    parallelism = ModelParallelism(pipeline_parallelism=instance_cfg.pipeline_parallelism,
                                   tensor_parallelism=instance_cfg.tensor_parallelism)

    for sku_name in servers:
        for server in servers[sku_name]:
            allocator.start_spin_up_instance(instance_cfg=instance_cfg,
                                             processors=server.processors,
                                             parallelism=parallelism,
                                             pre_start=True)


def splitwise(start_state_cfg, cluster, applications, **kwargs):
    """
    Initialize all servers with a single instance of the application.
    Separate prompt and token instances with different kinds of parallelism.
    TODO: use preferences and constraints within scheduler instead
    """
    application = applications[start_state_cfg.application_id]
    allocator = application.allocator
    servers = cluster.servers

    prompt_cfg = start_state_cfg.prompt
    token_cfg = start_state_cfg.token
    prompt_parallelism = ModelParallelism(pipeline_parallelism=prompt_cfg.pipeline_parallelism,
                                          tensor_parallelism=prompt_cfg.tensor_parallelism)
    token_parallelism = ModelParallelism(pipeline_parallelism=token_cfg.pipeline_parallelism,
                                         tensor_parallelism=token_cfg.tensor_parallelism)

    split_type = start_state_cfg.split_type

    if split_type == "homogeneous":
        n_prompts = prompt_cfg.num_instances
        n_tokens = token_cfg.num_instances
        # allocate n_prompt instance of prompt
        all_servers = [server for sku_name in servers for server in servers[sku_name]]
        for server in all_servers[:n_prompts]:
            gpus = [processor for processor in server.processors if processor.processor_type.value == 2]
            cpus = [processor for processor in server.processors if processor.processor_type.value == 1]
            for proc_id in range(0, len(gpus), prompt_parallelism.tensor_parallelism):
                # allocate instance processors
                instance_processors = get_instance_processors(proc_id, proc_id+prompt_parallelism.tensor_parallelism,
                                                              cpus, gpus)
                allocator.start_spin_up_instance(instance_cfg=prompt_cfg,
                                                 processors=instance_processors,
                                                 parallelism=prompt_parallelism,
                                                 pre_start=True,
                                                 tag="prompt")
        for server in all_servers[n_prompts:n_prompts+n_tokens]:
            gpus = [processor for processor in server.processors if processor.processor_type.value == 2]
            cpus = [processor for processor in server.processors if processor.processor_type.value == 1]
            for proc_id in range(0, len(gpus), token_parallelism.tensor_parallelism):
                instance_processors = get_instance_processors(proc_id, proc_id+token_parallelism.tensor_parallelism,
                                                              cpus, gpus)
                allocator.start_spin_up_instance(instance_cfg=token_cfg,
                                                 processors=instance_processors,
                                                 parallelism=token_parallelism,
                                                 pre_start=True,
                                                 tag="token")

    if split_type == "heterogeneous":
        prompt_instances = prompt_cfg.instance_names
        token_instances = token_cfg.instance_names
        for sku_name in servers:
            for server in servers[sku_name]:
                gpus = [processor for processor in server.processors if processor.processor_type.value == 2]
                cpus = [processor for processor in server.processors if processor.processor_type.value == 1]
                if sku_name in prompt_instances:
                    # allocate as many prompt instances as possible
                    for proc_id in range(0, len(gpus), prompt_parallelism.tensor_parallelism):
                        instance_processors = get_instance_processors(proc_id,
                                                                      proc_id + prompt_parallelism.tensor_parallelism,
                                                                      cpus, gpus)
                        allocator.start_spin_up_instance(instance_cfg=prompt_cfg,
                                                         processors=instance_processors,
                                                         parallelism=prompt_parallelism,
                                                         pre_start=True,
                                                         tag="prompt")
                elif sku_name in token_instances:
                    # allocate as many token instances as possible
                    for proc_id in range(0, len(gpus), token_parallelism.tensor_parallelism):
                        instance_processors = get_instance_processors(proc_id,
                                                                      proc_id + token_parallelism.tensor_parallelism,
                                                                      cpus, gpus)
                        allocator.start_spin_up_instance(instance_cfg=token_cfg,
                                                         processors=instance_processors,
                                                         parallelism=token_parallelism,
                                                         pre_start=True,
                                                         tag="token")
                else:
                    raise ValueError(f"Unsupported sku_name: {sku_name}")


def get_instance_processors(gpu_start_idx, gpu_end_idx, cpus, gpus):
    instance_processors = cpus + gpus[gpu_start_idx:gpu_start_idx + gpu_end_idx]
    return instance_processors
