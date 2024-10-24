import logging
import os

from dataclasses import dataclass, field
from enum import IntEnum

from core_residency import gen_core_id
from core_residency import sampler
from instance import Instance
from simulator import clock, schedule_event, cancel_event, reschedule_event

LOGICAL_CORE_COUNT = 256


class ProcessorType(IntEnum):
    DEFAULT = 0
    CPU = 1
    GPU = 2


@dataclass(kw_only=True)
class Processor():
    """
    Processor is the lowest-level processing unit that can run computations (Tasks).
    Multiple Processors constitute a Server and may be linked via Interconnects.
    For example, CPU and GPU are both different types of Processors.

    Each Processor can belong to only one Server
    Processor could eventually run multiple Instances/Tasks.

    Attributes:
        processor_type (ProcessorType): The type of the Processor.
        memory_size (float): The memory size of the Processor.
        memory_used (float): The memory used by the Processor.
        server (Server): The Server that the Processor belongs to.
        instances (list[Instance]): Instances running on this Processor.
        interconnects (list[Link]): Peers that this Processor is directly connected to.
    """
    processor_type: ProcessorType
    name: str
    server: 'Server'
    memory_size: int
    memory_used: int
    _memory_used: int = 0
    power: float = 0.
    _power: float = 0.
    instances: list[Instance] = field(default_factory=list)
    interconnects: list['Link'] = field(default_factory=list)

    @property
    def server(self):
        return self._server

    @server.setter
    def server(self, server):
        if type(server) is property:
            server = None
        self._server = server

    @property
    def memory_used(self):
        return self._memory_used

    @memory_used.setter
    def memory_used(self, memory_used):
        if type(memory_used) is property:
            memory_used = 0
        if memory_used < 0:
            raise ValueError("Memory cannot be negative")
        # if OOM, log instance details
        if memory_used > self.memory_size:
            if os.path.exists("oom.csv") is False:
                with open("oom.csv", "w", encoding="UTF-8") as f:
                    fields = ["time",
                              "instance_name",
                              "instance_id",
                              "memory_used",
                              "processor_memory",
                              "pending_queue_length"]
                    f.write(",".join(fields) + "\n")
            with open("oom.csv", "a", encoding="UTF-8") as f:
                instance = self.instances[0]
                csv_entry = []
                csv_entry.append(clock())
                csv_entry.append(instance.name)
                csv_entry.append(instance.instance_id)
                csv_entry.append(memory_used)
                csv_entry.append(self.memory_size)
                csv_entry.append(len(instance.pending_queue))
                f.write(",".join(map(str, csv_entry)) + "\n")
            # raise OOM error
            #raise ValueError("OOM")
        self._memory_used = memory_used

    @property
    def memory_free(self):
        return self.memory_size - self.memory_used

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, power):
        if type(power) is property:
            power = 0.
        if power < 0:
            raise ValueError("Power cannot be negative")
        self._power = power

    @property
    def peers(self):
        pass


@dataclass(kw_only=True)
class CPU(Processor):
    """
    Modified to model an AMD EPYC 7742 CPU with 128 physical cores.

    Attributes:
        processor_type (ProcessorType): The type of the Processor.
        cpu_cores (list[bool]): The logical cores of the CPU. Boolean state indicates if the core is in use.
    """
    processor_type: ProcessorType = ProcessorType.CPU
    cpu_cores: [False] * LOGICAL_CORE_COUNT
    core_activity_log: []

    def assign_core_to_iteration(self):
        """Assume that each iteration requires a single logical core."""
        allocated_cores = {index for index, core in enumerate(self.cpu_cores) if core is True}
        core_id = gen_core_id(sampler=sampler, allocated_cores=allocated_cores, max_retries=10, range=LOGICAL_CORE_COUNT)
        self.cpu_cores[core_id] = True
        self.log_core_activity()
        return core_id

    def release_core_from_iteration(self, iteration_core_id):
        self.cpu_cores[iteration_core_id] = False
        self.log_core_activity()

    def log_core_activity(self):
        # Log core activity
        log_entry = {"clock": clock()}
        for i in range(LOGICAL_CORE_COUNT):
            log_entry[f"core_{i}"] = self.cpu_cores[i]
        # log_entry = {"clock": clock(), "core_id": core_id, "is_allocated": True, "is_released": False}
        self.core_activity_log.append(log_entry)


@dataclass(kw_only=True)
class GPU(Processor):
    processor_type: ProcessorType = ProcessorType.GPU


if __name__ == "__main__":
    pass
