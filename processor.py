import os
from dataclasses import dataclass, field
from enum import IntEnum

from core_power import CStates, calculate_core_power, get_c_state_from_idle_governor, CState, APPROX_INFINITY_S
from core_residency import allocate_cores_argane_swing_dst
from instance import Instance
from simulator import clock

NUM_CORES = 128

CORE_IS_FREE = ''

#LOGICAL_CORE_COUNT = 256

class Core:
    """A core in the cpu"""
    id: int
    model: str = None
    c_state: CState = CStates.C0.value
    last_idle_durations = []
    last_idle_set_time: float = 0.0 # last time that the core is set to idle.

    # init for all params.
    def __init__(self, id: int):
        self.id = id
        self.model = None
        self.c_state = CStates.C1.value
        self.last_idle_durations = []
        self.last_idle_set_time = 0.0

    def __str__(self):
        return (
            f"Core(id={self.id}, "
            f"model={self.model}, "
            f"c_state={self.c_state}, "
            f"last_idle_durations={self.last_idle_durations}, "
            f"last_idle_set_time={self.last_idle_set_time})"
        )


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
    Modified to model an 2 * AMD EPYC CPU with 128 physical cores having hyper-threading disabled. In that way, each
    logical core maps to a physical core. The profiling data can then be faithfully used. Profiling data of core
    residency is measured from the operating system level, thus maps to occupancy of physical cores. Profiling data of
    per-core power measurements were conducted by executing one inference task at a time, thus maps to power consumption
    of a single logical core in our setup. Existing production AMD EPCY CPUs are able to provide this much of CPU cores.
    Examples: https://www.titancomputers.com/AMD-EPYC-Genoa-9004-Series-up-to-256-Cores-Workstation-PC-s/1270.htm

    Attributes:
        processor_type (ProcessorType): The type of the Processor.
        cpu_cores (list[bool]): The logical cores of the CPU. Boolean state indicates if the core is in use.
    """
    processor_type: ProcessorType = ProcessorType.CPU
    cpu_cores: list[Core] = field(default_factory=lambda: [Core(id=idx) for idx in range(NUM_CORES)])
    core_activity_log: []

    def assign_core_to_iteration(self, model):
        assigned_core, time_to_wake = self.get_a_core_to_assign()

        # set the core to serve
        assigned_core.model = model.name
        assigned_core.c_state = CStates.C0.value # set core to busy
        assigned_core.last_idle_durations.append(clock() - assigned_core.last_idle_set_time)
        assigned_core.last_idle_set_time = None

        # maintain a list of last 8 idle durations
        if len(assigned_core.last_idle_durations) == 9:
            assigned_core.last_idle_durations.pop(0)

        self.log_cpu_state()
        return assigned_core.id, time_to_wake

    def release_core_from_iteration(self, iteration_core_id):
        # free the core
        core = list(filter(lambda c: c.id == iteration_core_id, self.cpu_cores))[0]
        core.model = None
        core.c_state = CStates.C1.value # set core to idle

        # update core idle state
        next_c_state = get_c_state_from_idle_governor(last_8_idle_durations_s=core.last_idle_durations,
                                                      latency_limit_core_wake_s=APPROX_INFINITY_S)
        core.c_state = next_c_state
        core.last_idle_set_time = clock()

        self.log_cpu_state()

    def get_a_core_to_assign(self):
        cores_in_use = list(filter(lambda core: core.model is not None, self.cpu_cores))
        assigned_core_id = allocate_cores_argane_swing_dst(cores_in_use=cores_in_use,
                                                           max_retries=128,
                                                           num_of_logical_cores=len(self.cpu_cores))
        assigned_core = list(filter(lambda core: core.id == assigned_core_id, self.cpu_cores))[0]
        if assigned_core.model is not None:
            raise ValueError("Core already allocated")

        transition_latency = assigned_core.c_state.transition_time_s
        assigned_core.c_state = CStates.C1.value # set core to idle

        return assigned_core, transition_latency

    def log_cpu_state(self):
        for core in self.cpu_cores:
            self.core_activity_log.append({
                'clock': clock(),
                'id': core.id,
                'model': core.model,
                'c-state': core.c_state.state,
                'power': calculate_core_power(c_state=core.c_state, model=core.model)
            })


@dataclass(kw_only=True)
class GPU(Processor):
    processor_type: ProcessorType = ProcessorType.GPU


if __name__ == "__main__":
    pass
