import math
import os
from dataclasses import dataclass, field
from enum import IntEnum

from core_power import CStates, calculate_core_power, get_c_state_from_idle_governor, CState, APPROX_INFINITY_S, \
    generate_initial_frequencies, calc_delta_vth, calc_aged_freq
from core_residency import allocate_cores_argane_swing_dst
from instance import Instance
from simulator import clock

# ServerlessLLM: With optimization, LLM serving is done with 4 cores. In their setup, 4 cores achieved the maximum
# bandwidth utilization.

CORE_IS_FREE = ''
ENABLE_DEBUG_LOGS = False


# LOGICAL_CORE_COUNT = 256

class Core:
    """A core in the cpu"""
    id: int
    task: str = None
    c_state: CState = CStates.C1.value
    last_idle_durations = []
    last_idle_set_time: float = 0.0  # last time that the core is set to idle.
    temp: float  # temperature of the core in degrees of celcius
    freq_nominal: float  # expected frequency without aging in Hz
    freq_0: float  # initial frequency of the core before aging in Hz
    freq: float  # frequency of the core in Hz
    vth_delta: float
    last_state_change_time: float = 0.0  # last time that the core state is changed.

    # init for all params.
    def __init__(self, id: int, f_init: float = 2.25 * math.pow(10, 9), temp_init=54):
        self.id = id
        self.task = None
        self.c_state = CStates.C1.value
        self.last_idle_durations = []
        self.last_idle_set_time = 0.0
        self.temp = temp_init
        self.freq = f_init
        self.freq_0 = f_init
        self.last_state_change_time = 0.0
        self.vth_delta = 0.0
        self.freq_nominal = 2.25 * math.pow(10, 9) # AMD EPCY 7742 base frequency is 2.25 GHz

    def __str__(self):
        return (
            f"Core(id={self.id}, "
            f"task={self.task}, "
            f"c_state={self.c_state}, "
            f"last_idle_durations={self.last_idle_durations}, "
            f"last_idle_set_time={self.last_idle_set_time})"
        )

    def get_record(self):
        return {
            'id': self.id,
            'task': self.task,
            'c_state': self.c_state.state,
            'last_idle_durations': self.last_idle_durations,
            'last_idle_set_time': self.last_idle_set_time,
            'temp_c': self.temp,
            'freq_hz': self.freq,
        }


class ProcessorType(IntEnum):
    DEFAULT = 0
    CPU = 1
    GPU = 2


class Temperatures(IntEnum):
    C0_RTEVAL = 54
    C0_POLL = 51.08
    C6 = 48


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
            # raise ValueError("OOM")
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

    todo: Baseline technique. Implement the support for DVFS.
        - From Nautilus'19: 13 DVFS steps. each step drops temp from peak 77 to idle 48. thats 29 degrees drop. For
        that they measure 2 degree drop per step. Ours task execution temp is 54@C0 (green cores testbed). Sleep is 48.
        thats 6 degrees drop. For 13 steps, it counts to 0.414 degrees per step. When we implement DVFS technique, we
        need to model this temperature drop per step.
        Our proposed: Wake is 54 degrees and sleep is 48 degrees. When ours sleep, core clock is stopped. No switching
        activity. So, there is no NBTI aging.

    todo:   1.  [implemented] Implement the core temperature changes based on C0/C1 and C6 states (54 to 48): done, needs testing
            2.  [implemented] Implement the over time aging: temperature induced frequency drop: done, needs testing
            3.  [implemented] Implement the frequency degreation impact to LLM inference. Each executing task now takes more time,
                so increased overhead time. : done, needs testing
            4.  [Pending] Run the vanilla system and plot the frequency degregation over time (aging), and impact to app metrics: TTFT, TBT
            5.  [Pending] Implement the DVFS technique (determine healthy non-healthy cores by deviding current frequency with initial freqeuncy.
            GPU tasks cpu-intensive since important with highest step, others not so with lowest step.). Verify/fix plots.
            6.  [Pending] Implement the proposed technique (Sleep/Wake sensitive to request rate. Dynamic core-set selection based
            on relative frequency(relative health)). Verify/fix plots.

    Attributes:
        processor_type (ProcessorType): The type of the Processor.
        cpu_cores (list[bool]): The logical cores of the CPU. Boolean state indicates if the core is in use.
    """
    processor_type: ProcessorType = ProcessorType.CPU
    core_count: int
    # cpu_cores: list[Core] = field(default_factory=lambda: [Core(id=idx) for idx in range(core_count)])
    cpu_cores: list[Core] = field(default_factory=lambda: [Core(id=idx) for idx in range(1)])
    core_activity_log: []

    def __post_init__(self):
        # Now we can use core_count to initialize cpu_cores
        process_variation_induced_initial_core_fq = generate_initial_frequencies(n_cores=self.core_count)

        # initial temperature is 54 degrees celcius. Data modelled after experiments from Green Core testbed.
        self.cpu_cores = [Core(id=idx, f_init=init_fq, temp_init=Temperatures.C0_POLL) for idx, init_fq in
                          process_variation_induced_initial_core_fq]

    def assign_core_to_cpu_task(self, task, override_task_description=None):
        assigned_core, time_to_wake = self.get_a_core_to_assign()

        self.update_aging(assigned_core)

        # set the core to serve
        # print(task.value)
        assigned_core.task = task.value["info"]
        if override_task_description is not None:
            assigned_core.task = override_task_description
        assigned_core.c_state = CStates.C0.value  # set core to busy
        assigned_core.temp = Temperatures.C0_RTEVAL # model temperature for executing task
        assigned_core.last_idle_durations.append(clock() - assigned_core.last_idle_set_time)
        assigned_core.last_idle_set_time = None

        # maintain a list of last 8 idle durations
        if len(assigned_core.last_idle_durations) == 9:
            assigned_core.last_idle_durations.pop(0)

        age_induced_freq_scaling_factor = assigned_core.freq / assigned_core.freq_nominal

        # debug
        if ENABLE_DEBUG_LOGS:
            print(f"Assign core: {assigned_core.id} to task: {assigned_core.task} at time: {clock()}")

        self.log_cpu_state(core=assigned_core, time_to_wake=time_to_wake)
        return assigned_core.id, time_to_wake, age_induced_freq_scaling_factor

    def release_core_from_cpu_task(self, task_core_id):

        # free the core
        core = list(filter(lambda c: c.id == task_core_id, self.cpu_cores))[0]

        self.update_aging(core)

        # debug
        if ENABLE_DEBUG_LOGS:
            print(f"Release core: {core.id} from task: {core.task} at time: {clock()}")

        core.task = None
        core.c_state = CStates.C1.value  # set core to idle

        # update core idle state
        next_c_state = get_c_state_from_idle_governor(last_8_idle_durations_s=core.last_idle_durations,
                                                      latency_limit_core_wake_s=APPROX_INFINITY_S)
        core.c_state = next_c_state
        core.temp = next_c_state.temp # model temperature for idle core
        core.last_idle_set_time = clock()

        self.log_cpu_state(core=core, time_to_wake=None)

    def update_aging(self, assigned_core):
        clk = clock()
        time_delta_since_last_state_change = clk - assigned_core.last_state_change_time
        assigned_core.last_state_change_time = clk
        vth_delta = calc_delta_vth(t_elapsed_time=time_delta_since_last_state_change, temp_kelvin=assigned_core.temp)
        assigned_core.vth_delta += vth_delta
        aged_freq = calc_aged_freq(initial_freq=assigned_core.freq_0, delta_vth=assigned_core.vth_delta)
        assigned_core.freq = aged_freq

    def get_a_core_to_assign(self):
        assigned_core = allocate_cores_argane_swing_dst(available_cores=self.cpu_cores, max_retries=self.core_count)
        transition_latency = assigned_core.c_state.transition_time_s
        assigned_core.c_state = CStates.C1.value  # set core to idle
        return assigned_core, transition_latency

    def log_cpu_state(self, core=None, time_to_wake=None):
        if core is None:
            for core in self.cpu_cores:
                self.core_activity_log.append({
                    'clock': clock(),
                    'id': core.id,
                    'task': core.task,
                    'c-state': core.c_state.state,
                    'power': calculate_core_power(c_state=core.c_state, model=core.task),
                    'temp': core.temp,
                    'freq': core.freq,
                })

        core_state = core.get_record()
        core_state['clock'] = clock()
        core_state['c_state_wake_latency'] = time_to_wake
        self.core_activity_log.append(core_state)


@dataclass(kw_only=True)
class GPU(Processor):
    processor_type: ProcessorType = ProcessorType.GPU


if __name__ == "__main__":
    pass
