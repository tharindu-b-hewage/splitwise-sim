import math
import os
import threading
import uuid
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from core_power import CStates, calculate_core_power, get_c_state_from_idle_governor, CState, APPROX_INFINITY_S, \
    generate_initial_frequencies, calc_delta_vth, calc_aged_freq, Temperatures, CPU_CORE_ADJ_INTERVAL
from core_residency import task_schedule_linux
from instance import Instance, CpuTaskType
from simulator import clock

# ServerlessLLM: With optimization, LLM serving is done with 4 cores. In their setup, 4 cores achieved the maximum
# bandwidth utilization.

CORE_IS_FREE = ''
ENABLE_DEBUG_LOGS = False

CPU_CONFIGS = None


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
    forced_to_sleep: bool = False

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
        self.forced_to_sleep = False

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
            'freq': self.freq,
            'health': self.freq / self.freq_0,
        }


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


def task_schedule_zhao23(cpu_cores, max_retries):
    """Zhao et al. 2023:
    Proposes a task scheduling approach that can be enforced at the resource management level.
    We employ it at the cloud orchestration level. The idea behind the approach is, 'the OS can migrate and
     swap affinitized threads from one core that has significantly aged or thermally loaded to another core'. It requires
     aging of the core and move the tasks, which essentially signals to balance the load across the cores according
     to the age status. Since inference tasks are short-lived, we do not migrate tasks, but at the task-to-core allocation
     level, we select the cores to balance the load according to the health.
     """
    selected_core = None
    for core in cpu_cores:
        if core.task is not None: # avoid already allocated cores.
            continue

        if selected_core is None: # if a core has not chosen yet, select the first core.
            selected_core = core
            continue

        # select the core with the highest health.
        selected_core_health = selected_core.freq / selected_core.freq_0
        core_health = core.freq / core.freq_0
        if core_health > selected_core_health:
            selected_core = core

    # if none, pick the first free core.
    if selected_core is None:
        for core in cpu_cores:
            if core.task is None:
                selected_core = core
                break

    return selected_core


def task_schedule_proposed(cpu_cores, max_retries):
    selected_core = None

    # todo: implement the logic for our algorithm.

    # if none, pick the first free core.
    if selected_core is None:
        for core in cpu_cores:
            if core.task is None:
                selected_core = core
                break

    return selected_core


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
            4.  [implemented] Run the vanilla system and plot the frequency degregation over time (aging), and impact to app metrics: TTFT, TBT
            5.  [implemented] Implement the Zhaos23 technique. Verify/fix plots.
            6.  [Pending] Implement the proposed technique (Sleep/Wake sensitive to request rate. Dynamic core-set selection based
            on relative frequency(relative health)). Verify/fix plots.

    todo: Identified issues:

    Attributes:
        processor_type (ProcessorType): The type of the Processor.
        cpu_cores (list[bool]): The logical cores of the CPU. Boolean state indicates if the core is in use.
    """
    processor_type: ProcessorType = ProcessorType.CPU
    core_count: int
    # cpu_cores: list[Core] = field(default_factory=lambda: [Core(id=idx) for idx in range(core_count)])
    #cpu_cores: list[Core] = field(default_factory=lambda: [Core(id=idx) for idx in range(1)])
    #core_activity_log: []
    #oversubscribed_task_count_log: int = 0
    #total_task_count_log: int = 0


    def __post_init__(self):

        # Now we can use core_count to initialize cpu_cores
        process_variation_induced_initial_core_fq = generate_initial_frequencies(n_cores=self.core_count)

        # initial temperature is 54 degrees celcius. Data modelled after experiments from Green Core testbed.
        self.cpu_cores = [Core(id=idx, f_init=init_fq, temp_init=Temperatures.C0_POLL) for idx, init_fq in
                          enumerate(process_variation_induced_initial_core_fq)]
        self.core_activity_log = []
        self.oversubscribed_task_count_log = 0
        self.total_task_count_log = 0

        self.temp_T_ts = []
        self.temp_running_tasks = []
        self.temp_running_tasks_counter = 0

        # manage core sleeping.
        self.active_core_limit = len(self.cpu_cores) # Initial parameter. controller should adjust to optmize.
        self.put_to_sleep(to_sleep=len(self.cpu_cores) - self.active_core_limit, cores=self.cpu_cores) # initial core-set to sleep. Done taking the health into account.
        self.core_oversubscribe_tasks = {
            "past_e_t": [],
            "core_oversubscribe_tasks": 0,
            "last_core_adjust_time": 0.0,
            "core_adjust_dt": CPU_CORE_ADJ_INTERVAL,
            "err_integral": 0.0,
            "prev_error": 0.0,
        }

        self.sleep_manager_logs = []
        self.id = uuid.uuid4()

    # todo: check if task to core balance it + or -. then trigger periodic event to adjust the core count.
    def assign_core_to_cpu_task(self, task, override_task_description=None):
        self.temp_running_tasks_counter += 1

        mem_used = 0.0
        mem_total = 0.0
        for p in self.server.processors:
            if p == self:
                continue
            mem_used += p.memory_used
            mem_total += p.memory_size
        mem_util = mem_used / mem_total

        self.temp_running_tasks.append([clock(), self.temp_running_tasks_counter, mem_util])

        self.total_task_count_log += 1

        assigned_core, time_to_wake = self.get_a_core_to_assign()
        if assigned_core is None:
            self.oversubscribed_task_count_log += 1
            self.core_oversubscribe_tasks['core_oversubscribe_tasks'] = self.core_oversubscribe_tasks['core_oversubscribe_tasks'] + 1
            return -1, 0.0, 1 # there was no free core to assign. Task is assumed to be oversubscribing cpu.

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
        self.temp_running_tasks_counter -= 1

        if task_core_id == -1:
            self.core_oversubscribe_tasks['core_oversubscribe_tasks'] = self.core_oversubscribe_tasks['core_oversubscribe_tasks'] - 1
            return

        # free the core
        core = list(filter(lambda c: c.id == task_core_id, self.cpu_cores))[0]
        self.free(core)

    def free(self, core):
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
        core.temp = next_c_state.temp  # model temperature for idle core
        core.last_idle_set_time = clock()
        self.log_cpu_state(core=core, time_to_wake=None)

    def adjust_sleeping_cores(self):

        oversub_tasks = self.core_oversubscribe_tasks["core_oversubscribe_tasks"]
        normal_tasks = len(list(filter(lambda c: c.task is not None and not c.forced_to_sleep, self.cpu_cores)))
        T_t = normal_tasks + oversub_tasks
        self.temp_T_ts.append(T_t)

        active_cores = len(list(filter(lambda c: not c.forced_to_sleep, self.cpu_cores)))
        N = len(self.cpu_cores)
        C_SLP_t = N - active_cores

        e_t = N - C_SLP_t - T_t
        e_t = min(N, e_t)
        past_e_t = self.core_oversubscribe_tasks["past_e_t"]
        if len(past_e_t) == 8:
            past_e_t.pop(0)
        past_e_t.append(e_t)

        k = 90
        e_t_pkth = np.percentile(past_e_t, k)
        e_t_pkth_nrml = e_t_pkth / N

        F_e_t_pkth_nrml = e_t_pkth_nrml
        if e_t_pkth_nrml >= 0:
            F_e_t_pkth_nrml = math.tan(0.785 * e_t_pkth_nrml)
        else:
            F_e_t_pkth_nrml = math.atan(1.55 * e_t_pkth_nrml)

        corrected_e_t = N * F_e_t_pkth_nrml

        if corrected_e_t > 0:
            self.put_to_sleep(to_sleep=int(corrected_e_t), cores=self.cpu_cores)
        elif corrected_e_t < 0:
            self.put_to_wake(to_wake=-int(corrected_e_t), cores=self.cpu_cores)

        # # stabilize to core over-subscription
        # dt = self.core_oversubscribe_tasks['core_adjust_dt']
        # t_delta = clock() - self.core_oversubscribe_tasks['last_core_adjust_time']
        # if t_delta < dt:
        #     return
        #
        #
        # self.core_oversubscribe_tasks['last_core_adjust_time'] = clock()
        #
        # """
        # We use an async approach to manage the number of awaken cores. When available number of cores are not
        # enough, some of the sleeping cores need to be awaken. What we do is, let the task to oversubscribe, so that any
        # delays incurring from deep-sleep wake is avoided, compromising the potential cpu time steal latency from the
        # oversubscription. We count the number of such tasks. Then we apply a PID controller to adjust the number of
        # cpu cores that are awaken.
        # """
        # k_p = 0.2
        # k_i = 0.1
        # k_d = 0.1
        # scaling_factor = 1.0
        #
        # oversub_tasks = self.core_oversubscribe_tasks["core_oversubscribe_tasks"]
        # normal_tasks = len(list(filter(lambda c: c.task is not None and not c.forced_to_sleep, self.cpu_cores)))
        # active_cores = len(list(filter(lambda c: not c.forced_to_sleep, self.cpu_cores)))
        #
        # total_tasks = oversub_tasks + normal_tasks
        # workload_ratio = total_tasks / active_cores
        #
        # # we want workload ratio to be 1.0. If more cores are active than tasks, the ratio is less than zero. If less
        # # cores are active than tasks, the ratio is more than 1.0. magnitude of the latter case is higher than former.
        # # this is what we need.
        # error = 1.0 - workload_ratio
        #
        # prop_term = k_p * error
        # self.core_oversubscribe_tasks['err_integral'] += error * dt
        # int_term = k_i * self.core_oversubscribe_tasks['err_integral']
        # der_term = k_d * ((error - self.core_oversubscribe_tasks['prev_error']) / dt)
        # self.core_oversubscribe_tasks['prev_error'] = error
        #
        # adjust = scaling_factor * (prop_term + int_term + der_term)
        #
        # if adjust > 0:
        #     self.put_to_wake(to_wake=int(adjust), cores=self.cpu_cores)
        # elif adjust < 0:
        #     self.put_to_sleep(to_sleep=-int(adjust), cores=self.cpu_cores)

        self.sleep_manager_logs.append({
            'clock': clock(),
            'oversub_tasks': oversub_tasks,
            'normal_tasks': normal_tasks,
            'T_t': T_t,
            'active_cores': active_cores,
            'asleep_cores': len(self.cpu_cores) - active_cores,
        })

    def update_aging(self, assigned_core):
        clk = clock()
        time_delta_since_last_state_change = clk - assigned_core.last_state_change_time
        assigned_core.last_state_change_time = clk
        vth_delta = calc_delta_vth(t_elapsed_time=time_delta_since_last_state_change, temp_kelvin=assigned_core.temp)
        assigned_core.vth_delta += vth_delta
        aged_freq = calc_aged_freq(initial_freq=assigned_core.freq_0, delta_vth=assigned_core.vth_delta)
        assigned_core.freq = aged_freq

    def get_a_core_to_assign(self):

        """Algorithm assigning a core to a task"""
        import configparser

        def load_properties(file_path):
            config = configparser.ConfigParser()
            # ConfigParser requires section headers, so we add a fake one
            absolute_directory = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(absolute_directory, file_path), 'r') as file:
                properties_data = f"[DEFAULT]\n{file.read()}"
            config.read_string(properties_data)
            return config['DEFAULT']

        global CPU_CONFIGS
        if CPU_CONFIGS is None:
            CPU_CONFIGS = load_properties('cpu_configs.properties')
            #print("cpu configs loaded", CPU_CONFIGS)

        # omit allocating sleeping cores.
        awaken_cores = list(filter(lambda c: not c.forced_to_sleep, self.cpu_cores))

        task_allocation_algo = CPU_CONFIGS.get("task_allocation_algo")
        if task_allocation_algo == 'linux':
            assigned_core = task_schedule_linux(cpu_cores=awaken_cores, max_retries=len(awaken_cores))
        elif task_allocation_algo == 'zhao23':
            assigned_core = task_schedule_zhao23(cpu_cores=awaken_cores, max_retries=len(awaken_cores))
        elif task_allocation_algo == 'proposed':
            assigned_core = task_schedule_proposed(cpu_cores=awaken_cores, max_retries=len(awaken_cores))
        else:
            raise ValueError(f"Unknown task allocation algorithm: {task_allocation_algo}")

        if assigned_core is None:
            return None, None

        transition_latency = assigned_core.c_state.transition_time_s
        assigned_core.c_state = CStates.C1.value  # set core to idle
        return assigned_core, transition_latency

    def trigger_state_update(self):
        """update states at the end of the simulation"""
        '''
        Releasing a task from the core updates multiple stats, such as aging. In the simulation, core status is 
        only updated either a task is assigned or released. However, in cases such as a task was never assigned,
        or the time between task release and end of the simulation, we need to still update stats, such as the 
        aging occur due to idle temperature. So we assign a completion task and release the core.
        '''
        for _ in self.cpu_cores:
            self.assign_core_to_cpu_task(task=CpuTaskType.SIM_STATUS_UPDATE_TASK) # assign to next free core.
        for core in self.cpu_cores:
            self.release_core_from_cpu_task(core.id) # release from each core.


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

    def put_to_sleep(self, to_sleep, cores):
        """
        Amongst awaken cores, filter free cores. Then sleep the amount of cores having the lowest health.
        """
        free_awake_cores = list(filter(lambda c: c.task is None and not c.forced_to_sleep, cores))

        # health is calculated as the current degraded frequency over the nominal frequency.
        free_awake_cores = sorted(free_awake_cores, key=lambda c: c.freq / c.freq_0) # health low to high.
        for core in free_awake_cores[:min(to_sleep, len(free_awake_cores))]:
            self.force_sleep(core)

    def put_to_wake(self, to_wake, cores):
        """
        Amongst awaken cores, filter free cores. Then sleep the amount of cores having the lowest health.
        """
        asleep_cores = list(filter(lambda c: c.forced_to_sleep, cores))

        # health is calculated as the current degraded frequency over the nominal frequency.
        asleep_cores = sorted(asleep_cores, key=lambda c: c.freq / c.freq_0, reverse=True) # health low to high.
        for core in asleep_cores[:min(to_wake, len(asleep_cores))]:
            self.force_wake(core)


    def force_sleep(self, core):
        core.forced_to_sleep = True
        core.task = None
        core.c_state = CStates.C6.value
        core.temp = Temperatures.C6

    def force_wake(self, core):
        core.forced_to_sleep = False
        core.c_state = CStates.C1.value
        core.temp = Temperatures.C0_POLL

@dataclass(kw_only=True)
class GPU(Processor):
    processor_type: ProcessorType = ProcessorType.GPU


if __name__ == "__main__":
    pass
