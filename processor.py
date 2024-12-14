import math
import os
import uuid
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from core_power import CStates, calculate_core_power, get_c_state_from_idle_governor, CState, APPROX_INFINITY_S, \
    generate_initial_frequencies, calc_ADH, calc_aged_freq, Temperatures, CPU_CORE_ADJ_INTERVAL, \
    calc_long_term_vth_shift, gen_init_fq
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
    MAX_FQ = 0.0

    # init for all params.
    def __init__(self, processor_id:uuid, id: int, f_init: float = 2.25 * math.pow(10, 9), temp_init=54):
        # keep track of max frequency for normalization purpose.
        if Core.MAX_FQ < f_init:
            Core.MAX_FQ = f_init

        self.id = id
        self.processor_id = processor_id
        self.task = None
        self.c_state = CStates.C1.value
        self.last_idle_durations = []
        self.last_idle_set_time = 0.0
        self.temp = temp_init
        self.freq = f_init
        self.freq_0 = f_init
        self.last_state_change_time = 0.0
        self.vth_shift = 0.0
        self.cum_aged_time = 0.0
        self.freq_nominal = 2.25 * math.pow(10, 9) # AMD EPCY 7742 base frequency is 2.25 GHz
        self.forced_to_sleep = False
        self.last_temp_update = 0.0

    def set_temp(self, new_temp):
        clk = clock()

        # update aging effects.
        prev_period_tmp = self.temp
        was_sleep = self.forced_to_sleep
        prev_period_length = clk - self.last_temp_update

        if was_sleep:
            self.vth_shift = self.vth_shift
        else:
            self.vth_shift = calc_long_term_vth_shift(vth_old=self.vth_shift, t_temp=prev_period_tmp, t_length=prev_period_length)
        self.cum_aged_time += prev_period_length
        fq_new = calc_aged_freq(initial_freq=self.freq_0, cum_delta_vth=self.vth_shift)
        self.freq = fq_new

        # set new core temperature.
        self.temp = new_temp
        self.last_temp_update = clk

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
            'health': self.get_health(),
            'cum_aged_time': self.cum_aged_time,
            'cum_vth_delta': self.vth_shift,
        }

    def get_health(self):
        #return self.freq / self.freq_0
        return self.freq / Core.MAX_FQ


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


def task_schedule_zhao23(cpu_cores):
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
        selected_core_health = selected_core.get_health()
        core_health = core.get_health()
        if core_health > selected_core_health:
            selected_core = core

    # if none, pick the first free core.
    if selected_core is None:
        for core in cpu_cores:
            if core.task is None:
                selected_core = core
                break

    return selected_core


def task_schedule_proposed(cpu_cores):
    selected_core = None
    selected_idle_score = 0.0
    for core in cpu_cores:
        if core.task is not None: # avoid already busy cores.
            continue

        idle_score = sum(core.last_idle_durations) # tap into the idle subsystem to even-out core usage
        if selected_core is None or idle_score > selected_idle_score:
            selected_core = core
            selected_idle_score = idle_score

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
        #process_variation_induced_initial_core_fq = generate_initial_frequencies(n_cores=self.core_count)

        process_variation_induced_initial_core_fq = gen_init_fq(n_cores=self.core_count)

        self.id = uuid.uuid4()

        # initial temperature is 54 degrees celcius. Data modelled after experiments from Green Core testbed.
        self.cpu_cores = [Core(processor_id=self.id, id=idx, f_init=init_fq, temp_init=Temperatures.C0_POLL.value) for idx, init_fq in
                          enumerate(process_variation_induced_initial_core_fq)]
        self.core_activity_log = []
        self.oversubscribed_task_count_log = 0
        self.total_task_count_log = 0

        self.temp_T_ts = []
        self.temp_running_tasks = []
        self.temp_running_tasks_counter = 0

        # manage core sleeping.
        #self.active_core_limit = len(self.cpu_cores) # Initial parameter. controller should adjust to optmize.
        #self.put_to_sleep(to_sleep=len(self.cpu_cores) - self.active_core_limit, cores=self.cpu_cores) # initial core-set to sleep. Done taking the health into account.
        self.core_oversubscribe_tasks = {
            "past_e_t": [],
            "core_oversubscribe_tasks": 0,
            "last_core_adjust_time": 0.0,
            "core_adjust_dt": CPU_CORE_ADJ_INTERVAL,
            "err_integral": 0.0,
            "prev_error": 0.0,
        }

        self.sleep_manager_logs = []

    # todo: check if task to core balance it + or -. then trigger periodic event to adjust the core count.
    def assign_core_to_cpu_task(self, task, override_task_description=None):
        # monitoring

        # total running tasks
        self.temp_running_tasks_counter += 1

        # gpu memory usage
        mem_used = 0.0
        mem_total = 0.0
        for p in self.server.processors:
            if p == self:
                continue
            mem_used += p.memory_used
            mem_total += p.memory_size
        mem_util = mem_used / mem_total

        # core sleep management
        awaken_cores = len(list(filter(lambda c: not c.forced_to_sleep, self.cpu_cores)))

        # log entry
        self.temp_running_tasks.append([clock(), self.temp_running_tasks_counter, mem_util, awaken_cores])

        self.total_task_count_log += 1

        # assign logic
        assigned_core, time_to_wake = self.get_a_core_to_assign()
        if assigned_core is None:
            self.oversubscribed_task_count_log += 1
            self.core_oversubscribe_tasks['core_oversubscribe_tasks'] = self.core_oversubscribe_tasks['core_oversubscribe_tasks'] + 1
            return -1, 0.0, 1 # there was no free core to assign. Task is assumed to be oversubscribing cpu.

        #self.update_aging(assigned_core)

        # set the core to serve
        # print(task.value)
        assigned_core.task = task.value["info"]
        if override_task_description is not None:
            assigned_core.task = override_task_description
        assigned_core.c_state = CStates.C0.value  # set core to busy
        #assigned_core.temp = Temperatures.C0_RTEVAL # model temperature for executing task
        assigned_core.set_temp(new_temp=Temperatures.C0_RTEVAL.value)
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
        #self.update_aging(core)
        # debug
        if ENABLE_DEBUG_LOGS:
            print(f"Release core: {core.id} from task: {core.task} at time: {clock()}")
        core.task = None
        core.c_state = CStates.C1.value  # set core to idle
        # update core idle state
        next_c_state = get_c_state_from_idle_governor(last_8_idle_durations_s=core.last_idle_durations,
                                                      latency_limit_core_wake_s=APPROX_INFINITY_S)
        core.c_state = next_c_state
        #core.temp = next_c_state.temp  # model temperature for idle core
        core.set_temp(new_temp=next_c_state.temp.value)
        core.last_idle_set_time = clock()
        self.log_cpu_state(core=core, time_to_wake=None)

    def adjust_sleeping_cores(self):

        # cores
        N = len(self.cpu_cores)

        active_cores = len(list(filter(lambda c: not c.forced_to_sleep, self.cpu_cores)))
        C_SLP_t = N - active_cores

        # tasks
        oversub_tasks = self.core_oversubscribe_tasks["core_oversubscribe_tasks"]
        normal_tasks = len(list(filter(lambda c: c.task is not None and not c.forced_to_sleep, self.cpu_cores)))
        T_t = normal_tasks + oversub_tasks
        self.temp_T_ts.append(T_t)

        # assumed in llm inference servers number of available cores are excessive.
        # this is an algorithmic estimation, not
        # a part of the system model.
        # algorithm performs online optimization based on this assumption.
        T_t = min(N, T_t)

        # error signal
        e_t = N - C_SLP_t - T_t

        e_t_prd = e_t

        # normalize
        e_t_prd = e_t_prd / N

        # apply reaction function
        F_e_t_prd = e_t_prd
        if e_t_prd >= 0:
            F_e_t_prd = math.tan(0.785 * e_t_prd)
        else:
            F_e_t_prd = math.atan(1.55 * e_t_prd)

        # scale up
        e_t_corr = N * F_e_t_prd

        # final error signal
        e_t_corr = int(e_t_corr)

        # put cores to sleep
        delta_cores = abs(e_t_corr)
        if e_t_corr > 0:
            self.put_to_sleep(to_sleep=delta_cores, cores=self.cpu_cores)
        elif e_t_corr < 0:
            self.put_to_wake(to_wake=delta_cores, cores=self.cpu_cores)

        self.sleep_manager_logs.append({
            'clock': clock(),
            'oversub_tasks': oversub_tasks,
            'normal_tasks': normal_tasks,
            'T_t': T_t,
            'active_cores': active_cores,
            'asleep_cores': len(self.cpu_cores) - active_cores,
        })

    # def update_aging(self, assigned_core):
    #     clk = clock()
    #
    #     time_delta_since_last_state_change = clk - assigned_core.last_state_change_time
    #     assigned_core.last_state_change_time = clk
    #
    #     is_sleep = assigned_core.forced_to_sleep
    #     if is_sleep:
    #         vth_delta = 0.0 # stress is zero when core forced to sleep.Else,
    #         # there is a possibility that a floating task or LLM task used the core,
    #         # causing the stress.
    #     else:
    #         vth_delta = calc_ADH(t_elapsed_time=time_delta_since_last_state_change, temp_celsius=assigned_core.temp.value)
    #
    #     assigned_core.vth_delta += vth_delta
    #     aged_freq = calc_aged_freq(initial_freq=assigned_core.freq_0, delta_vth=assigned_core.vth_delta)
    #     assigned_core.freq = aged_freq

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
        if task_allocation_algo == "linux":
            assigned_core = task_schedule_linux(cpu_cores=awaken_cores)
        elif task_allocation_algo == "zhao23":
            assigned_core = task_schedule_zhao23(cpu_cores=awaken_cores)
        elif task_allocation_algo == "proposed":
            assigned_core = task_schedule_proposed(cpu_cores=awaken_cores)
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
        free_awake_cores = sorted(free_awake_cores, key=lambda c: c.get_health()) # health low to high.
        sleep_count = 0
        for idx in range(len(free_awake_cores)):
            core = free_awake_cores[idx]
            #self.update_aging(core)
            self.force_sleep(core)
            self.log_cpu_state(core=core, time_to_wake=None)
            sleep_count += 1
            if sleep_count >= to_sleep:
                break

    def put_to_wake(self, to_wake, cores):
        """
        Amongst awaken cores, filter free cores. Then sleep the amount of cores having the lowest health.
        """
        asleep_cores = list(filter(lambda c: c.forced_to_sleep, cores))

        # health is calculated as the current degraded frequency over the nominal frequency.
        asleep_cores = sorted(asleep_cores, key=lambda c: c.get_health(), reverse=True) # health low to high.
        wake_count = 0
        for idx in range(len(asleep_cores)):
            core = asleep_cores[idx]
            #self.update_aging(core)
            self.force_wake(core)
            self.log_cpu_state(core=core, time_to_wake=None)
            wake_count += 1
            if wake_count >= to_wake:
                break

    def force_sleep(self, core):
        core.set_temp(new_temp=Temperatures.C6.value) # needs to be at top.
        core.forced_to_sleep = True
        core.task = None
        core.c_state = CStates.C6.value

    def force_wake(self, core):
        core.set_temp(new_temp=Temperatures.C0_POLL.value) # needs to be at top.
        core.forced_to_sleep = False
        core.c_state = CStates.C1.value

@dataclass(kw_only=True)
class GPU(Processor):
    processor_type: ProcessorType = ProcessorType.GPU


if __name__ == "__main__":
    pass
