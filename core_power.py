import math
from enum import Enum
from xml.dom.expatbuilder import FilterVisibilityController

# approximate infinity value for the latency limit of the core wake-up time.
# requirement is to have an upper bound for all core idle state transition times.
# we set to an hour, because all core transition times are practically less than that.
APPROX_INFINITY_S = 60 * 60


class CState:
    state: str
    target_residency_s: float
    transition_time_s: float
    power_w: float
    p_state: str

    def __init__(self, state, target_residency_s, transition_time_s, power_w, p_state):
        self.state = state
        self.target_residency_s = target_residency_s
        self.transition_time_s = transition_time_s
        self.power_w = power_w
        self.p_state = p_state

    def __str__(self):
        return (
            f"CState(state={self.state}, "
            f"target_residency_s={self.target_residency_s}, "
            f"transition_time_s={self.transition_time_s}, "
            f"power_w={self.power_w})"
        )


class CStates(Enum):
    """Server CPU C-states from Table 1 of [1].
    [1] J. H. Yahya et al., "AgileWatts: An Energy-Efficient CPU Core Idle-State Architecture for Latency-Sensitive
    Server Applications," 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO), Chicago, IL, USA,
    2022, pp. 835-850, doi: 10.1109/MICRO56248.2022.00063. keywords: {Degradation;Program processors;Microarchitecture;
    Coherence;Market research;Energy efficiency;Generators;Energy Efficiency;power management;Latency Sensitive applications},
    """
    C0 = CState('C0', 0.0, 0.0, 4.0, 'P1')  # active and executing instructions at highest performance state
    C1 = CState('C1', 2e-6, 2e-6, 1.44, 'P1')  # idle but online
    C6 = CState('C6', 0.0006, 0.000133, 0.1, p_state=None)  # deep sleep state


c_state_data = {
    'dual-amd-rome-7742': {
        'C0': {
            "state": "C0",
            "transition_time_s": 0.0,
            "target_residency_s": 0.0,
            "core_power_w": 2.572,
            "IPC": 1.0  # indicative value, not the actual. We approximate it to 1.0 for active state. In reality,
            # the value depends on the workload. With inference, CPU workload is almost homogeneous, as it does not
            # change according to request characteristics such as token count.
        },
        'C1': {
            "state": "C1",
            "transition_time_s": 2e-6,
            "target_residency_s": 2e-6,
            "core_power_w": 2.572 * 0.30,
            "IPC": 0.0  # halted state. no instructions executed.
        },
        'C6': {
            "state": "C6",
            "transition_time_s": 0.000133,
            "target_residency_s": 0.0006,
            "core_power_w": 2.572 * 0.025,
            "IPC": 0.0
        },
    },
    'dual-xeon-platinum-8480c': {
        'C0': {
            "state": "C0",
            "transition_time_s": 0.0,
            "target_residency_s": 0.0,
            "core_power_w": 4.0,
            "IPC": 1.0  # indicative value, not the actual. We approximate it to 1.0 for active state. In reality,
            # the value depends on the workload. With inference, CPU workload is almost homogeneous, as it does not
            # change according to request characteristics such as token count.
        },
        'C1': {
            "state": "C1",
            "transition_time_s": 2e-6,
            "target_residency_s": 2e-6,
            "core_power_w": 4.0 * 0.30,  # ref: 'sleep well' paper
            "IPC": 0.0  # halted state. no instructions executed.
        },
        'C6': {
            "state": "C6",
            "transition_time_s": 0.000133,
            "target_residency_s": 0.0006,
            "core_power_w": 4.0 * 0.025,
            "IPC": 0.0
        },
    }
}


def get_c_states(cpu_model):
    """Server CPU C-states.
    To model idle states of server CPUs, we create a model based on specification values provided for Intel server CPUs.
    [1] J. H. Yahya et al., "AgileWatts: An Energy-Efficient CPU Core Idle-State Architecture for Latency-Sensitive
    Server Applications," 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO), Chicago, IL, USA,
    2022, pp. 835-850, doi: 10.1109/MICRO56248.2022.00063. keywords: {Degradation;Program processors;Microarchitecture;
    Coherence;Market research;Energy efficiency;Generators;Energy Efficiency;power management;Latency Sensitive applications},

    information: https://lenovopress.lenovo.com/lp1945-using-processor-idle-c-states-with-linux-on-thinksystem-servers
    """

    return c_state_data[cpu_model]

    # if cpu_model == 'dual-amd-rome-7742':
    #     # relative to intel: TPD: 350 vs 225 ==> 1.56 times less power
    #     c0_pw_w = 2.572 # 4.0 * 0.643 = 2.572
    #     return [
    #         {
    #             "state": "C0",
    #             "transition_time_s": 0.0,
    #             "target_residency_s": 0.0,
    #             "core_power_w": c0_pw_w,
    #             "IPC": 1.0  # indicative value, not the actual. We approximate it to 1.0 for active state. In reality,
    #             # the value depends on the workload. With inference, CPU workload is almost homogeneous, as it does not
    #             # change according to request characteristics such as token count.
    #         },
    #         {
    #             "state": "C1",
    #             "transition_time_s": 2e-6,
    #             "target_residency_s": 2e-6,
    #             "core_power_w": c0_pw_w * 0.30,
    #             "IPC": 0.0  # halted state. no instructions executed.
    #         },
    #         {
    #             "state": "C6",
    #             "transition_time_s": 0.000133,
    #             "target_residency_s": 0.0006,
    #             "core_power_w": c0_pw_w * 0.025,
    #             "IPC": 0.0
    #         },
    #         # {
    #         #     # https://www.techpowerup.com/cpu-specs/epyc-7763.c2373
    #         #     "state": "package",
    #         #     "tdp": 205.0,
    #         #     "tdp_divided_per_core": 7.3,  # 205/28
    #         #     "c_state_power_at_tdp_per_core": 3.3,  # 7.3 - 4.0 (assuming core operates at C0 state)
    #         #     "package_overhead_per_core": 3.3
    #         # }
    #     ]
    #
    # elif cpu_model == 'dual-xeon-platinum-8480c':
    #     c0_pw_w = 4.0
    #     return [
    #         {
    #             "state": "C0",
    #             "transition_time_s": 0.0,
    #             "target_residency_s": 0.0,
    #             "core_power_w": c0_pw_w,
    #             "IPC": 1.0  # indicative value, not the actual. We approximate it to 1.0 for active state. In reality,
    #             # the value depends on the workload. With inference, CPU workload is almost homogeneous, as it does not
    #             # change according to request characteristics such as token count.
    #         },
    #         {
    #             "state": "C1",
    #             "transition_time_s": 2e-6,
    #             "target_residency_s": 2e-6,
    #             "core_power_w": c0_pw_w * 0.30, # ref: 'sleep well' paper
    #             "IPC": 0.0  # halted state. no instructions executed.
    #         },
    #         {
    #             "state": "C6",
    #             "transition_time_s": 0.000133,
    #             "target_residency_s": 0.0006,
    #             "core_power_w": c0_pw_w * 0.025,
    #             "IPC": 0.0
    #         },
    #         # {
    #         #     # https://www.techpowerup.com/cpu-specs/epyc-7763.c2373
    #         #     "state": "package",
    #         #     "tdp": 205.0,
    #         #     "tdp_divided_per_core": 7.3,  # 205/28
    #         #     "c_state_power_at_tdp_per_core": 3.3,  # 7.3 - 4.0 (assuming core operates at C0 state)
    #         #     "package_overhead_per_core": 3.3
    #         # }
    #     ]
    #
    # else:
    #     raise ValueError(f"Unknown CPU model: {cpu_model}")


'''specs
DGX H100 - https://resources.nvidia.com/en-us-dgx-systems/ai-enterprise-dgx?xs=489753
DGX A100 - https://images.nvidia.com/aem-dam/Solutions/Data-Center/nvidia-dgx-a100-datasheet.pdf
'''
machine_specs = {
    'dual-xeon-platinum-8480c': {  # Dual Intel® Xeon® Platinum 8480C
        'cores': 112,
        'refresh_cycle_years': 3,
        'cpu_tdp_w': 700,
        'rest_of_pkg_power_w': 252,
        'c0_power_w': 4.0,
        'c6_power_w': 0.1,
        # this is assumed to be a constant. rest_of_pkg_power_w + num_cores * c0_power = cpu_tdp_w
        # C-state power values are approximated with Intel Skylake c-state idle power consumption
    },
    'dual-amd-rome-7742': {  # Dual AMD Rome 7742
        'cores': 128,
        'refresh_cycle_years': 3,
        # https://mcomputers.cz/en/products-and-services/nvidia/dgx-systems/nvidia-dgx-a100/
        'cpu_tdp_w': 450,
        'rest_of_pkg_power_w': 117.2,
        # idle_power is 130 W (https://www.anandtech.com/show/16778/amd-epyc-milan-review-part-2/3). assume idle is all cores at c6.
        # num_cores * c6_power + rest_of_pkg_power_w = idle_power
        'c0_power_w': 2.6,
        # num_cores * c0_power + rest_of_pkg_power_w = cpu_tdp_w
        'c1_power_w': 0.936,
        # Approx. Intel skylake: C1 power is 0.36 times C6 power.
        'c6_power_w': 0.1,
        # approximated with Intel skylake C6
    },
}


def get_core_power_for_model(model_name):
    data = {
        'llama': 1.968789,
        'mistral': 3.040158,
        'falcon': 2.986340
    }

    model_name = list(filter(lambda name: name in model_name, data.keys()))
    if not model_name:
        model_name = None

    if model_name is None:
        # return average
        return sum(data.values()) / len(data)

    return data[model_name[0]]


def calculate_core_power(c_state, model):
    """
    Placeholder function to calculate the power of a core.
    Parameters:
    - c_state: The c-state of the core.
    - model: The model currently executed by the core.
    Returns:
    - power value.
    """
    # if model is not None:
    #     # Use observed data of core serving a model
    #     return get_core_power_for_model(model_name=model)

    # Use c-state for idle state power consumption
    return c_state.power_w


def get_c_state_from_idle_governor(last_8_idle_durations_s=None, latency_limit_core_wake_s=APPROX_INFINITY_S):
    """Implements Menu governer algorithm[1] to calculate the C-state.

    There are several steps in selecting a c-state per-core. Goal here is to correctly predict the idle duration of the
    cpu, and select the appropriate c-state such that power saving and transition latency are balanced.

    1. Initial idle duration is predicted based on next os scheduler event. In ours, we assume that all other system
    services are executed using dedicated cores and cores handle LLM inference are dedicated to that task only. Thus,
    scheduler events do not interrupt those cores.

    2. Predicted value is then adjusted for correction. For example, say predicted value is 50 ms. But typically cores
    never stay idle for that long. So pre-calculated correction factor, say 0.8, is applied. eg: 50ms * 0.8 = 40ms. In
    ours, we do not calculate that.

    3. Next pattern recognition. Last 8 idle durations observed are saved. If the variance of those 8 values are lower,
    the average of those values is considered as a typical idle duration. Compared to that average, if the predicted
    value so far is higher, then the average is taken as the idle duration (i.e. take min). In ours, since we do not
    calculate the initial idle duration, we start from the typical idle duration calculation and take that as the idle
    duration.

    4. Next, a latency limit is applied to help interactive workloads.

    5. Afterwards, the appropriate c-state is selected by comparing their target residency and transition latency with
    the calculated idle duration.

    [1] https://www.kernel.org/doc/html/v5.4/admin-guide/pm/cpuidle.html
    """
    if last_8_idle_durations_s is None:
        last_8_idle_durations_s = []
    predicted_idle_duration = APPROX_INFINITY_S
    while len(last_8_idle_durations_s) > 0:
        average = sum(last_8_idle_durations_s) / len(last_8_idle_durations_s)
        variance = sum((x - average) ** 2 for x in last_8_idle_durations_s) / len(last_8_idle_durations_s)
        standard_deviation = variance ** 0.5
        if variance < 0.0004 or average > 6 * standard_deviation:
            predicted_idle_duration = average
            break
        last_8_idle_durations_s.remove(max(last_8_idle_durations_s))

    latency_limit = predicted_idle_duration
    number_of_tasks_waiting_on_io = 0  # we assume LLM inference tasks are CPU bound
    latency_limit_of_power_mgt_qos = latency_limit_core_wake_s
    if number_of_tasks_waiting_on_io > 0:
        latency_limit = latency_limit / number_of_tasks_waiting_on_io
    latency_limit = min(latency_limit, latency_limit_of_power_mgt_qos)

    c_states = [state.value for state in CStates if
                state.value.state != "C0"]  # C0 indicate active and executing instructions, which is not idle
    chosen_c_state = list(filter(lambda x: x.state == "C1", c_states))[0]  # default to C1 = idle but online
    for c_state in c_states:
        target_residency = c_state.target_residency_s
        transition_time = c_state.transition_time_s
        if transition_time > latency_limit:
            continue
        if predicted_idle_duration > target_residency:
            gap = predicted_idle_duration - target_residency
            current_gap = predicted_idle_duration - chosen_c_state.target_residency_s
            if gap < current_gap:
                chosen_c_state = c_state

    return chosen_c_state


def calculate_WTTF(cpu_model, time_s, c_state):
    """
    Placeholder function to calculate the Weighted Time to First Failure (WTTF) of the system [1].

    [1] "The case of unsustainable affinity" - hotcarbon'23

    Returns:
    - WTTF value.
    """
    # c_state = list(filter(lambda state: state["state"] == c_state, get_c_states(cpu_model=cpu_model)))[0]
    c_state = get_c_states(cpu_model=cpu_model)[c_state]
    '''Calculation of WTTF
    WTTF = SUM(ipc * operating_frequency * delta_t)
    
    ipc: Instructions per cycle. Based on the workload. In inference, the CPU workload is almost homogeneous across 
    requests. It does not depends on the number of tokens in the request. Thus, we use an indicative constant value of 
    that per c-state. Across c-states, the indicative value is adjusted relatively.
    
    operating_frequency: To isolate the effect of usage, we do not delve into the effect of dynamic frequency. We assume
    servers are tuned to provide a constant performance via a constant cpu frequency.
    '''
    wttf = c_state['IPC'] * 1.0 * time_s
    return wttf


ATLAS_PARAMS = {
    '130nm': {
        'Vdd': 1.3,
        'Vth': 0.2,
        't_ox': 2.25
    },
    '45nm': {
        'Vdd': 1.1,
        'Vth': 0.2,
        't_ox': 1.75
    },
    '32nm': {
        'Vdd': 1.0,
        'Vth': 0.22,
        't_ox': 1.65
    },
    '22nm': {
        'Vdd': 0.9,
        'Vth': 0.25,
        't_ox': 1.4
    },
    '14nm': {
        'Vdd': 0.8,
        'Vth': 0.31,
        't_ox': 0.9
    },
}


def calc_delta_vth(t_elapsed_time=0.0, temp_kelvin=300):
    """
    Calculate the shift in threshold voltage.

    Source: ATLAS paper.
    """

    lithography = '22nm'

    Y_stress_mv = 50  # Stress voltage in mV for NBTI
    n = 0.17
    K_B_boltzman_constant = 8.617e-5
    E_0 = 0.189  # eV
    B = 0.075  # nm/V
    t_ox = ATLAS_PARAMS[lithography]['t_ox']
    Vdd = ATLAS_PARAMS[lithography]['Vdd']

    A_T_Vdd = (math.exp(-E_0 / (K_B_boltzman_constant * temp_kelvin))
               * math.exp((B * Vdd) / (t_ox * K_B_boltzman_constant * temp_kelvin)))

    '''
    SUIT paper: "modern sub 20 nm FinFET transistors degrades by approximately 15 % over a time span of 10 years at >100 C"
    Solving the below equation for 22nm lithography: running the core at 100 celcius for 10 years, the increase of 
    delta_vth is 0.15. Using that, k = 0.18026
    '''
    k_fitting_param = 0.18026
    delta_vth = k_fitting_param * A_T_Vdd * math.pow(Y_stress_mv, n) * math.pow(t_elapsed_time, n)

    return delta_vth


def calc_aged_freq(initial_freq, delta_vth):
    """
    Calculate the core frequency w.r.t. aging. initial frequency is the process variation induced initial frequency of
    the core.
    """
    lithography = '22nm'
    Vdd = ATLAS_PARAMS[lithography]['Vdd']
    Vth = ATLAS_PARAMS[lithography]['Vth']

    return initial_freq * (1 - (delta_vth / (Vdd - Vth)))


import numpy as np


def generate_initial_frequencies(n_cores=128):
    """
    Generate a set of initial frequency values for the cores.

    Parameters:
    - N_cores (int): Number of cores.
    - N_chip (int): Size of the grid (N_chip x N_chip).
    - E (float): Parameter in the exponential function for the correlation.
    - frequency_factor (float): Factor to multiply with the minimum value.

    Returns:
    - initial_frequency_values (list): List of initial frequency values for each core.
    """

    frequencies = []
    for num_cores in range(n_cores):
        '''
        B. Raghunathan, Y. Turakhia, S. Garg and D. Marculescu, "Cherry-picking: Exploiting process variations in 
        dark-silicon homogeneous chip multi-processors," 2013 Design, Automation & Test in Europe Conference & Exhibition 
        (DATE), Grenoble, France, 2013, pp. 39-44, doi: 10.7873/DATE.2013.023.,
        '''
        # Grid size
        grid_size = 100

        # Nominal frequency for each random variable (mean of the Gaussian distribution)
        nominal_frequency = 1.0  # You can adjust this value

        # Standard deviation of the Gaussian distribution
        std_dev = 0.1 * nominal_frequency

        # Compute the correlation parameter for spatial decay
        distance_threshold = 50
        target_correlation = 0.01
        neg_param = -np.log(target_correlation) / distance_threshold

        # Generate a grid of points
        x = np.arange(grid_size)
        y = np.arange(grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Compute the correlation matrix
        num_points = grid_points.shape[0]
        correlation_matrix = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(num_points):
                distance = np.linalg.norm(grid_points[i] - grid_points[j])
                correlation_matrix[i, j] = np.exp(neg_param * distance)

        # Generate the Gaussian random variables with spatial correlation
        mean = np.full(num_points, nominal_frequency)
        covariance_matrix = correlation_matrix * (std_dev ** 2)
        samples = np.random.multivariate_normal(mean, covariance_matrix, size=1).reshape(grid_size, grid_size)

        # Invert the sampled values and find the minimum inverted value
        inverted_samples = 1 / samples
        aged_frequency = 1 * np.min(inverted_samples) # we set technology dependent parameter to 1
        frequencies.append(aged_frequency)
    return frequencies
