import math
from enum import Enum, IntEnum
from xml.dom.expatbuilder import FilterVisibilityController

# approximate infinity value for the latency limit of the core wake-up time.
# requirement is to have an upper bound for all core idle state transition times.
# we set to an hour, because all core transition times are practically less than that.
APPROX_INFINITY_S = 60 * 60

CPU_CORE_ADJ_INTERVAL = 1


class CState:
    state: str
    target_residency_s: float
    transition_time_s: float
    power_w: float
    p_state: str
    temp: float

    def __init__(self, state, target_residency_s, transition_time_s, power_w, p_state, temp):
        self.state = state
        self.target_residency_s = target_residency_s
        self.transition_time_s = transition_time_s
        self.power_w = power_w
        self.p_state = p_state
        self.temp = temp

    def __str__(self):
        return (
            f"CState(state={self.state}, "
            f"target_residency_s={self.target_residency_s}, "
            f"transition_time_s={self.transition_time_s}, "
            f"power_w={self.power_w})"
            f"temp_c={self.temp})"
        )


class Temperatures(float, Enum):
    C0_RTEVAL = 54.00
    C0_POLL = 51.08
    C6 = 48.00


class CStates(Enum):
    """Server CPU C-states from Table 1 of [1].
    [1] J. H. Yahya et al., "AgileWatts: An Energy-Efficient CPU Core Idle-State Architecture for Latency-Sensitive
    Server Applications," 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO), Chicago, IL, USA,
    2022, pp. 835-850, doi: 10.1109/MICRO56248.2022.00063. keywords: {Degradation;Program processors;Microarchitecture;
    Coherence;Market research;Energy efficiency;Generators;Energy Efficiency;power management;Latency Sensitive applications},
    """
    C0 = CState('C0', 0.0, 0.0, 4.0, 'P1',
                temp=Temperatures.C0_POLL)  # active and executing instructions at highest performance state
    C1 = CState('C1', 2e-6, 2e-6, 1.44, 'P1', temp=Temperatures.C0_POLL)  # idle but online
    C6 = CState('C6', 0.0006, 0.000133, 0.1, p_state=None, temp=Temperatures.C6)  # deep sleep state


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
    idle_queue = last_8_idle_durations_s.copy()
    if idle_queue is None:
        idle_queue = []
    predicted_idle_duration = APPROX_INFINITY_S
    while len(idle_queue) > 0:
        average = sum(idle_queue) / len(idle_queue)
        variance = sum((x - average) ** 2 for x in idle_queue) / len(idle_queue)
        standard_deviation = variance ** 0.5
        if variance < 0.0004 or average > 6 * standard_deviation:
            predicted_idle_duration = average
            break
        idle_queue.remove(max(idle_queue))

    latency_limit = predicted_idle_duration
    number_of_tasks_waiting_on_io = 0  # we assume LLM inference tasks are CPU bound
    latency_limit_of_power_mgt_qos = latency_limit_core_wake_s
    if number_of_tasks_waiting_on_io > 0:
        latency_limit = latency_limit / number_of_tasks_waiting_on_io
    latency_limit = min(latency_limit, latency_limit_of_power_mgt_qos)

    c_states = [state.value for state in CStates if
                state.value.state != "C0"]  # C0 indicate active and executing instructions, which is not idle
    chosen_c_state = list(filter(lambda x: x.state == "C1", c_states))[0]  # default to C1 = idle but online

    # === We assume system tasks floats on the cores. Which means, last 8 idle durations are not accurate for OS level
    # idle state modeling (a core could be executing a system task). So we only let C1. Enter to C6 is not allowed. ===
    # for c_state in c_states:
    #     target_residency = c_state.target_residency_s
    #     transition_time = c_state.transition_time_s
    #     if transition_time > latency_limit:
    #         continue
    #     if predicted_idle_duration > target_residency:
    #         gap = predicted_idle_duration - target_residency
    #         current_gap = predicted_idle_duration - chosen_c_state.target_residency_s
    #         if gap < current_gap:
    #             chosen_c_state = c_state

    return chosen_c_state


def calculate_WTTF(cpu_model, time_s, c_state, freq):
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
    wttf = c_state['IPC'] * freq * time_s
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


def calc_long_term_vth_shift(vth_old, t_length, t_temp, n=0.17):
    """
    We use a recursive vth calculation model from,
    Moghaddasi, I., Fouman, A., Salehi, M. E., & Kargahi, M. (2018). Instruction-level NBTI stress estimation and its
    application in runtime aging prediction for embedded processors. IEEE Transactions on Computer-Aided Design of
    Integrated Circuits and Systems, 38(8), 1427-1437.

    Split time into each measurement interval. For each interval, time length and temperature is given.

    t_length: time in seconds
    t_temp: temperature in Celsius
    """
    ADH = calc_ADH(temp_celsius=t_temp)

    f_1 = vth_old / ADH
    f_2 = math.pow(f_1, 1 / n)
    vth_new = ADH * math.pow((f_2 + t_length), n)
    return vth_new


def calc_ADH(temp_celsius=26.0, n=0.17):
    """
    Calculate the shift in threshold voltage.

    Source: ATLAS paper.
    """
    temp_kelvin = temp_celsius + 273.15

    lithography = '22nm'

    K_B_boltzman_constant = 0.00008617
    E_0 = 0.1897  # eV
    B = 0.075  # nm/V
    t_ox = ATLAS_PARAMS[lithography]['t_ox']
    Vdd = ATLAS_PARAMS[lithography]['Vdd']

    A_T_Vdd = (math.exp(-E_0 / (K_B_boltzman_constant * temp_kelvin))
               * math.exp((B * Vdd) / (t_ox * K_B_boltzman_constant * temp_kelvin)))

    """ATLAS: for 22nm, worstcase degradation is 30% after 10 years. Our system worst temperature is 54.
    If a core continously operate at 54C with 1.0 stress (full utilization) for 10 years, then frequency should degrade by 30%.
    Solving the delta_vth equation for this scenario yield following fitting parameter.
    """
    k_fitting_param = 1.06980863

    # we assume that all tasks yield 1.0 stress (max utilization). Even not execute a task, if core is awake, it might serve floating sytem task.
    # in our model, only forced sleep cores are truly having 0 stress. Caller should not call this function for sleeping cores.
    Y = 1.0  # amount of stress.

    # delta_vth = k_fitting_param * A_T_Vdd * math.pow(Y, n) * math.pow(t_elapsed_time, n)
    ADH = k_fitting_param * A_T_Vdd * math.pow(Y, n)
    return ADH


def calc_aged_freq(initial_freq, cum_delta_vth):
    """
    Calculate the core frequency w.r.t. aging. initial frequency is the process variation induced initial frequency of
    the core.
    """
    lithography = '22nm'
    Vdd = ATLAS_PARAMS[lithography]['Vdd']
    Vth = ATLAS_PARAMS[lithography]['Vth']

    return initial_freq * (1 - (cum_delta_vth / (Vdd - Vth)))


import numpy as np


def gen_init_fq(n_cores=128):
    """
    Generates the initial frequencies for a given number of processor cores
    based on process parameters modeled as a Gaussian distribution. The function
    calculates the correlations of process parameters across a 2D grid of tiles
    in each core and derives the maximum frequency (f_max) for each core.

    Process variation model is derived from the following paper: Raghunathan, B., Turakhia, Y., Garg, S., & Marculescu,
    D. (2013, March). Cherry-picking: Exploiting process
    variations in dark-silicon homogeneous chip multi-processors. In 2013 Design, Automation & Test in Europe
    Conference & Exhibition (DATE) (pp. 39-44). IEEE.

    Args:
    - n_cores (int, optional): The number of processor cores. Defaults to 128.

    Returns:
    - List[float]: A list of maximum frequencies (f_max) for the given number
      of cores.

    Attributes:
    - f_nom (float): The nominal frequency of the processor in GHz, set to 2.25 GHz.
    - N (int): The grid dimension for the core, set to 100 for a 100x100 grid.
    - mu_p (float): Mean process parameter value derived from the nominal frequency.
    - sig_p (float): Standard deviation of the process parameter, set as
      10% of mu_p.
    """
    f_nom = 2.25  # GHz

    # We assume that critical paths are uniformly distributed across the core (all grid tiles).
    N = 10  # 100x100 grid

    """
    K_dash = 1 and say no process variations. Then, f_max of the core = 1 * min (1 / process_parameter). No pro. para.
    means f_max = 1/ pro.para. Then f_max must match nominal fq. Which derives, pro. para. = 1 / f_nominal. Without 
    variations, pro.para. should match the mean of the gaussian dst, to which pro.para is modelled. Thus, mu_p = 1 / f_nominal.
    """
    mu_p = 1.0 / f_nom
    sig_p = 0.1 * mu_p  # 10% of mu_p

    fqs = []
    for idx in range(n_cores):
        print(f"Generating initial frequency for a core: {idx}...")

        # Grid point coordinates
        x, y = np.meshgrid(np.arange(N), np.arange(N))
        grid_points = np.column_stack([x.ravel(), y.ravel()])  # Shape (N*N, 2)

        # Calculate pairwise Euclidean distances
        distances = np.linalg.norm(grid_points[:, np.newaxis, :] - grid_points[np.newaxis, :, :], axis=2)

        """
        Estimate half-of die with half of N_chip. At that distance apart, correlation coefficient of pro. paras. are 0.1.
        """
        alpha = 4.60512 / N # from cherry pick paper: halfway distance, correlation < 0.1. Solve for that to get this equation.

        correlation_matrix = np.exp(-1 * alpha * distances)

        # Create covariance matrix
        covariance_matrix = (sig_p ** 2) * correlation_matrix

        # Generate samples using multivariate normal distribution
        rho_vals = np.random.multivariate_normal(
            mean=np.full(N * N, mu_p),  # Mean vector
            cov=covariance_matrix  # Covariance matrix
        )

        # Reshape samples to match the grid shape
        rho_vals = rho_vals.reshape(N, N)

        # take inverse of each value in the rho_vals
        rho_vals = 1 / rho_vals

        # take minimum of the rho_vals
        f_max = min(rho_vals.flatten())
        fqs.append(f_max)

    return fqs


def generate_initial_frequencies(n_cores=128):
    """Raghunathan, B., Turakhia, Y., Garg, S., & Marculescu, D. (2013, March). Cherry-picking: Exploiting process
    variations in dark-silicon homogeneous chip multi-processors. In 2013 Design, Automation & Test in Europe
    Conference & Exhibition (DATE) (pp. 39-44). IEEE.
    """
    frequencies = []
    CORE_FINE_GRID_LENGTH = 10
    for num_cores in range(n_cores):
        # Constants
        f_nominal = 2.25  # GHz
        mu_p = 1.0  # Mean of process parameter
        sigma_p = 0.1 * mu_p  # Standard deviation (10% of mu_p)
        alpha = 9.21 / CORE_FINE_GRID_LENGTH  # Spatial correlation parameter, assuming L_die = 100

        # Core grid dimensions and critical path cell set
        N_chip = CORE_FINE_GRID_LENGTH  # 100x100 grid
        N_cp = 10  # Number of critical paths per core
        critical_path_cells = np.random.choice(range(N_chip ** 2), N_cp,
                                               replace=False)  # Random selection of cells for SCP

        # Generate process parameters
        grid = np.random.normal(mu_p, sigma_p, (N_chip, N_chip))  # Gaussian process parameters

        # Spatial correlation modeling
        correlation_matrix = np.zeros((N_chip, N_chip))
        for i in range(N_chip):
            for j in range(N_chip):
                for k in range(N_chip):
                    for l in range(N_chip):
                        dist = np.sqrt((i - k) ** 2 + (j - l) ** 2)
                        correlation_matrix[i, j] = np.exp(-alpha * dist)

        # Adjust process parameters for spatial correlation
        for i in range(N_chip):
            for j in range(N_chip):
                grid[i, j] += np.random.normal(0, sigma_p) * correlation_matrix[i, j]

        # Calculate f_MAX for a core
        critical_path_values = [1 / grid[cell // N_chip, cell % N_chip] for cell in critical_path_cells]
        f_max = f_nominal * (mu_p / min(critical_path_values))

        # Output sampled f_MAX
        frequencies.append(float(f_max))
    return frequencies


import matplotlib.pyplot as plt


def unit_test_pv_induced_fq():
    # fqs = gen_init_fq(n_cores=128)
    fqs = gen_init_fq(n_cores=1)
    # # plot distribution
    # plt.hist(fqs, bins=len(fqs), color='blue', edgecolor='black')
    # plt.title('PV-led Frequency Distribution (nominal is 2.25 GHz)')
    # plt.show()


def unit_test_aging():
    # testing cal_delta_vth
    # T_c = 51.08
    # time = 1.409

    plt.figure(figsize=(7, 4))

    f_o = 2.7

    t_slots = [54.0, 51.08, 48.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0]
    x = plot_aging(f_o, t_slots, lbl='[54.0, 51.08, 48.0]')

    t_slots = [51.08 for _ in range(10)]
    x = plot_aging(f_o, t_slots, lbl='[54.0 all]')

    t_slots = [51.08 if i % 2 == 0 else -1 for i in range(10)]
    x = plot_aging(f_o, t_slots, lbl='[54.0 with sleep]')

    t_slots = [54.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0]
    x = plot_aging(f_o, t_slots, lbl='[54.0, 54.0, 54.0]')

    t_slots = [54.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0]
    x = plot_aging(f_o, t_slots, lbl='ref:static 54C', is_ref=True)

    plt.xticks(x)
    plt.grid()
    plt.legend()
    plt.xlabel('Years')
    plt.ylabel('Frequency Degradation (%)')
    plt.title('NBTI induced aging: Frequency Degradation Over Time')
    plt.show()


def plot_aging(f_o, t_slots, lbl, is_ref=False):
    x = [0]
    y = [0]
    f_now = f_o
    v_th_now = 0.0
    for year in range(1, len(t_slots) + 1):
        elapsed_t = 365 * 24 * 60 * 60
        T_c = t_slots[year - 1]

        if not is_ref:
            if T_c != -1:
                v_th_now = calc_long_term_vth_shift(vth_old=v_th_now, t_length=elapsed_t, t_temp=54.0)
            else:
                v_th_now = v_th_now
            f_now = calc_aged_freq(initial_freq=f_o, cum_delta_vth=v_th_now)
        else:
            v_th_now = calc_long_term_vth_shift(vth_old=0, t_length=elapsed_t * year, t_temp=54.0)
            f_now = calc_aged_freq(initial_freq=f_o, cum_delta_vth=v_th_now)

        x.append(year)
        degredation = round((f_o - f_now) * 100, 2) / f_o
        y.append(degredation)
    # plot y vs x
    plt.plot(x, y, marker='o', label=lbl)
    return x


# unit_test_aging()
#unit_test_pv_induced_fq()
