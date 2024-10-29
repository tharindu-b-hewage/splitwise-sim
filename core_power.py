from enum import Enum

class CState:
    state: str
    target_residency_s: float
    transition_time_s: float
    power_w: float

    def __init__(self, state, target_residency_s, transition_time_s, power_w):
        self.state = state
        self.target_residency_s = target_residency_s
        self.transition_time_s = transition_time_s
        self.power_w = power_w

    def __str__(self):
        return (
            f"CState(state={self.state}, "
            f"target_residency_s={self.target_residency_s}, "
            f"transition_time_s={self.transition_time_s}, "
            f"power_w={self.power_w})"
        )

class CStates(Enum):
    C0 = CState('C0', 0.0, 0.0, 4.0)
    C1 = CState('C1', 2e-6, 2e-6, 1.44)
    C6 = CState('C6', 0.0006, 0.000133, 0.1)

def get_c_states():
    """Server CPU C-states.
    To model idle states of server CPUs, we create a model based on specification values provided for Intel server CPUs.
    [1] J. H. Yahya et al., "AgileWatts: An Energy-Efficient CPU Core Idle-State Architecture for Latency-Sensitive
    Server Applications," 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO), Chicago, IL, USA,
    2022, pp. 835-850, doi: 10.1109/MICRO56248.2022.00063. keywords: {Degradation;Program processors;Microarchitecture;
    Coherence;Market research;Energy efficiency;Generators;Energy Efficiency;power management;Latency Sensitive applications},
    """
    return [
        {
            "state": "C0",
            "transition_time_s": 0.0,
            "target_residency_s": 0.0,
            "power_w": 4.0
        },
        {
            "state": "C1",
            "transition_time_s": 2e-6,
            "target_residency_s": 2e-6,
            "power_w": 1.44
        },
        {
            "state": "C6",
            "transition_time_s": 0.000133,
            "target_residency_s": 0.0006,
            "power_w": 0.1
        },
    ]

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
    if model is not None:
        # Use observed data of core serving a model
        return get_core_power_for_model(model_name=model)

    # Use c-state for idle state power consumption
    return c_state.power_w


def calculate_c_state(last_8_idle_durations_s=None):
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
    predicted_idle_duration = float('inf')
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
    latency_limit_of_power_mgt_qos = float(
        'inf')  # we assume the worst, where power management quality of service do not apply a latency limit
    if number_of_tasks_waiting_on_io > 0:
        latency_limit = latency_limit / number_of_tasks_waiting_on_io
    latency_limit = min(latency_limit, latency_limit_of_power_mgt_qos)

    c_states = [state.value for state in CStates]
    chosen_c_state = list(filter(lambda x: x.state == "C0", c_states))[0]  # default to C0
    for c_state in c_states:
        target_residency = c_state.target_residency_s
        transition_time = c_state.transition_time_s
        if transition_time > latency_limit:
            continue
        if target_residency >= predicted_idle_duration:
            gap = target_residency - predicted_idle_duration
            current_gap = chosen_c_state.target_residency_s - predicted_idle_duration
            if gap < current_gap:
                chosen_c_state = c_state

    return chosen_c_state
