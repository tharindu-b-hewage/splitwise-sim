import random

from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

BW_ADJUST = 0.1


def remove_overlap(df):
    """We study isolated inference tasks. For example, this approach allows calculating CPU usage per-task, given all idle cores, which core to pick, etc."""
    # Compute the end times
    df['End'] = df['Timestamp'] + df['Runtime (s)']

    # Sort the DataFrame by the start times
    df_sorted = df.sort_values('Timestamp').reset_index(drop=True)

    # Initialize variables for detecting overlaps
    overlapping_indices = set()
    active_intervals = []

    for idx, row in df_sorted.iterrows():
        current_start = row['Timestamp']
        current_end = row['End']
        current_index = idx

        # Remove intervals that have ended
        active_intervals = [interval for interval in active_intervals if interval['End'] > current_start]

        # Check for overlaps with active intervals
        for interval in active_intervals:
            if interval['End'] > current_start:
                overlapping_indices.add(current_index)
                overlapping_indices.add(interval['Index'])

        # Add the current interval to the active list
        active_intervals.append({'Index': current_index, 'End': current_end})

    # Remove overlapping rows
    df_cleaned = df_sorted.drop(list(overlapping_indices)).reset_index(drop=True)

    return df_cleaned


def get_formatted_data(df):
    df = remove_overlap(df)
    df = df[df['Phase'].str.contains('start-inference')]
    ret_df = pd.DataFrame()
    ret_df['time'] = df['Timestamp']
    ret_df['token_in'] = df['Number of Input Tokens']
    ret_df['token_out'] = df['Output Token Limit']
    ret_df['model'] = df['Model']
    ret_df['runtime'] = df['Runtime (s)']
    ret_df['core'] = df['CPU Core']
    ret_df['gpus'] = df['Number of GPUs']
    if 'GPU Energy (J)' in df.columns:
        ret_df['gpu_power'] = df['GPU Energy (J)'] / df['Runtime (s)']
    elif 'GPU Energy (mJ)' in df.columns:
        ret_df['gpu_power'] = df['GPU Energy (mJ)'] / df['Runtime (s)']
    if 'CPU Energy (J)' in df.columns:
        ret_df['core_power'] = df['CPU Energy (J)'] / df['Runtime (s)']
    return ret_df


def core_id_sampler(df, bw_adjust=0.1):
    # Fit KDE to the core data
    core_data = df['core'].values
    kde = gaussian_kde(core_data, bw_method=bw_adjust)

    # Generator function to sample Core ID values
    while True:
        yield int(np.round(kde.resample(1)[0]))


def task_schedule_linux(cpu_cores):
    """Implements core assignment behavior observed in the energy inference project [1].

    This function collects telemetry data from inference tasks [1] to observe CPU core residency. Based on the typical
    operating system state of an inference server, it creates a probabilistic model to replicate core assignment behavior.

    platform: https://www.lcrc.anl.gov/systems/swing

    [1] https://github.com/grantwilkins/energy-inference.git
    """
    free_core_ids = [core.id for core in cpu_cores if core.task is None]

    max_retries = len(free_core_ids)
    core_id = None
    retries = 0
    while core_id is None:
        core_id = get_core_id_of_argane_swing(num_cores=len(cpu_cores))
        if core_id not in free_core_ids:
            core_id = None
            retries += 1
        if retries >= max_retries:
            if len(free_core_ids) > 0:
                core_id = random.choice(free_core_ids)
            else:
                return None

    return list(filter(lambda core: core.id == core_id, cpu_cores))[0]


def get_core_id_of_argane_swing(num_cores):
    TOTAL_CORES_OF_THE_SAMPLER_MODEL = 256
    id = next(sampler)
    # scale the core id
    ratio = (num_cores / TOTAL_CORES_OF_THE_SAMPLER_MODEL)
    scaled_id = id * ratio
    return int(scaled_id)


df_core_residency = get_formatted_data(pd.read_csv('data/infer-amd-swing-llama270b.csv'))
sampler = core_id_sampler(df_core_residency, bw_adjust=BW_ADJUST)
