import os
import sys

from pandas.errors import EmptyDataError

from core_power import CStates, get_c_states, machine_specs, calculate_WTTF
import re
import numpy as np
import pandas as pd

def get_cpu_metrics(data_path, sim_end_time):
    cls_energy = 0.0
    cls_wttf_cvs = []
    cls_core_health_cvs = []
    cls_max_task_throughputs = []
    for m_file_name in os.listdir(data_path):
        if 'cpu_usage' not in m_file_name:
            #print(f"Skipping {m_file_name} since not a recognized cpu usage file")
            continue
        segg_name = m_file_name.split('cpu_usage_')[1].replace('.csv', '').split('_')
        cpu_n = segg_name[0]
        m_id = int(segg_name[1])
        #print(f"Processing {cpu_n} {m_id}")

        try:
            m = pd.read_csv(os.path.join(data_path, m_file_name))
        except EmptyDataError:
            print(f"Skipping {m_file_name} since it's empty...")
            continue

        c_states = get_c_states(cpu_model=cpu_n)
        c_count = machine_specs[cpu_n]['cores']

        m_energy = 0.0
        c_wttfs = []
        core_healths = []
        next_row = None
        for c_id in range(c_count):
            c_data = m[m['id'] == c_id].sort_values('clock', ascending=True)
            clk = 0.0
            c_state = 'C1'
            freq = 2.25
            c_wttf = 0.0
            c_energy = 0.0
            for idx, row in c_data.iterrows():

                next_clk = row['clock']
                next_c_state = row['c_state']
                next_freq = row['freq_hz']

                # update for the time period
                energy, wttf = calculate_metrics(c_state, c_states, clk, cpu_n, next_clk, freq)
                c_energy += energy
                c_wttf += wttf

                # move to next time period
                clk = next_clk
                c_state = next_c_state
                next_row = row
                freq = next_freq

            # update at the sim completion. we need this to account for time from last state to sim end
            next_clk = sim_end_time
            energy, wttf = calculate_metrics(c_state, c_states, clk, cpu_n, next_clk, freq)
            c_energy += energy
            c_wttf += wttf

            # calculate the core health metric
            core_health_at_end = next_row['health']
            core_healths.append(core_health_at_end)

            # update per core metric collections
            c_wttfs.append(c_wttf)
            m_energy += c_energy

        # update metric collection
        # calculate coefficient of variation of wttf
        wttf_mean = np.mean(c_wttfs)
        if wttf_mean > 0: # if cores were used at any point
            cls_wttf_cvs.append(np.std(c_wttfs) / np.mean(c_wttfs))
        cls_core_health_cvs.append(np.std(core_healths) / np.mean(core_healths))
        cls_energy += m_energy

        # parallel cpu tasks calculation
        t_sorted = m[['clock','task']].sort_values('clock', ascending=True)
        t_sorted['running_tasks'] = t_sorted['task'].apply(lambda x: 1 if pd.notna(x) and x != '' else -1).cumsum()
        cls_max_task_throughputs.append(t_sorted['running_tasks'].max())



    return np.percentile(cls_core_health_cvs, 90), np.percentile(cls_wttf_cvs, 90), cls_energy, np.percentile(cls_max_task_throughputs, 90), np.max(
        cls_max_task_throughputs)


def calculate_metrics(c_state, c_states, clk, cpu_n, next_clk, freq):
    t_delta = next_clk - clk

    energy = c_states[c_state]['core_power_w'] * t_delta
    wttf = calculate_WTTF(cpu_model=cpu_n, c_state=c_state, time_s=t_delta, freq=freq)

    return energy, wttf


def get_parsed_data(experiment_root):
    data = []
    for technique in os.listdir(experiment_root):
        if 'tq_' not in technique:
            #print(f"Skipping {technique} since not a recognized technique")
            continue
        technique_path = os.path.join(experiment_root, technique)
        for exp in os.listdir(technique_path):
            if 'rr_' not in exp:
                #print(f"Skipping {exp} since not a recognized experiment")
                continue
            print('Processing', exp)
            pattern = r"rr_(\w+)_(\w+)"
            match = re.match(pattern, exp)
            if match:
                kind = match.group(1)
                rate = match.group(2)
                exp_data_root = os.path.join(experiment_root, technique, exp, '26_25', 'bloom-176b', 'mixed_pool')
                sim_time = pd.read_csv(os.path.join(exp_data_root, 'simulator.csv'))['time'][0]

                app_df = pd.read_csv(os.path.join(exp_data_root, 'summary.csv'))[
                    ['prompt_sizes_p90', 'token_sizes_p90', 'ttft_times_p90', 'tbt_times_p90', 'response_times_p90',
                     'queue_times_p90']]
                p90_cv_server_cpu_health, p90_cv_server_cpu_wttf, energy_cluster_cpu, p90_server_cpu_max_parallel_cpu_tasks, max_server_cpu_max_parallel_cpu_tasks = get_cpu_metrics(
                    os.path.join(exp_data_root, 'cpu_usage'), sim_end_time=sim_time)

                data.append({
                    'kind': kind,
                    'rate': rate,
                    'technique': technique,
                    'prompt_sizes_p90': app_df['prompt_sizes_p90'].mean(),
                    'token_sizes_p90': app_df['token_sizes_p90'].mean(),
                    'ttft_times_p90': app_df['ttft_times_p90'].mean(),
                    'tbt_times_p90': app_df['tbt_times_p90'].mean(),
                    'response_times_p90': app_df['response_times_p90'].mean(),
                    'queue_times_p90': app_df['queue_times_p90'].mean(),
                    # there is only one row in the app_df. So, mean is the value itself
                    'core_wttf_cv_p90': p90_cv_server_cpu_wttf,
                    'core_health_cv_p90': p90_cv_server_cpu_health,
                    'cluster_energy_j': energy_cluster_cpu,
                    'cpu_tasks_throughput_p90': p90_server_cpu_max_parallel_cpu_tasks,
                    'cpu_tasks_throughput_max': max_server_cpu_max_parallel_cpu_tasks
                })

    # convert data to a pandas dataframe
    df = pd.DataFrame(data)
    return df


#df = get_parsed_data(experiment_root="/Users/tharindu/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/phd-student/projects/dynamic-affinity/simulations")
df = get_parsed_data(experiment_root="/Users/tharindu/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/phd-student/projects/dynamic-affinity/debug")

data_file = './notebooks/helper/parsed_data.csv'
# delete file if exists
if os.path.exists(data_file):
    os.remove(data_file)
df.to_csv(data_file, index=False)