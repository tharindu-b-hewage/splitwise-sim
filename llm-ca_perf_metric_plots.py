import ast
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

CODE_PREFIX = "rr_code_"

CONV_PREFIX = "rr_conv_"

IDENTITY_MAP = {
    'linux': {
        'color': '#E9002D',
        'marker': 'o'
    },
    'zhao23': {
        'color': '#FFAA0O',
        'marker': 'v'
    },
    'proposed': {
        'color': '#008000',
        'marker': 's'
    },
}


def list_dirs(root):
    return [folder for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))]


def list_files(root, prefix=None):
    return [file for file in os.listdir(root) if
            not os.path.isdir(os.path.join(root, file)) and file.startswith(prefix)]


def process_machine(m_df):
    # group by column 'id' and traverse each group
    core_health_dst = []
    for name, core_data in m_df.groupby('id'):
        core_data = core_data.sort_values(by='clock')
        # get the last row
        last_row = core_data.iloc[-1]
        core_health_after = last_row['health']
        core_health_dst.append(core_health_after)

    # calculate the coefficient of variation
    m_core_health_cv = np.std(core_health_dst) / np.mean(core_health_dst)
    return m_core_health_cv


def process_cpu_usage_files(cpu_data_loc, m_cpu_usage):
    cls_m_core_health_cv_dst = []
    for machine in m_cpu_usage:
        m_df = pd.read_csv(os.path.join(cpu_data_loc, machine))
        m_core_health_cv = process_machine(m_df)
        cls_m_core_health_cv_dst.append(m_core_health_cv)
    cls_m_core_health_cv_p99 = np.percentile(cls_m_core_health_cv_dst, 99)
    cls_m_core_health_cv_p90 = np.percentile(cls_m_core_health_cv_dst, 90)
    cls_m_core_health_cv_p50 = np.percentile(cls_m_core_health_cv_dst, 50)
    return {
        "cls_m_core_health_cv_p99": cls_m_core_health_cv_p99,
        "cls_m_core_health_cv_p90": cls_m_core_health_cv_p90,
        "cls_m_core_health_cv_p50": cls_m_core_health_cv_p50
    }


def process_task_data_files(cpu_data_loc, m_task_log):
    tot_nrm_diffs = pd.DataFrame(columns=['nrm_diff'])
    for machine in m_task_log:
        m_df = pd.read_csv(os.path.join(cpu_data_loc, machine))
        m_df = pd.DataFrame(ast.literal_eval(m_df["tasks_count"].loc[0]),
                            columns=['clock', 'running_tasks', 'gpu_mem_util', 'awaken_cores'])
        m_df['nrm_diff'] = (m_df['awaken_cores'] - m_df[
            'running_tasks']) / 112  # 112 = total cores of the servers in the cluster

        # expand tot_nrm_diffs
        tot_nrm_diffs = pd.concat([tot_nrm_diffs, m_df[['nrm_diff']]], ignore_index=True)
    return tot_nrm_diffs


def process_exps(root, exps, prefix, technique):
    parsed_cpu_health_data = []
    parsed_nrm_core_to_task_diff_dst = pd.DataFrame(columns=['nrm_diff', 'technique', 'rate'])
    for exp in exps:
        print(f"Processing {exp}")
        rq_rate = exp.split(prefix)[1]
        cpu_data_loc = os.path.join(root, exp, "0_22", "bloom-176b", "mixed_pool", "cpu_usage")
        m_cpu_usage = list_files(root=cpu_data_loc, prefix="cpu_usage_")
        # m_slp_mgt = list_files(root=cpu_data_loc, prefix="slp_mgt_")
        m_task_log = list_files(root=cpu_data_loc, prefix="task_log_")

        cpu_data = process_cpu_usage_files(cpu_data_loc, m_cpu_usage)
        # cpu_data["trace"] = trace
        cpu_data["technique"] = technique
        cpu_data["rate"] = rq_rate
        parsed_cpu_health_data.append(cpu_data)

        cls_nrm_core_to_task_diff_dst = process_task_data_files(cpu_data_loc, m_task_log)
        cls_nrm_core_to_task_diff_dst['technique'] = technique
        cls_nrm_core_to_task_diff_dst['rate'] = rq_rate
        parsed_nrm_core_to_task_diff_dst = pd.concat([parsed_nrm_core_to_task_diff_dst, cls_nrm_core_to_task_diff_dst],
                                                     ignore_index=True)

    return parsed_cpu_health_data, parsed_nrm_core_to_task_diff_dst


def plot_core_task_diff_data(df):
    rates = df["rate"].unique()
    n_rates = len(rates)
    techniques = df["technique"].unique()

    # Create subplots with one plot per rate
    # fig, axes = plt.subplots(nrows=1, ncols=n_rates, figsize=(5 * n_rates, 4), sharey=True)
    fig, axes = plt.subplots(nrows=1, ncols=len(techniques), figsize=(5 * len(techniques), 4), sharey=True)

    if n_rates == 1:
        axes = [axes]

    # for i, rate in enumerate(rates):
    for i, tech in enumerate(techniques):

        ax = axes[i]
        tech_data = df[df["technique"] == tech]

        for rate in tech_data["rate"].unique():
            rate_data = tech_data[tech_data["rate"] == rate]
            sorted_nrm_diff = sorted(rate_data["nrm_diff"])
            cumsum = np.cumsum(np.ones_like(sorted_nrm_diff)) / len(sorted_nrm_diff)
            for x, c_sum in enumerate(cumsum):
                if c_sum > 0.5:
                    x_thr = x
                    y_thr = c_sum
                    break
            ax.plot(
                sorted_nrm_diff,
                cumsum,
                label=tech + '-' + str(rate),
                color=IDENTITY_MAP[technique]['color'],
            )

            # Mark the point where y exceeds the threshold
            plt.scatter(x_thr, y_thr, color='red', zorder=5, label=f'Exceeds Threshold ({x_thr}, {y_thr})')

            # Add vertical and horizontal lines
            plt.axhline(y=x_thr, color='gray', linestyle='--', label='Threshold')
            plt.axvline(x=y_thr, color='red', linestyle='--', label=f'x={x_thr}')

            # Annotate the point
            plt.annotate(f'({x_thr}, {y_thr})',
                         xy=(x_thr, y_thr),
                         xytext=(x_thr + 0.5, y_thr + 1),
                         arrowprops=dict(facecolor='black', arrowstyle='->'),
                         fontsize=10)

        # ax = axes[i]
        # rate_data = df[df["rate"] == rate]
        #
        # for technique in rate_data["technique"].unique():
        #     tech_data = rate_data[rate_data["technique"] == technique]
        #     sorted_nrm_diff = sorted(tech_data["nrm_diff"])
        #     cumsum = np.cumsum(np.ones_like(sorted_nrm_diff)) / len(sorted_nrm_diff)
        #     ax.plot(
        #         sorted_nrm_diff, cumsum,
        #         label=technique,
        #         color=IDENTITY_MAP[technique]['color'],
        #     )

        ax.set_title(f"Rate: {tech}")
        ax.set_xlim(-1, 1)
        ax.set_xlabel(
            "Normalized Core Availability\n(negative == cores oversubscribed, positive == cores under-subscribed)")
        if i == 0:
            ax.set_ylabel("Cumulative Distribution")
        ax.legend()
        ax.grid(True)

    # Adjust layout
    fig.suptitle("Core Availability for Task Execution")
    plt.tight_layout()
    plt.savefig("temp_results/core_availability_for_task_execution.svg")


def plot_core_health_cv(df):
    # Extract unique traces and metric types
    # unique_traces = df["trace"].unique()
    metrics = ["cls_m_core_health_cv_p99", "cls_m_core_health_cv_p90", "cls_m_core_health_cv_p50"]
    metrics_lbl = ["p99", "p90", "p50"]

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(10, 4))

    for i, _ in enumerate([1]):
        for j, metric in enumerate(metrics):
            ax = axes[j]

            # Plot for each technique
            nrm_vals = []
            for technique in df["technique"].unique():
                max_metric = df[metric].max()
                tech_data = df[df["technique"] == technique]
                nrm_val = tech_data[metric] / max_metric
                nrm_vals.append(nrm_val)
                ax.plot(
                    tech_data["rate"],
                    nrm_val,
                    marker=IDENTITY_MAP[technique]['marker'],
                    label=technique,
                    color=IDENTITY_MAP[technique]['color']
                )

            ax.set_xlabel("Request Rate (req/s)")
            ax.set_ylabel('Normalized CV of \nCore Frequency Dist.' + metrics_lbl[j])
            # ax.set_yscale("logit")
            ax.legend()
            ax.grid(True)

    # Adjust layout
    fig.suptitle("Managing NBTI- and PV-Induced Uneven Frequency Distribution of Machine Cores in the Cluster")
    plt.tight_layout()
    plt.savefig("temp_results/cpu_aging_impact.svg")


ROOT_LOC = "/Users/tharindu/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/phd-student/projects/dynamic-affinity/experiments"
# ROOT_LOC = "/Users/tharindu/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/phd-student/projects/dynamic-affinity/bk/2024-12-15_02-52-14"
"""At root experiments folder, create sub folder for each technique. Copy each 'rr_{code or conv}_{rate' folders to the relevant technique folder."""

# dev: avoid processing data when fixing the plots
dev_is_plot_fix = True

techniques = list_dirs(root=ROOT_LOC)
tot_parsed_health_data_conv = []
health_data_df = None
tot_parsed_core_task_diff_data = pd.DataFrame(columns=['nrm_diff', 'technique', 'rate'])

if not os.path.isfile('health_data_df.csv'):
    dev_is_plot_fix = False

if not dev_is_plot_fix:
    for technique in techniques:
        print(f"Processing technique: {technique}")
        curr_loc = os.path.join(ROOT_LOC, technique)
        traces = list_dirs(root=curr_loc)
        conv_traces = [trace for trace in traces if CONV_PREFIX in trace]
        code_traces = [trace for trace in traces if CODE_PREFIX in trace]

        parsed_health_data, parsed_core_task_diff_data = process_exps(root=curr_loc, exps=conv_traces,
                                                                      prefix=CONV_PREFIX,
                                                                      technique=technique)

        tot_parsed_health_data_conv.extend(parsed_health_data)
        tot_parsed_core_task_diff_data = pd.concat([tot_parsed_core_task_diff_data, parsed_core_task_diff_data],
                                                   ignore_index=True)

    health_data_df = pd.DataFrame(tot_parsed_health_data_conv)
    health_data_df["rate"] = health_data_df["rate"].astype(int)
    health_data_df = health_data_df.sort_values(by=['rate'])

    tot_parsed_core_task_diff_data["rate"] = tot_parsed_core_task_diff_data["rate"].astype(int)
    tot_parsed_core_task_diff_data = tot_parsed_core_task_diff_data.sort_values(by=['rate'])


    print('saving data to cache...')
    health_data_df.to_csv('health_data_df.csv', index=False)
    tot_parsed_core_task_diff_data.to_csv('tot_parsed_core_task_diff_data.csv', index=False)
else:
    print("Loading data from cache...")


    def dev_load_data_cache(cache_file_name):
        if os.path.exists(cache_file_name):
            return pd.read_csv(cache_file_name)
        else:
            return None


    health_data_df = dev_load_data_cache('health_data_df.csv')
    tot_parsed_core_task_diff_data = dev_load_data_cache('tot_parsed_core_task_diff_data.csv')

plot_core_health_cv(df=health_data_df)
plot_core_task_diff_data(df=tot_parsed_core_task_diff_data)
