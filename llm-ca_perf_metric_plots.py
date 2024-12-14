import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

CODE_PREFIX = "rr_code_"

CONV_PREFIX = "rr_conv_"


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


def process_exps(root, exps, prefix, technique):
    parsed_data = []
    for exp in exps:
        print(f"Processing {exp}")
        rq_rate = exp.split(prefix)[1]
        cpu_data_loc = os.path.join(root, exp, "0_22", "bloom-176b", "mixed_pool", "cpu_usage")
        m_cpu_usage = list_files(root=cpu_data_loc, prefix="cpu_usage_")
        # m_slp_mgt = list_files(cpu_data_loc=cpu_data_loc, prefix="slp_mgt_")
        # m_task_log = list_files(cpu_data_loc=cpu_data_loc, prefix="task_log_")

        cpu_data = process_cpu_usage_files(cpu_data_loc, m_cpu_usage)
        #cpu_data["trace"] = trace
        cpu_data["technique"] = technique
        cpu_data["rate"] = rq_rate

        parsed_data.append(cpu_data)

    return parsed_data


def plot_core_health_cv(df):
    # Extract unique traces and metric types
    #unique_traces = df["trace"].unique()
    metrics = ["cls_m_core_health_cv_p99", "cls_m_core_health_cv_p90", "cls_m_core_health_cv_p50"]
    metrics_lbl = ["Core Health CV p99", "Core Health CV p90", "Core Health CV p50"]

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(10, 4))

    for i, _ in enumerate([1]):
        for j, metric in enumerate(metrics):
            ax = axes[j]

            # Plot for each technique
            for technique in df["technique"].unique():
                tech_data = df[df["technique"] == technique]
                ax.plot(tech_data["rate"], tech_data[metric], marker='o', label=technique)

            #ax.set_title(f"Trace: {trace} - {metric}")
            ax.set_xlabel("Request Rate (req/s)")
            ax.set_ylabel(metrics_lbl[j])
            ax.legend()
            ax.grid(True)

    # Adjust layout
    fig.suptitle("Distribution of Core Health Coefficient of Variation (CV) Across Cluster Machines")
    plt.tight_layout()
    plt.show()


ROOT_LOC = "/Users/tharindu/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/phd-student/projects/dynamic-affinity/experiments"
"""At root experiments folder, create sub folder for each technique. Copy each 'rr_{code or conv}_{rate' folders to the relevant technique folder."""

techniques = list_dirs(root=ROOT_LOC)
tot_parsed_data_conv = []
tot_parsed_data_code = []
for technique in techniques:
    print(f"Processing technique: {technique}")
    curr_loc = os.path.join(ROOT_LOC, technique)
    traces = list_dirs(root=curr_loc)
    conv_traces = [trace for trace in traces if CONV_PREFIX in trace]
    code_traces = [trace for trace in traces if CODE_PREFIX in trace]

    parsed_data_conv = process_exps(root=curr_loc, exps=conv_traces, prefix=CONV_PREFIX,
                                    technique=technique)
    parsed_data_code = process_exps(root=curr_loc, exps=code_traces, prefix=CONV_PREFIX,
                                    technique=technique)

    tot_parsed_data_conv.extend(parsed_data_conv)
    tot_parsed_data_code.extend(parsed_data_code)

# dict array to df
data_df_conv = pd.DataFrame(tot_parsed_data_conv)
plot_core_health_cv(df=data_df_conv)

# cpu_data_loc = "splitwise_5_17/rr_conv_30/0_22/bloom-176b/mixed_pool/cpu_usage"
