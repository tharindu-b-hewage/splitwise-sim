import ast
import os
import re
from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.size"] = 12

CODE_PREFIX = "rr_code_"

CONV_PREFIX = "rr_conv_"

IDENTITY_MAP = {
    'linux': {
        'color': '#E9002D',
        'marker': 'o'
    },
    'least-aged': {
        'color': '#FFAA00',
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
    core_health_dst_after = []
    core_health_dst_before = []
    est_core_fq_before = 1.0
    est_core_fq_after = 1.0
    for name, core_data in m_df.groupby('id'):
        core_data = core_data.sort_values(by='clock')
        # get the last row
        first_row = core_data.iloc[0]
        last_row = core_data.iloc[-1]
        core_fq_before = first_row['health']  # health is frequency normalized.
        core_fq_after = last_row['health']  # health is frequency normalized.

        core_health_dst_after.append(core_fq_after)
        core_health_dst_before.append(core_fq_before)

        if core_fq_before < est_core_fq_before:
            est_core_fq_before = core_fq_before

        if core_fq_after < est_core_fq_after:
            est_core_fq_after = core_fq_after

    # calculate the coefficient of variation
    m_core_health_cv = np.std(core_health_dst_after) / np.mean(core_health_dst_after)

    est_core_fq_drop = np.mean(core_health_dst_before) - np.mean(core_health_dst_after)

    return m_core_health_cv, est_core_fq_after, est_core_fq_before, est_core_fq_drop


def process_cpu_usage_files(cpu_data_loc, m_cpu_usage):
    cls_m_core_health_cv_dst = []
    cls_m_core_worst_fq_dst_before = []
    cls_m_core_worst_fq_dst_after = []
    cls_m_est_core_fq_drops = []
    for machine in m_cpu_usage:
        m_df = pd.read_csv(os.path.join(cpu_data_loc, machine))
        m_core_health_cv, worst_core_fq_after, worst_core_fq_before, est_core_fq_drop = process_machine(m_df)
        cls_m_core_health_cv_dst.append(m_core_health_cv)
        cls_m_core_worst_fq_dst_after.append(worst_core_fq_after)
        cls_m_core_worst_fq_dst_before.append(worst_core_fq_before)
        cls_m_est_core_fq_drops.append(est_core_fq_drop)

    cls_m_core_health_cv_p99 = np.percentile(cls_m_core_health_cv_dst, 99)
    cls_m_core_health_cv_p90 = np.percentile(cls_m_core_health_cv_dst, 90)
    cls_m_core_health_cv_p50 = np.percentile(cls_m_core_health_cv_dst, 50)

    cls_m_core_worst_fq_p99_after = np.percentile(cls_m_core_worst_fq_dst_after, 99)
    cls_m_core_worst_fq_p90_after = np.percentile(cls_m_core_worst_fq_dst_after, 90)
    cls_m_core_worst_fq_p50_after = np.percentile(cls_m_core_worst_fq_dst_after, 50)

    cls_m_core_worst_fq_p99_before = np.percentile(cls_m_core_worst_fq_dst_before, 99)
    cls_m_core_worst_fq_p90_before = np.percentile(cls_m_core_worst_fq_dst_before, 90)
    cls_m_core_worst_fq_p50_before = np.percentile(cls_m_core_worst_fq_dst_before, 50)

    cls_m_est_core_fq_drops_p99 = np.percentile(cls_m_est_core_fq_drops, 99)
    cls_m_est_core_fq_drops_p90 = np.percentile(cls_m_est_core_fq_drops, 90)
    cls_m_est_core_fq_drops_p50 = np.percentile(cls_m_est_core_fq_drops, 50)

    return {
        "cls_m_core_health_cv_p99": cls_m_core_health_cv_p99,
        "cls_m_core_health_cv_p90": cls_m_core_health_cv_p90,
        "cls_m_core_health_cv_p50": cls_m_core_health_cv_p50,
        "cls_m_core_worst_fq_p99_after": cls_m_core_worst_fq_p99_after,
        "cls_m_core_worst_fq_p90_after": cls_m_core_worst_fq_p90_after,
        "cls_m_core_worst_fq_p50_after": cls_m_core_worst_fq_p50_after,
        "cls_m_core_worst_fq_p99_before": cls_m_core_worst_fq_p99_before,
        "cls_m_core_worst_fq_p90_before": cls_m_core_worst_fq_p90_before,
        "cls_m_core_worst_fq_p50_before": cls_m_core_worst_fq_p50_before,
        "cls_m_est_core_fq_drops_p99": cls_m_est_core_fq_drops_p99,
        "cls_m_est_core_fq_drops_p90": cls_m_est_core_fq_drops_p90,
        "cls_m_est_core_fq_drops_p50": cls_m_est_core_fq_drops_p50,
    }


def process_task_data_files(cpu_data_loc, m_task_log, cores):
    tot_nrm_diffs = pd.DataFrame(columns=['nrm_diff'])
    for machine in m_task_log:
        m_df = pd.read_csv(os.path.join(cpu_data_loc, machine))
        m_df = pd.DataFrame(ast.literal_eval(m_df["tasks_count"].loc[0]),
                            columns=['clock', 'running_tasks', 'gpu_mem_util', 'awaken_cores'])
        m_df['nrm_diff'] = (m_df['awaken_cores'] - m_df[
            'running_tasks']) / cores  # 112 = total cores of the servers in the cluster

        # expand tot_nrm_diffs
        tot_nrm_diffs = pd.concat([tot_nrm_diffs, m_df[['nrm_diff']]], ignore_index=True)
    return tot_nrm_diffs


def process_exps(root, exps, prefix, technique, cores):
    parsed_cpu_health_data = []
    parsed_nrm_core_to_task_diff_dst = pd.DataFrame(columns=['nrm_diff', 'technique', 'rate', 'cores'])
    for exp in exps:
        print(f"Processing {exp}")
        rq_rate = exp.split(prefix)[1]
        cpu_data_loc = os.path.join(root, exp, "0_22", "bloom-176b", "mixed_pool", "cpu_usage")
        m_cpu_usage = list_files(root=cpu_data_loc, prefix="cpu_usage_")
        # m_slp_mgt = list_files(root=cpu_data_loc, prefix="slp_mgt_")
        m_task_log = list_files(root=cpu_data_loc, prefix="task_log_")

        cpu_data = process_cpu_usage_files(cpu_data_loc, m_cpu_usage)
        # cpu_data["trace"] = trace
        cpu_data["cores"] = cores
        cpu_data["technique"] = technique
        cpu_data["rate"] = rq_rate
        parsed_cpu_health_data.append(cpu_data)

        cls_nrm_core_to_task_diff_dst = process_task_data_files(cpu_data_loc, m_task_log, cores)
        cls_nrm_core_to_task_diff_dst["cores"] = cores
        cls_nrm_core_to_task_diff_dst['technique'] = technique
        cls_nrm_core_to_task_diff_dst['rate'] = rq_rate
        parsed_nrm_core_to_task_diff_dst = pd.concat([parsed_nrm_core_to_task_diff_dst, cls_nrm_core_to_task_diff_dst],
                                                     ignore_index=True)

    return parsed_cpu_health_data, parsed_nrm_core_to_task_diff_dst


def plot_core_task_diff_data(df):
    rates_colors = {
        '40': '#4053d3',
        '60': '#ddb310',
        '80': '#b51d14',
        '100': '#00beff',
        '230': '#fb49b0',
        '250': '#00b25d',
    }

    vm_cores = [40, 80, 112]
    for cores in vm_cores:
        filt_df = df[df["cores"] == cores]
        rates = filt_df["rate"].unique()
        n_rates = len(rates)
        techniques = ['linux', 'least-aged', 'proposed']

        # Create subplots with one plot per rate
        fig, axes = plt.subplots(nrows=1, ncols=len(techniques), figsize=(4 * len(techniques), 2.3), sharey=True,
                                 sharex=True)

        # if n_rates == 1:
        #     axes = [axes]

        for i, tech in enumerate(techniques):

            ax = axes[i]
            tech_data = filt_df[filt_df["technique"] == tech]

            p90_vals = []
            p1_vals = []
            ax.grid(True, zorder=0)
            for rate in tech_data["rate"].unique():
                rate_data = tech_data[tech_data["rate"] == rate]
                sorted_nrm_diff = sorted(rate_data["nrm_diff"])
                cumsum = np.cumsum(np.ones_like(sorted_nrm_diff)) / len(sorted_nrm_diff)
                p90_val = np.percentile(sorted_nrm_diff, 90)
                p1_val = np.percentile(sorted_nrm_diff, 1)
                p90_vals.append(p90_val)
                p1_vals.append(p1_val)

                ax.plot(
                    sorted_nrm_diff,
                    cumsum,
                    # label=tech + '@' + str(rate) + 'req/s',
                    label=str(rate) + 'req/s',
                    color=rates_colors[str(rate)],
                )

            plot_p90_val = round(max(p90_vals), 3)
            plot_p1_val = round(min(p1_vals), 3)
            ax.vlines(x=plot_p90_val, ymin=0.0, ymax=1.0, linewidth=0.7, linestyles='dashed', color='black',
                      label=f'p90 = {plot_p90_val}')
            ax.vlines(x=plot_p1_val, ymin=0.0, ymax=1.0, linewidth=0.7, linestyles='dashed', color='blue',
                      label=f'p1 = {plot_p1_val}')

            ax.set_title(f"{tech}")

            if tech != "proposed":
                ax.set_xlim([plot_p1_val, 1.0])
            else:
                ax.set_xlim([plot_p1_val, 1.0])

            ax.set_xlabel("Normalized Idle CPU Cores")
            if i == 0:
                ax.set_ylabel("Cumulative\n Measurements")
            handles, labels = ax.get_legend_handles_labels()
            show_items = ['p90', 'p1']
            filtered_handles = [h for h, l in zip(handles, labels) if any(sub in l for sub in show_items)]
            filtered_labels = [l for l in labels if any(sub in l for sub in show_items)]
            ax.legend(filtered_handles, filtered_labels)

        # Adjust layout
        # fig.suptitle("Idle CPU Cores Across the Cluster Machines")
        handles, labels = ax.get_legend_handles_labels()
        show_items = ['40req/s', '60req/s', '80req/s', '100req/s']
        filtered_handles = [h for h, l in zip(handles, labels) if l in show_items]
        filtered_labels = [l for l in labels if l in show_items]
        fig.legend(filtered_handles, filtered_labels, bbox_to_anchor=(1.105, 0.75))
        fig.tight_layout()
        # plt.savefig("temp_results/aging/vm-cores_" + str(cores) + "_" + filename, bbox_inches='tight')
        # plt.tight_layout()
        plt.savefig(
            "temp_results/core-utilization/vm-cores_" + str(cores) + "_core_availability_for_task_execution.svg",
            bbox_inches='tight')


def plot_core_health_cv(df):
    # Extract unique traces and metric types
    # unique_traces = df["trace"].unique()
    metrics = ["cls_m_core_health_cv_p99", "cls_m_core_health_cv_p90", "cls_m_core_health_cv_p50",
               "cls_m_est_core_fq_drops_p99", "cls_m_est_core_fq_drops_p90", "cls_m_est_core_fq_drops_p50",
               "cls_m_core_worst_fq_p99_after", "cls_m_core_worst_fq_p90_after", "cls_m_core_worst_fq_p50_after",
               "cls_m_core_worst_fq_p99_before", "cls_m_core_worst_fq_p90_before", "cls_m_core_worst_fq_p50_before"
               ]
    metrics_lbl = ["p99", "p90", "p50", "p99", "p90", "p50"]
    vm_cores = [40, 80, 112]

    def plot_row_data(df, tech_used, metrics, metrics_lbl, filename, cores, is_carbon_bars):
        flt_df = df[df["cores"] == cores]
        # Create subplots
        if not is_carbon_bars:
            # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(4.5 * 3, 3 * 2), sharex=True, sharey='row')
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(2.5 * 3, 2.2 * 2), sharex=True, sharey='row')
        else:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(4.0 * 3, 3 * 0.82), sharex=True, sharey=True)
        for j, metric in enumerate(metrics):
            if not is_carbon_bars:
                row_id = j // 3
                ax = axes[row_id][j % 3]
                ax.grid(True, zorder=0)
                nrm_val = flt_df[metric].max()
                max_val = flt_df[metric].max()
                min_val = flt_df[metric].min()
                for technique in tech_used:
                    tech_data = flt_df[flt_df["technique"] == technique]
                    offset_to_avoid_zero_in_log = 0.000001
                    # tech_data[metric] = tech_data[metric] / nrm_val
                    tech_data[metric] = (nrm_val - tech_data[metric] + offset_to_avoid_zero_in_log) / nrm_val
                    # tech_data[metric] = (tech_data[metric] - min_val) / (max_val - min_val)
                    ax.plot(
                        tech_data['rate'],
                        tech_data[metric],
                        marker=IDENTITY_MAP[technique]['marker'],
                        label=technique,
                        color=IDENTITY_MAP[technique]['color']
                    )

                ax.set_xlabel("Request Rate (req/s)")

                # ax.set_yscale("symlog")
                ax.set_yscale("log")
                # ax.set_yscale('function', functions=(partial(np.power, 10.0), np.log10))

                if row_id == 0:
                    ax.set_ylabel(metrics_lbl[j] + ' Fq. CV Perf.' + '\n' + r'(1 - norm(fq. CV))')
                else:
                    ax.set_ylabel(metrics_lbl[j] + ' Mean Fq. Perf' + '\n' + r'(1 - norm(fq. drop))')
                # ax.legend()

            else:
                if j > 2:  # we only draw a single row
                    break



                # plot quantified embodied carbon. Our model estimates based on the worst_fq values.
                # ref: Li, et al: "Towards Carbon-efficient LLM Life Cycle" paper
                tot_cls_emb_carbon = 278.3 * 22  # kgCO2eq per server * num. of servers
                cls_refresh_cycle = 3  # existing systems with 3 years of lifespan. In ours, linux is the baseline for this.

                # fq_reduction_linux = flt_df[flt_df["technique"] == "linux"][metric.replace('_after', '_before')] - \
                #                      flt_df[flt_df["technique"] == "linux"][metric]
                fq_reduction_linux = flt_df[flt_df["technique"] == "linux"][metric]
                ax_carbon = axes[j % 3]
                tech_shift_2 = [-0.5, 0.5]
                tech_shift_3 = [-1, 0, 1]
                if len(tech_used) == 2:
                    tech_shift = tech_shift_2
                else:
                    tech_shift = tech_shift_3
                width = 5

                ax_carbon.grid(True, zorder=0)
                our_savings = []
                yearly_emb_carbon_linux = tot_cls_emb_carbon / cls_refresh_cycle
                for idx, technique in enumerate(tech_used):
                    tech_data = flt_df[flt_df["technique"] == technique]
                    # fq_reduction_tech = tech_data[metric.replace('_after', '_before')] - tech_data[metric].values
                    fq_reduction_tech = tech_data[metric]
                    ratios = fq_reduction_linux / fq_reduction_tech.values
                    yearly_emb_carbon_tech = yearly_emb_carbon_linux * (1 / ratios)
                    emb_carbon_savings = yearly_emb_carbon_linux - yearly_emb_carbon_tech.values
                    if "proposed" in technique:
                        our_savings.extend(emb_carbon_savings)

                    ax_carbon.bar(tech_data['rate'] + tech_shift[idx] * width, yearly_emb_carbon_tech, width,
                                  label=technique, color=IDENTITY_MAP[technique]['color'], edgecolor="black")

                avg_savings_proposed = sum(our_savings) / len(our_savings)
                avg_savings_perct = avg_savings_proposed / yearly_emb_carbon_linux
                print("VM cores: "+str(cores)+"| average carbon reduction of proposed for " + metric + " is " + str(round(100 * avg_savings_perct, 3)) + "%")

                ax_carbon.set_ylabel(r'$kgCO_2eq/year$')
                ax_carbon.set_xlabel("Request Rate (req/s)")
                ax_carbon.set_title(metrics_lbl[j] + ' Mean Freq.')
                # ax_carbon.legend(loc='lower right')

        # Adjust layout
        # fig.suptitle("Managing NBTI- and PV-Induced Uneven Frequency Distribution in Machines")

        if not is_carbon_bars:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.05), ncol=len(tech_used), loc="upper center")
            fig.tight_layout()
            plt.minorticks_on()
            plt.savefig("temp_results/aging/vm-cores_" + str(cores) + "_" + filename, bbox_inches='tight')
        else:
            handles, labels = ax_carbon.get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.1), ncol=len(tech_used), loc="upper center")
            fig.tight_layout()
            plt.minorticks_on()
            plt.savefig("temp_results/carbon-savings/vm-cores_" + str(cores) + "_" + filename, bbox_inches='tight')

    for cores in vm_cores:
        # plot_row_data(df, ['linux', 'least-aged'], metrics[:6], metrics_lbl[:6], "aging-impact_baselines.svg", cores,
        #               is_carbon_bars=False)
        plot_row_data(df, ['linux', 'least-aged', 'proposed'], metrics[:6], metrics_lbl[:6],
                      "aging-impact_baselines-vs-proposed.svg", cores, is_carbon_bars=False)
        # plot_row_data(df, ['linux', 'least-aged'], metrics[3:6], metrics_lbl[3:], "carbon-savings_baselines.svg", cores,
        #               is_carbon_bars=True)
        plot_row_data(df, ['linux', 'least-aged', 'proposed'], metrics[3:6], metrics_lbl[3:],
                      "carbon-savings_baselines-vs-proposed.svg", cores, is_carbon_bars=True)


ROOT_LOC = "/Users/tharindu/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/phd-student/projects/dynamic-affinity/experiments"
# ROOT_LOC = "/Users/tharindu/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/phd-student/projects/dynamic-affinity/bk/2024-12-15_02-52-14"
"""At root experiments folder, create sub folder for each technique. Copy each 'rr_{code or conv}_{rate' folders to the relevant technique folder."""

dev_is_plot_fix = True
vm_types = list_dirs(root=ROOT_LOC)
if not os.path.isfile('health_data_df.csv'):
    dev_is_plot_fix = False

if not dev_is_plot_fix:
    tot_parsed_health_data_conv = []
    health_data_df = None
    tot_parsed_core_task_diff_data = pd.DataFrame(columns=['nrm_diff', 'technique', 'rate', 'cores'])
    for vm_type in vm_types:
        print(f"--- vm_type: {vm_type}")
        vm_cores = int(re.search(r'vm(\d+)', vm_type).group(1))
        techniques = list_dirs(root=os.path.join(ROOT_LOC, vm_type))
        for technique in techniques:
            print(f"Processing technique: {technique}")
            curr_loc = os.path.join(ROOT_LOC, vm_type, technique)
            traces = list_dirs(root=curr_loc)
            conv_traces = [trace for trace in traces if CONV_PREFIX in trace]
            # code_traces = [trace for trace in traces if CODE_PREFIX in trace]

            parsed_health_data, parsed_core_task_diff_data = process_exps(root=curr_loc, exps=conv_traces,
                                                                          prefix=CONV_PREFIX,
                                                                          technique=technique,
                                                                          cores=vm_cores)

            tot_parsed_health_data_conv.extend(parsed_health_data)
            tot_parsed_core_task_diff_data = pd.concat([tot_parsed_core_task_diff_data, parsed_core_task_diff_data],
                                                       ignore_index=True)

    health_data_df = pd.DataFrame(tot_parsed_health_data_conv)
    health_data_df["cores"] = health_data_df["cores"].astype(int)
    health_data_df["rate"] = health_data_df["rate"].astype(int)
    health_data_df = health_data_df.sort_values(by=['rate'])

    tot_parsed_core_task_diff_data["cores"] = tot_parsed_core_task_diff_data["cores"].astype(int)
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
