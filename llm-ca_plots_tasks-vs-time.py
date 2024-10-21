import matplotlib.pyplot as plt
import ast
import pandas as pd
import glob
import os

ROOT_LOC = "/path/to/root/data/output/folder"

vm_types = ["dgx-h100-with-cpu-vm40", "dgx-h100-with-cpu-vm80", "dgx-h100-with-cpu-vm112"]
# techniques=["linux", "least-aged", "proposed"]
techniques = ["proposed"]

is_overall = False

for tech in techniques:
    for vm_type in vm_types:
        fig = None
        if not is_overall:
            fig, axes = plt.subplots(2, 2, figsize=(5 * 2, 2 * 2), sharey=True, sharex=True)

        for i, rate in enumerate([str(rate) for rate in [40, 60, 80, 100]]):
            top_root = ROOT_LOC
            root = os.path.join(top_root, vm_type, tech)
            rate_cmp = "/rr_conv_" + str(rate)
            path = root + rate_cmp + "/0_22/bloom-176b/mixed_pool/cpu_usage"

            if not os.path.exists(path):
                print(rate, "do not exist")
                continue
            print(rate, "...")

            # If not already in the given folder, change directory
            os.chdir(path)

            # Use glob to find all CSV files starting with 'task_log_'
            csv_files = glob.glob('task_log_*.csv')

            if is_overall:
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 12))

            # Load each CSV file into a DataFrame
            # max_tasks_log = []
            machine_tasks = []
            for file in csv_files:
                df = pd.read_csv(file)
                data_string = df["tasks_count"].iloc[0]
                # Convert the string to a list of tuples
                data = ast.literal_eval(data_string)

                # Separate the data into two lists: clock times and number of tasks
                clock_times = [item[0] for item in data if 1 < item[0] < 599]
                number_of_tasks = [item[1] for item in data if 1 < item[0] < 599]
                mem_util = [item[2] for item in data if 1 < item[0] < 599]
                awaken_cores = [item[3] for item in data if 1 < item[0] < 599]

                # max_tasks_log.append(max(number_of_tasks))
                machine_tasks.append(number_of_tasks)

                # Plot the data
                if is_overall:
                    ax1.plot(clock_times, number_of_tasks)
                    ax2.plot(clock_times, awaken_cores)
                    ax3.plot(clock_times, mem_util)

            # for each list in the machine_tasks, plot a box plot on ax4
            if is_overall:
                ax4.violinplot(machine_tasks)

                ax1.set_title('Tasks For rate' + str(rate))
                ax1.set_xlabel('Clock Time')
                ax1.set_ylabel('Number of Tasks')

                ax2.set_title('Awaken cores For rate' + str(rate))
                ax2.set_xlabel('Clock Time')
                ax2.set_ylabel('Number of Awaken Cores')

                ax3.set_title('Memory For rate' + str(rate))
                ax3.set_xlabel('Clock Time')
                ax3.set_ylabel('Memory Util.')

                ax4.set_title('Running tasks dist. For rate' + str(rate))
                ax4.set_xlabel('Machine Number')
                ax4.set_ylabel('Distributions')

                plt.grid(True, zorder=0)
                plt.tight_layout()
                plt.savefig("./results_cpu/tasks" + vm_type + '_' + tech + '_' + str(rate) + ".svg")
            else:
                ax = axes[i // 2][(i % 2)]
                ax.violinplot(machine_tasks)

                ax.set_xlabel('Machine Number')
                ax.set_ylabel(f'Task Count at Req./s: {rate}')
                ax.set_xticks(range(1, len(machine_tasks) + 1))
                ax.grid(True, zorder=0)

        if not is_overall:
            plt.grid(True, zorder=0)
            plt.tight_layout()
            plt.savefig("./results_cpu/tasks" + vm_type + '_' + tech + '_running_tasks.svg')
