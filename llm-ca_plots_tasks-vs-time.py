import matplotlib.pyplot as plt
import ast
import pandas as pd
import glob
import os

for rate in [str(rate) for rate in range(30, 251, 20)]:
    top_root = "/Users/tharindu/workspace/splitwise-sim"
    root = top_root + "/results/0/splitwise_5_17"
    rate_cmp = "/rr_conv_" + str(rate)
    path = root + rate_cmp + "/0_22/bloom-176b/mixed_pool/cpu_usage"
    #path = root + rate_cmp + "/21_30/bloom-176b/mixed_pool/cpu_usage"
    #path = root + rate_cmp + "/18_17/bloom-176b/mixed_pool/cpu_usage"

    if not os.path.exists(path):
        print(rate, "do not exist")
        continue
    print(rate, "...")

    # If not already in the given folder, change directory
    os.chdir(path)

    # Use glob to find all CSV files starting with 'task_log_'
    csv_files = glob.glob('task_log_*.csv')

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 12))

    # Load each CSV file into a DataFrame
    #max_tasks_log = []
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

        #max_tasks_log.append(max(number_of_tasks))
        machine_tasks.append(number_of_tasks)

        # Plot the data
        ax1.plot(clock_times, number_of_tasks)
        ax2.plot(clock_times, awaken_cores)
        ax3.plot(clock_times, mem_util)

    # for each list in the machine_tasks, plot a box plot on ax4
    ax4.boxplot(machine_tasks)


    #print(max_tasks_log)

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
    ax4.set_xlabel('Clock Time')
    ax4.set_ylabel('Distributions')


    plt.grid(True)
    plt.tight_layout()
    plt.savefig(top_root + "/temp_results/" + str(rate) + ".png")
