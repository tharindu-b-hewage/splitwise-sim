import configparser
import heapq
import logging
import os

from collections import defaultdict
from platform import processor

import numpy as np

import utils
from core_power import CPU_CORE_ADJ_INTERVAL

# global simulator that drives the simulation
# bad practice, but it works for now
sim = None


class Event:
    """
    Events are scheduled actions in the simulator.
    """
    def __init__(self, time, action):
        self.time = time
        self.action = action

    def __str__(self):
        return f"Event with time {self.time} and action {self.action}"

    def __lt__(self, other):
        return self.time < other.time


class Simulator:
    """
    A discrete event simulator that schedules and runs Events.
    """
    def __init__(self, end_time):
        global sim
        sim = self
        self.time = 0
        self.end_time = end_time
        self.events = []
        self.deleted_events = []
        logging.info("Simulator initialized")

        # logger for simulator events
        self.logger = utils.file_logger("simulator")
        self.logger.info("time,event")

    def schedule(self, delay, action):
        """
        Schedule an event by specifying delay and an action function.
        """
        # run immediately if delay is 0
        if delay == 0:
            action()
            return None
        event = Event(self.time + delay, action)
        heapq.heappush(self.events, event)
        return event

    def cancel(self, event):
        """
        Cancel an event.
        """
        self.deleted_events.append(event)

    def reschedule(self, event, delay):
        """
        Reschedule an event by cancelling and scheduling it again.
        """
        self.cancel(event)
        return self.schedule(delay, event.action)

    def run(self):
        """
        Run the simulation until the end time.
        """
        while self.events and self.time < self.end_time:
            event = heapq.heappop(self.events)
            if event in self.deleted_events:
                self.deleted_events.remove(event)
                continue
            self.time = event.time
            event.action()
            self.logger.debug(f"{event.time},{event.action}")


class TraceSimulator(Simulator):
    """
    A discrete event simulator that processes Request arrivals from a Trace.
    """
    def __init__(self,
                 trace,
                 cluster,
                 applications,
                 router,
                 arbiter,
                 end_time):
        super().__init__(end_time)
        self.trace = trace
        self.cluster = cluster
        self.applications = applications
        self.router = router
        self.arbiter = arbiter
        logging.info("TraceSimulator initialized")
        self.last_request_arrival = 0.0
        self.load_trace()

    def load_trace(self):
        """
        Load requests from the trace as arrival events.
        """
        for request in self.trace.requests:
            arrival_timestamp = request.arrival_timestamp
            self.schedule(request.arrival_timestamp,
                          lambda request=request: self.router.request_arrival(request))
            self.last_request_arrival = arrival_timestamp

    def run(self):
        # start simulation by scheduling a cluster run
        self.schedule(0, self.cluster.run)
        self.schedule(0, self.router.run)
        self.schedule(0, self.arbiter.run)

        # add a status entry at the beginning in the cpu usage log files. this is needed to process the collected data.
        self.cluster.trigger_state_update()

        # schedule periodic monitoring in servers
        def load_properties(file_path):
            config = configparser.ConfigParser()
            # ConfigParser requires section headers, so we add a fake one
            absolute_directory = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(absolute_directory, file_path), 'r') as file:
                properties_data = f"[DEFAULT]\n{file.read()}"
            config.read_string(properties_data)
            return config['DEFAULT']

        CPU_CONFIGS = load_properties('cpu_configs.properties')
        if CPU_CONFIGS.get("task_allocation_algo") == "proposed":
            periodic_interval = 1.0
            for interval_start in np.arange(0.0, self.last_request_arrival, periodic_interval):
                for sku in self.cluster.servers:
                    for server in self.cluster.servers[sku]:
                        cpu = list(filter(lambda p: p.processor_type.value == 1, server.processors))[0]
                        self.schedule(interval_start, lambda cpu=cpu: cpu.adjust_sleeping_cores())

        # run simulation
        super().run()
        self.logger.info(f"{self.time},end")
        logging.info(f"TraceSimulator completed at {self.time}")

        # below also triggers a status update call, such that each cpu core logs their status at the end of the simulation.
        self.save_results()

    def save_results(self, detailed=True):
        """
        Save results at the end of the simulation.
        """
        self.router.save_results()

        sched_results = {}
        alloc_results = {}
        for application_id, application in self.applications.items():
            allocator_results, scheduler_results = application.get_results()
            alloc_results[application_id] = allocator_results
            sched_results[application_id] = scheduler_results

        # summary sched results
        summary_results = defaultdict(list)
        for application_id, results_dict in sched_results.items():
            summary_results["application_id"].append(application_id)
            for key, values in results_dict.items():
                summary = utils.get_statistics(values)
                # merge summary into summary_results
                for metric, value in summary.items():
                    summary_results[f"{key}_{metric}"].append(value)

        # save summary results
        utils.save_dict_as_csv(summary_results, "summary.csv")

        if detailed:
            # create a dataframe of all requests, save as csv
            for application_id, result in sched_results.items():
                utils.save_dict_as_csv(result, f"detailed/{application_id}.csv")
            for application_id, result in alloc_results.items():
                utils.save_dict_as_csv(result, f"detailed/{application_id}_alloc.csv")

        # save CPU core activity
        server_cpu_usage = self.cluster.cpu_core_usage()
        for index, cpu_usage in enumerate(server_cpu_usage):
            name, usage = cpu_usage
            utils.save_dict_as_csv(usage, f"cpu_usage/cpu_usage_{name}_{index}.csv")

        task_logs = self.cluster.task_logs()
        for index, log in enumerate(task_logs):
            machine_name, data = log
            utils.save_dict_as_csv(data, f"cpu_usage/task_log_{machine_name}_{index}.csv")

        slp_mgt_logs = self.cluster.sleep_mgt_logs()
        for index, log in enumerate(slp_mgt_logs):
            machine_name, data = log
            utils.save_dict_as_csv(data, f"cpu_usage/slp_mgt_log_{machine_name}_{index}.csv")


# Convenience functions for simulator object

def clock():
    """
    Return the current time of the simulator.
    """
    return sim.time

def schedule_event(*args):
    """
    Schedule an event in the simulator at desired delay.
    """
    return sim.schedule(*args)

def cancel_event(*args):
    """
    Cancel existing event in the simulator.
    """
    return sim.cancel(*args)

def reschedule_event(*args):
    """
    Reschedule existing event in the simulator.
    Equivalent to cancelling and scheduling a new event.
    """
    return sim.reschedule(*args)
