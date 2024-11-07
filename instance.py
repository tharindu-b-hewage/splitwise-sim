import bisect
import logging
import os
import sys

from collections import defaultdict

import utils

from metrics import InstanceMetrics
from node import NodeState
from performance_model import get_duration, get_iteration_duration
#from processor import CPU
from simulator import clock, schedule_event, cancel_event, reschedule_event
from task import PromptTask, TokenTask


class Instance():
    """
    Instance is a scalable unit of deployment for a Model on Servers (Processors).
    Instances run Tasks or batches of Tasks and provide queues for Task execution.
    Instances must communicate with the Executor to run Tasks.

    Only compatible with get_duration from performance_model.

    NOTE: uses a FIFO task queue, not priority queue
    NOTE: preemptions, batching, etc. implemented in subclasses
    """

    cpu = None

    def __init__(self,
                 instance_id,
                 application,
                 name,
                 tag,
                 model,
                 processors,
                 overheads,
                 debug=False):
        self.instance_id = instance_id
        self.application = application
        self.name = name
        self.tag = tag
        self.model = model
        self.processors = processors
        self.overheads = overheads
        self.debug = debug

        ## other instance metadata
        self.metrics = InstanceMetrics()
        self.servers = set()
        for processor in processors:
            self.servers.add(processor.server)
            processor.instances.append(self)
        # needed to implement pause and preemption
        self.completion_events = {}

        ## memory management
        self.memory = self.model.size.total_size
        self.memory_allocs = defaultdict(int)
        self.memory_allocs["model"] = self.model.size.total_size

        gpus = [p for p in self.processors if p.processor_type.value == 2]
        self.max_memory = gpus[0].memory_size * len(gpus)

        ## task queues
        self.pending_queue = []
        self.completed_queue = []
        self.blocked_queue = []
        self.batch = []

        ## scheduler metadata
        self.sched_memory = self.model.size.total_size
        self.sched_pending_tokens = 0
        self.sched_tag = None

        ## instance logger
        if self.debug:
            logger_name = f"instances/{self.application.application_id}/{self.instance_id}"
            level = logging.DEBUG if debug else logging.INFO
            os.makedirs(os.path.dirname(logger_name), exist_ok=True)
            self.scheduler_logger = utils.file_logger(logger_name, level)

        ## set cpu
        self.cpu = list(filter(lambda p: p.processor_type.value == 1, self.processors))[0]

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, memory):
        self._memory = memory
        for processor in self.processors:
            if processor.processor_type.value == 1:
                # TODO: we omit CPU memory for now.
                continue
            processor.memory_used = memory / len(self.processors)

    def alloc_memory(self, tag, memory):
        """
        Allocate memory into the pool.
        """
        self.memory += memory
        self.memory_allocs[tag] += memory

    def free_memory(self, tag, memory):
        """
        Free memory from the pool.
        """
        self.memory -= memory
        self.memory_allocs[tag] -= memory
        self.sched_memory -= memory
        if self.memory_allocs[tag] == 0:
            del self.memory_allocs[tag]

    def task_arrival(self, task):
        """
        Task arrives at this Instance.
        """
        # todo: core assignment is not implemented here, as we only simulate with SpliwiseInstance. If to use this class,
        # core assignment should be implemented.
        task.instance = self
        task.arrive()
        self.pending_queue.append(task)
        if len(self.pending_queue) == 1 and len(self.batch) == 0:
            self.run_task(task)

    def task_completion(self, task, core_id=None):
        """
        Task completes at this Instance.
        """
        # todo: core assignment is not implemented here, as we only simulate with SpliwiseInstance. If to use this class,
        # core assignment should be implemented.
        task.complete()
        self.metrics.busy_time += clock() - self.metrics.run_timestamp
        self.metrics.run_timestamp = 0.
        self.batch.remove(task)
        self.completed_queue.append(task)
        task.executor.finish_task(task, self)
        if len(self.pending_queue) > 0:
            next_task = self.pending_queue[0]
            self.run_task(next_task)

        if core_id is not None:
            self.cpu.release_core_from_cpu_task(task_core_id=core_id)

    def notify_flow_completion(self, flow):
        """
        Notify instance of flow completion.
        """
        pass

    def update_power(self, task):
        """
        Ignore power for now.
        """
        pass

    def run_task(self, task):
        """
        Run a Task on this Instance to completion.
        Does not support iterations.
        """
        task.run()
        self.metrics.run_timestamp = clock()
        self.pending_queue.remove(task)
        self.batch.append(task)
        task.duration = get_duration(task=task,
                                     batch=[task],
                                     instance=self)

        core_id, core_overhead = self.cpu.assign_core_to_cpu_task(task=CpuTaskType.RUN_TASK)
        schedule_event(self.overheads.run + task.duration + core_overhead,
                       lambda instance=self,task=task: instance.task_completion(task, core_id=core_id))

    def preempt_task(self, task):
        """
        Preempt a Task on this Instance.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, instance_cfg, **kwargs):
        instance_type = instance_cfg.instance_type
        if instance_type == "DEFAULT":
            return Instance(**kwargs)
        elif instance_type == "ORCA":
            max_batch_size = instance_cfg.max_batch_size
            return ORCAInstance(max_batch_size=max_batch_size,
                                **kwargs)
        elif instance_type == "Splitwise":
            max_batch_size = instance_cfg.max_batch_size
            max_batch_tokens = instance_cfg.max_batch_tokens
            max_preemptions = instance_cfg.max_preemptions
            return SplitwiseInstance(max_batch_size=max_batch_size,
                                     max_batch_tokens=max_batch_tokens,
                                     max_preemptions=max_preemptions,
                                     **kwargs)
        else:
            raise ValueError(f"Instance type {instance_type} not supported")


class ORCAInstance(Instance):
    """
    Iteration-level FCFS scheduling and selective batching.
    Simulated using contiguous iterations rather than per iteration.
    Multiple tasks from the same request cannot execute concurrently.
    Does not support preemption.

    Only compatible with get_iteration_duration from performance_model.
    """
    def __init__(self,
                 instance_id,
                 application,
                 name,
                 tag,
                 model,
                 processors,
                 overheads,
                 max_batch_size,
                 debug=False):
        super().__init__(instance_id,
                         application,
                         name,
                         tag,
                         model,
                         processors,
                         overheads,
                         debug)

        ## batching metadata
        self.max_batch_size = max_batch_size
        # prompt and token tasks in the batch
        # TODO: track within the batch itself
        self.prompt_tasks_in_batch = []
        self.token_tasks_in_batch = []

        ## token-level tracking metadata
        self.pending_tokens = 0
        self.batch_tokens = 0
        # no max_batch_tokens limit for ORCAInstance
        self.max_batch_tokens = sys.maxsize

        ## contiguous iterations metadata
        self.iteration_duration = 0.
        self.num_contiguous_iterations = 0
        self.pause_next_iteration = False

        ## queues
        # pending requests (not tasks) ordered by arrival time
        # TODO: use an ordered set instead
        self.pending_requests = []
        # separate pending queue for prompt tasks (to prioritize prompts)
        self.pending_prompt_queue = []
        # map requests->tasks on this instance
        self.request_tasks = {}

        if self.debug:
            self.scheduler_logger.debug(
                "name,"
                "tag,"
                "iteration_start,"
                "iteration_end,"
                "batch_size,"
                "prompt_tasks_in_batch,"
                "memory,"
                "pending_requests,"
                "pending_tasks,"
                "blocked_tasks,"
                "pending_prompts,"
                "earliest_request_id,"
                "memory_allocs_size,"
                "num_contiguous_iterations,"
                "batch_tokens,"
                "pending_tokens"
            )

    def log_iteration(self):
        """
        Log Instance state at the end of an iteration.
        """
        if not self.debug:
            return

        iteration_start = self.completion_events["iteration"].time - \
                            (self.iteration_duration * self.num_contiguous_iterations)
        iteration_end = clock()
        try:
            self.scheduler_logger.debug("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
                                        f"{self.name}_{self.instance_id}",
                                        self.tag,
                                        iteration_start,
                                        iteration_end,
                                        len(self.batch),
                                        len(self.prompt_tasks_in_batch),
                                        self.memory,
                                        len(self.pending_requests),
                                        len(self.pending_queue),
                                        len(self.blocked_queue),
                                        len(self.pending_prompt_queue),
                                        self.pending_requests[0].request_id,
                                        len(self.memory_allocs),
                                        self.num_contiguous_iterations,
                                        self.batch_tokens,
                                        self.pending_tokens)
        except Exception as e:
            logging.info(e)
            logging.info("%s,%s,%s,%s,%s,%s,%s,%s",
                         "ERROR",
                         clock(),
                         self.name,
                         self.instance_id,
                         self.memory,
                         self.max_memory,
                         len(self.pending_queue),
                         len(self.pending_requests))

    def add_pending_task(self, task):
        """
        Add a Task to the pending queue.
        """
        self.pending_queue.append(task)
        if isinstance(task, PromptTask):
            self.pending_prompt_queue.append(task)
            self.pending_tokens += task.prompt_size
        elif isinstance(task, TokenTask):
            self.pending_tokens += 1
        else:
            raise ValueError(f"Unexpected task type {task.task_type} in add_pending_task")

    def remove_pending_task(self, task):
        """
        Remove a Task from the pending queue.
        """
        self.pending_queue.remove(task)
        if task in self.blocked_queue:
            self.blocked_queue.remove(task)
        if isinstance(task, PromptTask):
            self.pending_prompt_queue.remove(task)
            self.pending_tokens -= task.prompt_size
        elif isinstance(task, TokenTask):
            self.pending_tokens -= 1
        else:
            raise ValueError(f"Unexpected task type {task.task_type} in remove_pending_task")

    def add_to_pool(self, task):
        """
        Add a Task to the request pool.
        Request pool is ordered by request arrival time.
        """
        if task.request not in self.pending_requests:
            bisect.insort(self.pending_requests, task.request,
                          key=lambda x: x.arrival_timestamp)
            self.request_tasks[task.request] = [task]
        else:
            self.request_tasks[task.request].append(task)

    def remove_from_pool(self, task):
        """
        Remove a Task from the request pool.
        """
        self.request_tasks[task.request].remove(task)
        if len(self.request_tasks[task.request]) == 0:
            self.pending_requests.remove(task.request)
            del self.request_tasks[task.request]

    def task_arrival(self, task):
        # todo: core assignment is not implemented here, as we only simulate with SpliwiseInstance. If to use this class,
        # core assignment should be implemented.
        task.instance = self
        task.arrive()

        # add task to request pool and pending queue
        self.add_to_pool(task)
        self.add_pending_task(task)

        # if no tasks currently executing, start a new iteration
        if len(self.batch) == 0:
            # if instance is blocked due to memory constraints, do nothing
            if self.memory + task.memory > self.max_memory:
                return
            self.start_iteration()
            return

        # otherwise, add to executing batch on the next iteration
        if len(self.batch) < self.max_batch_size and self.batch_tokens <= self.max_batch_tokens:
            self.pause_iteration()
            return

    def add_to_batch(self, task):
        """
        Add a Task to the batch.
        """
        self.batch.append(task)

        if isinstance(task, PromptTask):
            self.prompt_tasks_in_batch.append(task)
        elif isinstance(task, TokenTask):
            self.token_tasks_in_batch.append(task)
        else:
            raise ValueError(f"Task type {task.task_type} not supported")

        # update metrics
        if len(self.batch) == 1:
            self.metrics.run_timestamp = clock()

    def remove_from_batch(self, task):
        """
        Remove a Task from the batch.
        """
        self.batch.remove(task)

        if isinstance(task, PromptTask):
            self.prompt_tasks_in_batch.remove(task)
        elif isinstance(task, TokenTask):
            self.token_tasks_in_batch.remove(task)
        else:
            raise ValueError(f"Task type {task.task_type} not supported")

        # update metrics
        if len(self.batch) == 0:
            self.metrics.busy_time += clock() - self.metrics.run_timestamp
            self.metrics.run_timestamp = 0.

    def get_num_contiguous_iterations(self):
        """
        Find the number of contiguous iterations to run.
        """
        if len(self.batch) == 0:
            return 0
        if len(self.prompt_tasks_in_batch) > 0:
            return 1
        # assumes all tasks are token tasks
        return min(task.token_size - task.generated_tokens for task in self.batch)

    def select_batch(self):
        """
        Select a batch of tasks to run.
        Retains existing tasks and adds new tasks from request pool to the batch.
        """
        old_batch = self.batch
        new_batch = []
        new_tasks = []
        preempted_tasks = []

        for task in old_batch:
            new_batch.append(task)

        memory = self.memory
        for request in self.pending_requests:
            if len(new_batch) == self.max_batch_size:
                break
            task = self.request_tasks[request][0]
            if task in old_batch:
                continue
            if task.state == NodeState.BLOCKED:
                new_batch.append(task)
                new_tasks.append(task)
                continue
            if task.memory + memory <= self.max_memory:
                new_batch.append(task)
                new_tasks.append(task)
                memory += task.memory
            else:
                break

        assert len(preempted_tasks) == 0
        return preempted_tasks, new_tasks

    def start_iteration(self):
        """
        Start a new iteration of a batch of tasks.
        """
        # select a new batch of tasks to run
        preempted_tasks, new_tasks = self.select_batch()

        for task in preempted_tasks:
            self.preempt_task(task)

        for task in new_tasks:
            self.remove_pending_task(task)
            self.add_to_batch(task)

        for request in self.pending_requests:
            task = self.request_tasks[request][0]
            if task not in self.batch:
                task.num_preemptions += 1

        if len(self.batch) == 0:
            if len(self.pending_requests) > 0:
                logging.info("%s,%s,%s,%s,%s,%s,%s,%s,%s",
                             "WARNING",
                             clock(),
                             self.name,
                             self.instance_id,
                             self.memory,
                             self.max_memory,
                             len(self.pending_queue),
                             len(self.pending_requests),
                             len(self.blocked_queue))
                self.application.scheduler.notify_busy_instance(self)
            else:
                self.application.scheduler.notify_free_instance(self)
            return

        # estimate duration of a single iteration
        self.iteration_duration = get_iteration_duration(batch=self.batch,
                                                         instance=self)

        # find number iterations to run contiguously
        self.num_contiguous_iterations = self.get_num_contiguous_iterations()
        for task in self.batch:
            task.generating_tokens = self.num_contiguous_iterations
            if isinstance(task, PromptTask):
                task.processing_tokens = task.prompt_size
            elif isinstance(task, TokenTask):
                task.processing_tokens = self.num_contiguous_iterations
            else:
                raise ValueError(f"Unexpected task type {task.task_type} in start_iteration")

            if task.state == NodeState.QUEUED:
                task.run()
            elif task.state == NodeState.BLOCKED:
                task.run_after_preempt()
            elif task.state == NodeState.RUNNING:
                pass
            else:
                raise ValueError(f"Unexpected task state {task.state} in start_iteration")

        # assign the cpu core that handles the iteration
        core_id, core_overhead = self.cpu.assign_core_to_cpu_task(task=CpuTaskType.INFERENCE_ITERATION, override_task_description=self.application.model_architecture)
        self.completion_events["iteration"] = schedule_event(
            (self.iteration_duration * self.num_contiguous_iterations) + core_overhead,
                        lambda instance=self: instance.complete_iteration(core_id=core_id))

    def pause_iteration(self):
        """
        Pause contiguous iterations at an iteration boundary by resetting the completion event.
        Used if a task arrives which must be executed in the next iteration (typically prompt).
        Assumes that all tasks in the batch are token tasks.
        """
        if self.pause_next_iteration or len(self.prompt_tasks_in_batch) > 0:
            return
        self.pause_next_iteration = True

        contiguous_iteration_duration_old = self.iteration_duration * self.num_contiguous_iterations
        iteration_start = self.completion_events["iteration"].time - \
                            contiguous_iteration_duration_old
        elapsed_time = clock() - iteration_start
        num_completed_iterations = (clock() - iteration_start) // self.iteration_duration
        self.num_contiguous_iterations = num_completed_iterations + 1

        for task in self.batch:
            task.generating_tokens = self.num_contiguous_iterations
            if isinstance(task, TokenTask):
                task.processing_tokens = self.num_contiguous_iterations
            else:
                raise ValueError(f"Unexpected task type {task.task_type} in pause_iteration")

        # reschedule completion event
        contiguous_iteration_duration_new = self.iteration_duration * self.num_contiguous_iterations
        remaining_time = contiguous_iteration_duration_new - elapsed_time

        '''simulating core allocation for the rescheduled event below
        Following event reduces the runtime of the current iteration. It mimics that the iteration process is still 
        executing on the original core, but the runtime is reduced due to the rescheduling of the event. Thus no need 
        for core allocation for this event.
        '''
        self.completion_events["iteration"] = reschedule_event(
                            self.completion_events["iteration"], remaining_time)

    def complete_iteration(self, core_id=None):
        """
        Complete an iteration of a batch tasks.
        Tasks which complete leave the batch.
        Other tasks continue executing in the next iteration.
        """
        if self.debug:
            self.log_iteration()

        # process iteration completion for each task
        completed_tasks = []
        for task in self.batch:
            task.complete_iteration()
            if task.is_complete():
                completed_tasks.append(task)

        # remove completed tasks from batch
        for task in completed_tasks:
            self.task_completion(task)

        # release the cpu core assigned to the iteration
        if core_id is not None:
            self.release_cpu_core(core_id=core_id)

        # start next iteration
        self.pause_next_iteration = False
        self.start_iteration()

    def release_cpu_core(self, core_id):
        self.cpu.release_core_from_cpu_task(task_core_id=core_id)

    def task_completion(self, task, core_id=None):
        """
        Task completes within a batch.
        """
        task.complete()
        self.remove_from_batch(task)
        self.remove_from_pool(task)
        self.completed_queue.append(task)
        task.executor.finish_task(task, self)
        if core_id is not None:
            self.cpu.release_core_from_cpu_task(task_core_id=core_id)

    def notify_flow_completion(self, flow, core_for_flow_completion_notify=None):
        """
        Notify instance of flow completion.
        """
        if core_for_flow_completion_notify is not None:
            self.cpu.release_core_from_cpu_task(task_core_id=core_for_flow_completion_notify)

        if len(self.pending_queue) == 0:
            return

        task = self.pending_queue[0]
        # if no tasks currently executing, start a new iteration
        if len(self.batch) == 0:
            # if instance is blocked due to memory constraints, do nothing
            if self.memory + task.memory > self.max_memory:
                return
            self.start_iteration()
            return

        # otherwise, add to executing batch on the next iteration
        if len(self.batch) < self.max_batch_size and self.batch_tokens < self.max_batch_tokens:
            self.pause_iteration()
            return

from enum import Enum

# default overhead time is an indicative value. we use default timeslice of real-time process in linux kernel,
# which is 100ms, assuming if none provided, task is treated as real-time process, in terms of the time slice.
LINUX_RR_PROCESS_TIMESLICE = 0.1
class CpuTaskType(Enum):
    HANDLE_TASK_ARRIVAL = {"overhead_time": LINUX_RR_PROCESS_TIMESLICE, "info": "handle task arrival"}
    INFERENCE_ITERATION = {"overhead_time": LINUX_RR_PROCESS_TIMESLICE, "info": "inference iteration"}
    RUN_TASK = {"overhead_time": LINUX_RR_PROCESS_TIMESLICE, "info": "run task"}
    FLOW_COMPLETION = {"overhead_time": LINUX_RR_PROCESS_TIMESLICE, "info": "flow completion"}

class CpuTask:
    task: CpuTaskType
    meta: str

    def __init__(self, task, meta):
        self.task: CpuTaskType = task
        self.meta = meta

class SplitwiseInstance(ORCAInstance):
    """
    Supports preemptions and configurable batch tokens.

    Only compatible with get_iteration_duration from performance_model.
    """
    def __init__(self,
                 instance_id,
                 application,
                 name,
                 tag,
                 model,
                 processors,
                 overheads,
                 max_batch_size,
                 max_preemptions,
                 max_batch_tokens,
                 debug=False):
        super().__init__(instance_id,
                         application,
                         name,
                         tag,
                         model,
                         processors,
                         overheads,
                         max_batch_size,
                         debug)
        self.max_preemptions = max_preemptions
        self.max_batch_tokens = max_batch_tokens

    def preempt_task(self, task):
        """
        Preempt a Task on this Instance.
        """
        task.preempt()
        if isinstance(task, PromptTask):
            raise ValueError("Prompt tasks cannot be preempted")
        self.remove_from_batch(task)
        self.add_pending_task(task, preempt=True)

    def add_pending_task(self, task, preempt=False):
        """
        Add a Task to the pending queue, ordered by number of preemptions and arrival time.
        """
        bisect.insort(self.pending_queue, task,
                      key=lambda x: (x.num_preemptions, x.request.arrival_timestamp))
        if preempt:
            self.blocked_queue.append(task)
        if isinstance(task, PromptTask):
            self.pending_prompt_queue.append(task)
            self.pending_tokens += task.prompt_size
        elif isinstance(task, TokenTask):
            self.pending_tokens += 1
        else:
            raise ValueError(f"Unexpected task type {task.task_type} in add_pending_task")

    def remove_pending_task(self, task):
        """
        Remove a Task from the pending queue.
        """
        self.pending_queue.remove(task)
        if task in self.blocked_queue:
            self.blocked_queue.remove(task)
        if isinstance(task, PromptTask):
            self.pending_prompt_queue.remove(task)
            self.pending_tokens -= task.prompt_size
        elif isinstance(task, TokenTask):
            self.pending_tokens -= 1
        else:
            raise ValueError(f"Unexpected task type {task.task_type} in remove_pending_task")

    def task_arrival(self, task, core_id_for_task_arrival_function=None):

        # when task arrival function is triggered, the caller is expected to trigger event with a delay to simulate
        # runtime of this function on the allocated cpu core. So we deallocated the core here, primarily
        # because three different outcomes of this function returns immediately.
        if core_id_for_task_arrival_function is not None:
            self.cpu.release_core_from_cpu_task(task_core_id=core_id_for_task_arrival_function)

        task.instance = self
        task.arrive()

        # add task to request pool and pending queue
        self.add_to_pool(task)
        self.add_pending_task(task)

        # if no tasks currently executing, start a new iteration
        if len(self.batch) == 0:
            # if instance is blocked due to memory constraints, do nothing
            if self.memory + task.memory > self.max_memory:
                return
            self.start_iteration()
            return

        # otherwise, add to executing batch on the next iteration
        if len(self.batch) < self.max_batch_size and \
            self.batch_tokens + task.tokens_per_iteration <= self.max_batch_tokens:
            self.pause_iteration()
            return

        # otherwise, check whether to preempt
        if isinstance(task, PromptTask):
            batch_prompt_tokens = self.batch_tokens - len(self.token_tasks_in_batch)
            if len(self.prompt_tasks_in_batch) < len(self.batch) and \
                batch_prompt_tokens + task.prompt_size <= self.max_batch_tokens:
                self.preempt_iteration()
                return

    def preempt_iteration(self):
        """
        Preempt contiguous iterations at an iteration boundary by resetting the completion event.
        Used if a task arrives that must be executed in the next iteration.
        Assumes that all tasks in the batch are token tasks.
        """
        return self.pause_iteration()

    def select_batch(self):
        """
        Select a batch of tasks to run.
        Preempt token tasks from the requests with the latest arrival times.
        Returns the list of preempted tasks.
        TODO: clean up and simplify logic
        """
        old_batch = self.batch
        new_batch = []
        batch_requests = set()
        new_tasks = []
        preempted_tasks = []

        batch_tokens = 0
        memory = self.memory

        # run any task that has been preempted too many times
        for task in self.pending_queue:
            if task.num_preemptions >= self.max_preemptions:
                if len(new_batch) == self.max_batch_size:
                    break
                if len(new_batch) > 0 and \
                    batch_tokens + task.tokens_per_iteration > self.max_batch_tokens:
                    break
                if task.request in batch_requests:
                    continue
                if task.state == NodeState.BLOCKED:
                    new_batch.append(task)
                    batch_requests.add(task.request)
                    batch_tokens += task.tokens_per_iteration
                    continue
                if task.memory + memory <= self.max_memory:
                    new_batch.append(task)
                    batch_requests.add(task.request)
                    memory += task.memory
                    batch_tokens += task.tokens_per_iteration
            else:
                break

        # add prompt tasks to the batch
        # assumes we don't have prompt tasks in old_batch since they completed
        for task in self.pending_prompt_queue:
            if len(new_batch) == self.max_batch_size:
                break
            if len(new_batch) > 0 and \
                batch_tokens + task.tokens_per_iteration > self.max_batch_tokens:
                break
            if task.request in batch_requests:
                continue
            if task.memory + memory <= self.max_memory:
                new_batch.append(task)
                batch_requests.add(task.request)
                memory += task.memory
                batch_tokens += task.tokens_per_iteration
            else:
                break

        # then add blocked token tasks to the batch
        for task in self.blocked_queue:
            if len(new_batch) == self.max_batch_size:
                break
            if len(new_batch) > 0 and \
                batch_tokens + task.tokens_per_iteration > self.max_batch_tokens:
                break
            if task.request in batch_requests:
                continue
            if task.state != NodeState.BLOCKED:
                raise ValueError("Task in blocked queue is not blocked")
            new_batch.append(task)
            batch_requests.add(task.request)
            batch_tokens += task.tokens_per_iteration

        # then add old_batch token tasks to the batch
        for task in old_batch:
            if len(new_batch) == self.max_batch_size:
                break
            if len(new_batch) > 0 and \
                batch_tokens + task.tokens_per_iteration > self.max_batch_tokens:
                break
            if task.request in batch_requests:
                continue
            new_batch.append(task)
            batch_requests.add(task.request)
            batch_tokens += task.tokens_per_iteration

        # then add any other token tasks to the batch
        for request in self.pending_requests:
            if len(new_batch) == self.max_batch_size:
                break
            task = self.request_tasks[request][0]
            if task.request in batch_requests:
                continue
            if len(new_batch) > 0 and \
                batch_tokens + task.tokens_per_iteration > self.max_batch_tokens:
                break
            if task.state == NodeState.BLOCKED:
                new_batch.append(task)
                batch_requests.add(task.request)
                batch_tokens += task.tokens_per_iteration
                continue
            if task.memory + memory <= self.max_memory:
                new_batch.append(task)
                batch_requests.add(task.request)
                memory += task.memory
                batch_tokens += task.tokens_per_iteration
            else:
                break
        self.batch_tokens = batch_tokens

        preempted_tasks = [task for task in old_batch if task not in new_batch]
        new_tasks = [task for task in new_batch if task not in old_batch]
        return preempted_tasks, new_tasks
