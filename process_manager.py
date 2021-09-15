from enum import Enum
from multiprocessing import Process, Manager
from threading import Lock, Thread
from typing import Optional, List, Callable, Tuple, Any, TypedDict
from tqdm import tqdm


class JobStatus(Enum):
    WAITING = 0
    RUNNING = 1
    DONE = 2
    ERROR = 3


class ProcessManagerOutput(TypedDict):
    results: List[Any]
    errors: List[Optional[Exception]]
    status: List[JobStatus]


DEFAULT_MAX_PROCS = 20


class ProcessManager:
    _lock: Optional[Lock] = None
    _max_procs: int = DEFAULT_MAX_PROCS
    _status: List[JobStatus] = []
    _func: Callable = None
    _arg_list: List[Tuple] = None
    _results: List[Any] = list()
    _errors: List[Optional[Exception]] = list()
    _common_kwargs: dict = dict()
    _worker_list: List[Process] = list()

    @property
    def num_waiting(self) -> int:
        return len([x for x in self._status if x == JobStatus.WAITING])

    @property
    def num_running(self) -> int:
        return len([x for x in self._status if x == JobStatus.RUNNING])

    @property
    def num_done(self) -> int:
        return len([x for x in self._status if x == JobStatus.DONE])

    @property
    def num_error(self) -> int:
        return len([x for x in self._status if x == JobStatus.ERROR])

    @property
    def num_finished(self) -> int:
        return self.num_error + self.num_done

    def __init__(self, max_num_procs: int = DEFAULT_MAX_PROCS):
        self._lock = Lock()
        self._max_procs = max_num_procs

    def __call__(self, func: Callable, arg_list: List[Tuple], monitor: bool = True, **kwargs) \
            -> ProcessManagerOutput:

        self._func = func
        self._arg_list = arg_list
        self._common_kwargs = kwargs
        self._num_procs = len(arg_list)
        self._process_manager = Manager()
        self._status = self._process_manager.list([JobStatus.WAITING for _ in range(self._num_procs)])
        self._results = self._process_manager.list([None for _ in range(self._num_procs)])
        self._errors = self._process_manager.list([None for _ in range(self._num_procs)])
        self._worker_list = list()
        self._monitor = monitor

        spawning_th = Thread(target=self._spawning_thread)
        collecting_th = Thread(target=self._collecting_thread)

        spawning_th.start()
        collecting_th.start()
        spawning_th.join()
        collecting_th.join()

        ret: ProcessManagerOutput = {
            'results': list(self._results),
            'errors': list(self._errors),
            'status': list(self._status)
        }
        return ret

    def _spawning_thread(self):
        while self.num_waiting > 0:
            if self.num_running < self._max_procs:
                idx = self._get_next_work()
                if idx is None:
                    break
                p = Process(target=self._wrapped_func, args=(idx,))
                p.start()
                with self._lock:
                    self._worker_list.append(p)

    def _wrapped_func(self, idx):
        if isinstance(idx, tuple):
            idx = idx[0]
        elif not isinstance(idx, int):
            raise TypeError
        if self._monitor:
            print(f"{idx} called")
        args = self._arg_list[idx]
        try:
            ret = self._func(*args, **self._common_kwargs)
            with self._lock:
                self._status[idx] = JobStatus.DONE
                self._results[idx] = ret
            if self._monitor:
                print(f"{idx} terminated safely")
        except Exception as e:
            with self._lock:
                self._status[idx] = JobStatus.ERROR
                self._errors[idx] = e
            if self._monitor:
                print(f"{idx} terminated with error {type(e)}: {e}")

    def _get_next_work(self):
        if self.num_waiting == 0:
            return None
        else:
            ret = self._status.index(JobStatus.WAITING)
            with self._lock:
                self._status[ret] = JobStatus.RUNNING
            return ret

    def _collecting_thread(self):
        while self.num_finished < self._num_procs:
            for worker in self._worker_list:
                if not worker.is_alive():
                    worker.join()
                    with self._lock:
                        self._worker_list.remove(worker)


if __name__ == "__main__":
    def func(x, y):
        if y % 2 :
            raise ValueError
        return (x+y) ** 2


    pr = ProcessManager(5)
    result = pr(func, [(x, x+1) for x in range(10)], True)
    print(result)
