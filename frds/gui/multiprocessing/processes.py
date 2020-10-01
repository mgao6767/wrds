from typing import Dict
import multiprocessing
import concurrent.futures

from PyQt5 import QtCore

from frds.settings import MAX_WORKERS, PROGRESS_UPDATE_INTERVAL_SECONDS
from frds.gui.multiprocessing import (
    STATUS_ERROR,
    STATUS_COMPLETE,
    STATUS_RUNNING,
    DEFAULT_STATE,
    WorkerSignals,
)


class ProcessManager(QtCore.QAbstractListModel):
    _state = {}
    status = QtCore.pyqtSignal(str)
    jobs: Dict[str, concurrent.futures.Future] = {}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.progress_dict = multiprocessing.Manager().dict()
        self.max_workers = MAX_WORKERS
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=self._process_initializer,
            initargs=(),
        )
        self.signals = WorkerSignals()
        self.status_timer = QtCore.QTimer()
        self.status_timer.setInterval(PROGRESS_UPDATE_INTERVAL_SECONDS * 1000)
        self.status_timer.timeout.connect(self.notify_status)
        self.status_timer.start()

    def close(self):
        self.cleanup()
        self.status_timer.stop()
        self.executor.shutdown(wait=False)

    @staticmethod
    def _process_initializer(*args, **kwargs):
        """Initializer function runs before every process submitted to the executor"""
        pass

    @QtCore.pyqtSlot()
    def notify_status(self):
        self.update_progress()
        running = min(self.running_jobs, self.max_workers)
        waiting = max(0, self.total_jobs - self.max_workers)
        self.status.emit(
            f"{running} running, {waiting} waiting, {self.max_workers} threads"
        )

    def cleanup(self):
        for job_id, state in self._state.copy().items():
            if state["status"] in (STATUS_COMPLETE, STATUS_ERROR):
                del self._state[job_id]
        self.layoutChanged.emit()

    def add_estimation_job(self, worker) -> None:
        self.jobs[worker.job_id] = self.executor.submit(
            worker.fn,
            *worker.args,
            job_id=worker.job_id,
            progress=self.progress_dict.update,
            **worker.kwargs,
        )
        self._state[worker.job_id] = DEFAULT_STATE.copy()
        self.layoutChanged.emit()

    def cancel_all_jobs(self) -> None:
        for _, f in self.jobs.items():
            f.cancel()

    @property
    def total_jobs(self) -> int:
        return len(self.jobs)

    @property
    def running_jobs(self) -> int:
        return len([f for f in self.jobs.values() if f.running()])

    @property
    def completed_jobs(self) -> int:
        return len([f for f in self.jobs.values() if f.done()])

    def update_progress(self) -> None:
        latest_progress = {}
        for job_id, f in self.jobs.copy().items():
            if f.done():
                self.jobs.pop(job_id)  # remove completed jobs
                self._state[job_id]["progress"] = 100
                err = f.exception()
                if err:
                    self.signals.error.emit(job_id, str(err))
                    print(err)
                    self._state[job_id]["status"] = STATUS_ERROR
                else:
                    result = f.result()
                    self.signals.result.emit(job_id, result)
                    self._state[job_id]["status"] = STATUS_COMPLETE
            elif f.running():
                self._state[job_id]["status"] = STATUS_RUNNING
                # Mark default progress as 1 in case the running estimation function
                # doesn't update progress to the monitor. If set to 0, it will not show
                # up in the monitor.
                self._state[job_id]["progress"] = min(
                    max(self.progress_dict.get(job_id, 1), 1), 100
                )
            elif f.cancelled():
                pass
            self.layoutChanged.emit()

    # Model interface
    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            job_ids = list(self._state.keys())
            job_id = job_ids[index.row()]
            return job_id, self._state[job_id]

    def rowCount(self, _):
        return len(self._state)
