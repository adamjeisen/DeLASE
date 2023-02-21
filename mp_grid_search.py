import logging
import numpy as np
import os
import pandas as pd
import queue
import time
import torch
import torch.multiprocessing as mp
import traceback
from tqdm.auto import tqdm
import sys

from delase import DeLASE
from parameter_choosing import fit_and_test_delase, ParameterGrid
from performance_metrics import get_autocorrel_funcs

def mp_worker(worker_num, task_queue, message_queue=None, use_cuda=False):
    # until the task queue is empty, keep taking tasks from the queue and completing them
    while True:
        try:
            # pull a task from the queue
            task_params = task_queue.get_nowait()
            signal, test_signal, window, p, fit_and_test_args, results_dir = task_params
            fit_and_test_args['message_queue'] = message_queue
            fit_and_test_args['worker_num'] = worker_num
            # -----------
            # Compute hankel matrix and SVD
            # -----------
            if use_cuda:
                fit_and_test_args['device']=worker_num
            else:
                fit_and_test_args['device']='cpu',
            
            results = fit_and_test_delase(signal, test_signal, window, p, **fit_and_test_args)
            pd.to_pickle(results, os.path.join(results_dir, f'ret_window_{window}_p_{p}'))
    
            task_queue.task_done()

        # handling what happens when the queue is found to be empty
        except queue.Empty:
            if message_queue is not None:
                message_queue.put((worker_num, "shutting down...", "INFO"))
            break
        # handling any other exceptions that might come up
        except:
            tb = traceback.format_exc()
            if message_queue is not None:
                message_queue.put((worker_num, tb, "ERROR"))
            task_queue.task_done()

if __name__ == '__main__':

    USE_TORCH = True
    USE_CUDA = True
    NUM_WORKERS = 2

    USE_LOGGING = True
    LOG_DIR = "."
    LOG_NAME = 'mp_delase'
    LOG_LEVEL = logging.DEBUG

    RESULTS_DIR = '/scratch2/weka/millerlab/eisenaj/ChaoticConsciousness/temp/results'

    COMPUTE_IP = True
    COMPUTE_CHROOTS = True

    TRACK_RESEEDS = True

    # ----------------------
    # Simulation parameters
    # ----------------------

    # parameter grid
    window_vals = np.array([10000])
    p_vals = np.array([50, 60, 70, 80, 90, 100])
    # r_thresh_vals=np.array([0.3, 0.4, 0.5])
    r_thresh_vals = np.array([0.25, 0.5])
    # lamb_vals = np.array([0, 1e-12, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    lamb_vals = np.array([0, 1e-3])
    reseed_vals=np.array([1, 5, 10, 15, 20, 30, 40, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000])
    parameter_grid = ParameterGrid(window_vals, p_vals, r_thresh_vals, lamb_vals, reseed_vals)

    train_data_path = '/scratch2/weka/millerlab/eisenaj/ChaoticConsciousness/temp/data/train.pkl'
    test_data_path = '/scratch2/weka/millerlab/eisenaj/ChaoticConsciousness/temp/data/test.pkl'

    train_signal = pd.read_pickle(train_data_path)
    test_signal = pd.read_pickle(test_data_path)
    dt = 0.001

    integrated_performance_kwargs = dict(
        metrics=['autocorrel_correl', 'fft_correl', 'fft_r2'], 
        weights='equal',
        num_lags=500,
        max_freq=500,
        fft_n=1000,
    )

    stability_max_freq = 500

    # ----------------------
    # Set up logging
    # ----------------------
    if USE_LOGGING:
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        os.makedirs(LOG_DIR, exist_ok=True)
        logging.basicConfig(filename=os.path.join(LOG_DIR,f"{LOG_NAME}_{timestamp}.log"),
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt="%Y-%m-%d %H:%M:%S",
                                level=LOG_LEVEL)
        logger = logging.getLogger('mp_log')

        logger.debug("HEY!")
        # logger.info(f"CPU count is: {os.cpu_count()}")

    # ----------------------
    # Set up multiprocessing
    # ----------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mp.set_start_method('spawn')
    task_queue = mp.Manager().JoinableQueue()
    message_queue = mp.Manager().JoinableQueue()
    num_tasks = parameter_grid.total_combinations
    if not TRACK_RESEEDS:
        num_tasks = int(num_tasks/len(parameter_grid.reseed_vals))

    if 'num_lags' in integrated_performance_kwargs.keys():
        autocorrel_kwargs = {'num_lags': integrated_performance_kwargs['num_lags']}
    else:
        autocorrel_kwargs = {}
    autocorrel_true = get_autocorrel_funcs(test_signal, use_torch=USE_TORCH, device='cuda' if USE_CUDA else 'cpu', **autocorrel_kwargs)

    fit_and_test_args = dict(
        parameter_grid=parameter_grid,
        dt=dt,
        compute_ip=COMPUTE_IP,
        autocorrel_true=autocorrel_true.cpu(),
        integrated_performance_kwargs=integrated_performance_kwargs,
        compute_chroots=COMPUTE_CHROOTS,
        stability_max_freq=stability_max_freq,
        use_torch=USE_TORCH,
        track_reseeds=TRACK_RESEEDS
    )

    for window in parameter_grid.window_vals:
        signal = train_signal[:window]
        for p in parameter_grid.p_vals:
            task_queue.put((signal, test_signal, window, p, fit_and_test_args, RESULTS_DIR))
    
    num_workers = NUM_WORKERS
    processes = []
    for worker_num in range(num_workers):
        p = mp.Process(target=mp_worker, args=(worker_num, task_queue, message_queue, USE_CUDA))
        p.start()
        processes.append(p)

    # monitor for messages from workers
    killed_workers = 0
    iterator = tqdm(total=num_tasks)
    while True:
        try:
            worker_num, message, log_level = message_queue.get_nowait()
            
            if message == 'task complete':
                iterator.update(1)
            elif message == "shutting down...":
                killed_workers += 1
            # print the message from the workr
            if USE_LOGGING:
                logger.log(getattr(logging, log_level), f"[worker {worker_num}]: {message}")
            else:
                print(f"[{log_level}] [worker {worker_num}]: {message}")

            message_queue.task_done()
            if killed_workers == num_workers and message_queue.qsize()==0:
                break
            
            sys.stdout.flush()
        except queue.Empty:
            time.sleep(0.5)
    message_queue.join()
    iterator.close()

    for p in processes:
        p.join()