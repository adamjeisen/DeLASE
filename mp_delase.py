import argparse 
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
from utils import load_window_from_chunks

def mp_worker(worker_num, task_queue, message_queue=None, use_cuda=False):
    # until the task queue is empty, keep taking tasks from the queue and completing them
    while True:
        try:
            # pull a task from the queue
            task_params = task_queue.get_nowait()
            data_loading_args, window, expansion_val, expansion_type, autocorrel_kwargs, fit_and_test_args, T_pred, RESULTS_DIR, session, area = task_params
            if expansion_type == 'p':
                p = expansion_val
            else: # expansion_type == 'matrix_size'
                matrix_size = expansion_val
            
            results_dir = os.path.join(RESULTS_DIR, session, area)

            os.makedirs(results_dir, exist_ok=True)
            save_path = os.path.join(results_dir, f"{data_loading_args['window_start']}_window_{window}_{expansion_type}_{expansion_val}")
            if os.path.exists(save_path):
                if message_queue is not None:
                    message_queue.put((worker_num, f"{save_path} is already complete", "DEBUG"))
                    message_queue.put((worker_num, "task complete", "DEBUG"))
            else:
                if message_queue is not None:
                    message_queue.put((worker_num, f"starting {save_path}", "DEBUG"))

                data_loading_args['window_end'] = data_loading_args['window_start'] + (window + T_pred)*fit_and_test_args['dt']
                signal = load_window_from_chunks(**data_loading_args)

                os.makedirs(results_dir, exist_ok=True)
                fit_and_test_args['message_queue'] = message_queue
                fit_and_test_args['worker_num'] = worker_num
                # -----------
                # Compute hankel matrix and SVD
                # -----------
                if use_cuda:
                    fit_and_test_args['device']=worker_num
                else:
                    fit_and_test_args['device']='cpu'
                
                autocorrel_true = get_autocorrel_funcs(signal[window:window + T_pred], use_torch=True, device=worker_num if use_cuda else 'cpu', **autocorrel_kwargs)
                fit_and_test_args['autocorrel_true'] = autocorrel_true

                if expansion_type == 'matrix_size':
                    p = int(np.ceil(matrix_size/signal.shape[1]))

                results = fit_and_test_delase(signal[:window], signal[window:window + T_pred], window, p, **fit_and_test_args)
                
                if expansion_type == 'matrix_size':
                    for i in range(len(results)):
                        results[i]['matrix_size'] = matrix_size
                
                pd.to_pickle(results, save_path)

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

    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument('path', type=str, 
                            help='Required path to the multiprocessing argument dictionary, pickled.')
    command_line_args = parser.parse_args()

    mp_args = argparse.Namespace(**pd.read_pickle(command_line_args.path))

    # ----------------------
    # Set up logging
    # ----------------------
    if mp_args.USE_LOGGING:
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        os.makedirs(mp_args.LOG_DIR, exist_ok=True)
        logging.basicConfig(filename=os.path.join(mp_args.LOG_DIR,f"{mp_args.LOG_NAME}_{timestamp}.log"),
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt="%Y-%m-%d %H:%M:%S",
                                level=mp_args.LOG_LEVEL)
        logger = logging.getLogger('mp_log')

        logger.debug("HEY!")
        # logger.info(f"CPU count is: {os.cpu_count()}")

    # ----------------------
    # Set up multiprocessing
    # ----------------------
    os.makedirs(mp_args.RESULTS_DIR, exist_ok=True)
    mp.set_start_method('spawn')
    task_queue = mp.Manager().JoinableQueue()
    message_queue = mp.Manager().JoinableQueue()

    if 'num_lags' in mp_args.integrated_performance_kwargs.keys():
        autocorrel_kwargs = {'num_lags': mp_args.integrated_performance_kwargs['num_lags']}
    else:
        autocorrel_kwargs = {}

    fit_and_test_args = dict(
        parameter_grid=mp_args.parameter_grid,
        dt=mp_args.dt,
        compute_ip=mp_args.COMPUTE_IP,
        integrated_performance_kwargs=mp_args.integrated_performance_kwargs,
        compute_chroots=mp_args.COMPUTE_CHROOTS,
        stability_max_freq=mp_args.stability_max_freq,
        stability_max_unstable_freq=mp_args.stability_max_unstable_freq,
        use_torch=mp_args.USE_TORCH,
        track_reseeds=mp_args.TRACK_RESEEDS
    )

    num_tasks = mp_args.parameter_grid.total_combinations
    if not mp_args.TRACK_RESEEDS:
        num_tasks = int(num_tasks/len(mp_args.parameter_grid.reseed_vals))
    num_tasks *= len(mp_args.data_processing_df)
    
    if mp_args.parameter_grid.p_vals is not None:
        for i, row in mp_args.data_processing_df.iterrows():
            for window in mp_args.parameter_grid.window_vals:
                for p in mp_args.parameter_grid.p_vals:
                    task_queue.put((row[['window_start', 'window_end', 'directory', 'dimension_inds']], window, p, 'p', autocorrel_kwargs, fit_and_test_args, mp_args.T_pred, mp_args.RESULTS_DIR, row.session, row.area))
    else: # mp_args.parameter_grid.matrix_size_vals is not None
        for i, row in mp_args.data_processing_df.iterrows():
            for window in mp_args.parameter_grid.window_vals:
                for matrix_size in mp_args.parameter_grid.matrix_size_vals:
                    task_queue.put((row[['window_start', 'window_end', 'directory', 'dimension_inds']], window, matrix_size, 'matrix_size', autocorrel_kwargs, fit_and_test_args, mp_args.T_pred, mp_args.RESULTS_DIR, row.session, row.area))

    num_workers = mp_args.NUM_WORKERS
    processes = []
    for worker_num in range(num_workers):
        p = mp.Process(target=mp_worker, args=(worker_num, task_queue, message_queue, mp_args.USE_CUDA))
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
            if mp_args.USE_LOGGING:
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
    
    os.remove(command_line_args.path)