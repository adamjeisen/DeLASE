import argparse 
from copy import deepcopy
import logging
import multiprocessing
import numpy as np
import os
import pandas as pd
import queue
import time
import torch
import torch.multiprocessing
import traceback
from tqdm.auto import tqdm
import sys

from delase import DeLASE
from parameter_choosing import fit_and_test_delase, ParameterGrid
from performance_metrics import get_autocorrel_funcs
from utils import load_window_from_chunks

sys.path.append('/om2/user/eisenaj/code/ChaoticConsciousness')
from data_utils import *

def mp_worker(worker_num, task_queue, message_queue=None, use_cuda=False):
    # until the task queue is empty, keep taking tasks from the queue and completing them
    while True:
        try:
            # pull a task from the queue
            task_params = task_queue.get_nowait()
            data_loading_args, window, expansion_val, autocorrel_kwargs, fit_and_test_args, T_pred, RESULTS_DIR, session, area, norm = task_params

            results_dir = os.path.join(RESULTS_DIR, os.path.join(session, 'NORMED' if norm else 'NOT_NORMED', area))

            os.makedirs(results_dir, exist_ok=True)
            save_path = os.path.join(results_dir, f"{data_loading_args['window_start']}_window_{window}_{fit_and_test_args['parameter_grid'].expansion_type}_{expansion_val}")
            if os.path.exists(save_path):
                if message_queue is not None:
                    message_queue.put((worker_num, f"{save_path} is already complete", "DEBUG"))
                    for r in fit_and_test_args['parameter_grid'].r_vals:
                        message_queue.put((worker_num, "task complete", "DEBUG"))
            else:
                if message_queue is not None:
                    message_queue.put((worker_num, f"starting {save_path}", "DEBUG"))

                data_loading_args['window_end'] = data_loading_args['window_start'] + (window + T_pred)*fit_and_test_args['dt'] + 0.1
                signal = load_window_from_chunks(**data_loading_args)
                if message_queue is not None:
                    message_queue.put((worker_num, f"data loaded!", "DEBUG"))

                if np.abs(signal).sum() < 1e-15:
                    if message_queue is not None:
                        message_queue.put((worker_num, f"data is empty - saving {save_path} as empty", "DEBUG"))
                    results = {
                        'window': [],
                        'matrix_size': [],
                        'r': [],
                        'AIC': [],
                        'stability_params': [],
                        'stability_freqs': []
                    }
                    pd.to_pickle(results, save_path)
                    for r in fit_and_test_args['parameter_grid'].r_vals:
                        message_queue.put((worker_num, "task complete", "DEBUG"))
                else:
                    if norm:
                        signal = (signal - signal.mean())/signal.std()

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
                    
                    if fit_and_test_args['compute_ip']:
                        autocorrel_true = get_autocorrel_funcs(signal[window:window + T_pred], use_torch=True, device=worker_num if use_cuda else 'cpu', **autocorrel_kwargs)
                        fit_and_test_args['autocorrel_true'] = autocorrel_true

                    fit_and_test_args['verbose'] = True
                    results = fit_and_test_delase(signal[:window], signal[window:window + T_pred], window, expansion_val, **fit_and_test_args)
                    
                    pd.to_pickle(results, save_path)

            task_queue.task_done()

        # handling what happens when the queue is found to be empty
        except queue.Empty:
            if task_queue.qsize()==0:
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
    parser.add_argument('--path', type=str, required=True,
                            help='Required path to the multiprocessing argument dictionary, pickled.')
    parser.add_argument('--job_num', type=str, required=False, default='', help='Job number from slurm.')
    command_line_args = parser.parse_args()
    job_num = command_line_args.job_num

    print(f"Now processing: {command_line_args.path}")
    mp_args = argparse.Namespace(**pd.read_pickle(command_line_args.path))

    # ----------------------
    # Set up logging
    # ----------------------
    if mp_args.USE_LOGGING:
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        os.makedirs(mp_args.LOG_DIR, exist_ok=True)
        logging.basicConfig(filename=os.path.join(mp_args.LOG_DIR,f"{mp_args.LOG_NAME}_{timestamp}" + (f"_{job_num}" if len(job_num) > 0 else job_num) + ".log"),
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
    if mp_args.USE_TORCH:
        mp = torch.multiprocessing
    else:
        mp = multiprocessing
    os.makedirs(mp_args.RESULTS_DIR, exist_ok=True)
    mp.set_start_method('spawn')
    task_queue = mp.Manager().JoinableQueue()
    message_queue = mp.Manager().JoinableQueue()

    if mp_args.COMPUTE_IP:
        if 'num_lags' in mp_args.integrated_performance_kwargs.keys():
            autocorrel_kwargs = {'num_lags': mp_args.integrated_performance_kwargs['num_lags']}
    else:
        autocorrel_kwargs = {}

    fit_and_test_args = dict(
        parameter_grid=mp_args.parameter_grid,
        dt=mp_args.dt,
        norm_aic=mp_args.NORM_AIC,
        compute_ip=mp_args.COMPUTE_IP,
        integrated_performance_kwargs=mp_args.integrated_performance_kwargs,
        compute_chroots=mp_args.COMPUTE_CHROOTS,
        stability_max_freq=mp_args.stability_max_freq,
        stability_max_unstable_freq=mp_args.stability_max_unstable_freq,
        use_torch=mp_args.USE_TORCH,
        dtype=mp_args.DTYPE,
        track_reseeds=mp_args.TRACK_RESEEDS
    )

    num_tasks = mp_args.parameter_grid.total_combinations
    if not mp_args.TRACK_RESEEDS:
        if mp_args.parameter_grid.reseed:
            num_tasks = int(num_tasks/len(mp_args.parameter_grid.reseed_vals))
    num_tasks *= len(mp_args.data_processing_df)
    
    if mp_args.parameter_grid.p_vals is not None:
        for i, row in mp_args.data_processing_df.iterrows():
            for window in mp_args.parameter_grid.window_vals:
                for p in mp_args.parameter_grid.p_vals:
                    task_queue.put((row[['window_start', 'window_end', 'directory', 'dimension_inds']], window, p, autocorrel_kwargs, fit_and_test_args, mp_args.T_pred, mp_args.RESULTS_DIR, row.session, row.area, mp_args.NORM))
    else: # mp_args.parameter_grid.matrix_size_vals is not None
        for i, row in mp_args.data_processing_df.iterrows():
            for window in mp_args.parameter_grid.window_vals:
                for matrix_size in mp_args.parameter_grid.matrix_size_vals:
                    task_queue.put((row[['window_start', 'window_end', 'directory', 'dimension_inds']], window, matrix_size, autocorrel_kwargs, fit_and_test_args, mp_args.T_pred, mp_args.RESULTS_DIR, row.session, row.area, mp_args.NORM))

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

    if mp_args.QUEUE_FULL_SESSION:
        logger.debug("Queueing full session")
        session_results = {}
        data_processing_df_grid = deepcopy(mp_args.data_processing_df)
        for session in np.unique(data_processing_df_grid.session):

            all_data_dir = '/scratch2/weka/millerlab/eisenaj/datasets/anesthesia/mat'
            data_class = get_data_class(session, all_data_dir)

            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
            variables = ['electrodeInfo', 'lfpSchema']
            session_vars, _, _, _ = load_session_data(session, all_data_dir, variables, data_class=data_class, verbose=False)
            electrode_info, lfp_schema = session_vars['electrodeInfo'], session_vars['lfpSchema']

            session_results[session] = {}
            norm_folder = "NOT_NORMED" if not mp_args.NORM else "NORMED"
            areas = os.listdir(os.path.join(mp_args.RESULTS_DIR, session, norm_folder))

            directory = data_processing_df_grid[data_processing_df_grid.session == session].directory.iloc[0]
            for area in areas:
                if area != 'all':
                    df = pd.DataFrame({'window': [], 'matrix_size': [], 'r': [], 'AICs': [], 'time_vals': [], 'file_paths': []}).set_index(['window', 'matrix_size', 'r'])
                    for f in os.listdir(os.path.join(mp_args.RESULTS_DIR, session, norm_folder, area)):
                        t = float(f.split('_')[0])
                        file_path = os.path.join(mp_args.RESULTS_DIR, session, norm_folder, area, f)
                        df_new = pd.DataFrame(pd.read_pickle(file_path))
                        if np.isnan(df_new.AIC).sum() > 0:
                            print(file_path)
                        df_new = df_new.set_index(['window', 'matrix_size', 'r'])
                        for i, row in df_new.iterrows():
                            if i in df.index:
                                df.loc[i, 'AICs'].append(row.AIC)
                                df.loc[i, 'time_vals'].append(t)
                                df.loc[i, 'file_paths'].append(file_path)
                            else:
                                df.loc[i] = {'AICs': [row.AIC], 'time_vals': [t], 'file_paths': [file_path]}
                    
                    df = df.loc[df.index.sortlevel()[0]]
                    session_results[session][area] = df

            window, matrix_size, r, all_results = combine_grid_results({key: result for key, result in session_results[session].items() if key !='all'})
            logger.debug(f"Session {session} chosen values are:")
            logger.debug(f"window = {window}")
            logger.debug(f"matrix_size = {matrix_size}")
            logger.debug(f"r = {r}")
            mp_args.RESULTS_DIR = os.path.join(os.path.dirname(mp_args.RESULTS_DIR), 'session_results')
            mp_args.parameter_grid = ParameterGrid(
                window_vals = np.array([window]),
                matrix_size_vals = np.array([matrix_size]),
                r_vals = np.array([r]),
            )
            mp_args.COMPUTE_CHROOTS = True
            mp_args.NUM_WORKERS = 4

            stride = window
            data_processing_rows = []
            for area in areas:
                results_dir = os.path.join(mp_args.RESULTS_DIR, os.path.join(session, 'NORMED' if mp_args.NORM else 'NOT_NORMED', area))
                if area != 'all':
                    if area == 'all':
                        unit_indices = np.arange(len(electrode_info['area']))
                    else:
                        unit_indices = np.where(electrode_info['area'] == area)[0]

                    t = 0
                    while t + window <= len(lfp_schema['index'][0]):
                        finished = True
                        file_path = os.path.join(results_dir, f"{int(t)/1000}_window_{window}_{mp_args.parameter_grid.expansion_type}_{matrix_size}")
                        if not os.path.exists(file_path):
                            row = dict(
                                session=session,
                                area=area,
                                window_start=t*mp_args.dt,
                                window_end=(t + window + mp_args.T_pred)*mp_args.dt,
                                directory=directory,
                                dimension_inds=unit_indices
                            )
                            data_processing_rows.append(row)
                        t += stride
            data_processing_df = pd.DataFrame(data_processing_rows)
            mp_args.data_processing_df = data_processing_df
            data_processing_path = os.path.join(os.path.dirname(command_line_args.path), f"mp_args_{session}_FULL{'_NORMED' if mp_args.NORM else ''}.pkl")
            mp_args.QUEUE_FULL_SESSION = False
            mp_args.NUM_WORKERS = 1
            pd.to_pickle(vars(mp_args), data_processing_path)

            if not mp_args.USE_CUDA:
                os.system(f"sbatch --gres=gpu:0 --ntasks=1 --cpus-per-task={int(mp_args.NUM_WORKERS) + 4} --mem={int(mp_args.NUM_WORKERS*8)}GB /om2/user/eisenaj/code/shell_scripts/DeLASE/mp_delase.sh {data_processing_path}")
            else:
                os.system(f"sbatch --gres=gpu:{mp_args.NUM_WORKERS} /om2/user/eisenaj/code/shell_scripts/DeLASE/mp_delase.sh {data_processing_path}")

    logger.info("complete!!!!")