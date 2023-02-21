import logging
import os
import queue
import time
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
import sys

def mp_worker(worker_num, task_queue, message_queue=None, use_cuda=False):
    # until the task queue is empty, keep taking tasks from the queue and completing them
    while True:
        try:
            # pull a task from the queue
            task_params = task_queue.get_nowait()
            i = task_params[0]

            # our current task here is to send a message with the value the worker has and then do nothing for 3 seconds
            # replace this with whatever task you're trying to multiprocess
            if message_queue is not None:
                message_queue.put((worker_num, f"I have the value {i}", "DEBUG"))

            if use_cuda:
                tensor = torch.randn(100, 100).to(worker_num)
            else:
                tensor = torch.randn(100, 100).cpu()
            if message_queue is not None:
                message_queue.put((worker_num, f"tensor is on {tensor.device}", "INFO"))
            time.sleep(3)

            # send a message that the task is complete
            if message_queue is not None:
                message_queue.put((worker_num, "task complete", "DEBUG"))
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

    USE_CUDA = True
    NUM_WORKERS = 2

    USE_LOGGING = True
    LOG_DIR = "."
    LOG_NAME = 'mp_test'
    LOG_LEVEL = logging.DEBUG
    
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
    task_queue = mp.Manager().JoinableQueue()
    message_queue = mp.Manager().JoinableQueue()
    num_tasks = 75
    for i in range(num_tasks):
        task_queue.put((i, ))

    num_workers = NUM_WORKERS
    # if USE_CUDA:
        # num_workers = torch.cuda.device_count()
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