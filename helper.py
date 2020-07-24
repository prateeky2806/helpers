import numpy as np, sys, os, random, ipdb as pdb, json, uuid, time, argparse, pickle, copy
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict
from collections.abc import Iterable
from ordered_set import OrderedSet
from matplotlib import pyplot as plt
import functools, itertools, operator
from itertools import product
from joblib import Parallel, delayed

# PyTorch related imports
import torch
from torch.nn import functional as F
import torch.distributions as tdist
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter

np.set_printoptions(precision=4)

def iterable_(obj):
    '''Check if the input object is an iterable or not'''
    return isinstance(obj, Iterable)

def timer(func):
	"""Print the runtime of the decorated function"""
	@functools.wraps(func)
	def wrapper_timer(*args, **kwargs):
		start_time = time.perf_counter()  # 1
		value = func(*args, **kwargs)
		end_time = time.perf_counter()  # 2
		run_time = end_time - start_time  # 3
		print(f"Finished {func.__name__!r} in {run_time:.6f} secs")
		return value
	return wrapper_timer


def set_gpu(gpus):
	"""
	Sets the GPU to be used for the run

	Parameters
	----------
	gpus:           List of GPUs to be used for the run
	
	Returns
	-------
		
	"""
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def checkFile(filename):
	"""
	Check whether file is present or not
	Parameters
	----------
	filename:       Path of the file to check
	
	Returns
	-------
	"""
	return pathlib.Path(filename).is_file()

def make_dir(dir_path):
	"""
	Creates the directory if doesn't exist
	Parameters
	----------
	dir_path:       Path of the directory
	
	Returns
	-------
	"""
	if not os.path.exists(dir_path): 
		os.makedirs(dir_path)

def get_logger(name, log_dir, config_dir):
	"""
	Creates a logger object

	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
		
	"""
	config_dict = json.load(open( config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def getChunks(inp_list, chunk_size):
	"""
	Splits inp_list into lists of size chunk_size
	Parameters
	----------
	inp_list:       List to be splittted
	chunk_size:     Size of each chunk required
	
	Returns
	-------
	chunks of the inp_list each of size chunk_size, last one can be smaller (leftout data)
	"""
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def partition(inp_list, n):
	"""
	Paritions a given list into chunks of size n
	Parameters
	----------
	inp_list:       List to be splittted
	n:     		Number of equal partitions needed
	
	Returns
	-------
	Splits inp_list into n equal chunks
	"""
	division = len(inp_list) / float(n)
	return [ inp_list[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def mergeList(list_of_list):
	"""
	Merges list of list into a list
	Parameters
	----------
	list_of_list:   List of list
	
	Returns
	-------
	A single list (union of all given lists)
	"""
	return list(itertools.chain.from_iterable(list_of_list))
