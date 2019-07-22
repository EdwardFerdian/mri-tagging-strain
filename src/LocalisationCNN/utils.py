"""
This script mainly contains functions needed for logging
Author: Edward Ferdian
Date:   27/02/2019
"""
import logging
from datetime import datetime
from time import time

logger = logging.getLogger(__name__)

def prepare_logfile(filename):
    start = time()
    ts = datetime.fromtimestamp(start).strftime('%Y-%m-%d') #time stamp for the log file
    logname = "{}-{}.log".format(ts, filename)
    logging.basicConfig(filename=logname, level=logging.DEBUG)

def calculate_time_elapsed(start):
    '''
    This function calculates the time elapsed
    Input:  
        start = start time
    Output: 
        hrs, mins, secs = time elapsed in hours, minutes, seconds format
    '''
    end = time()
    hrs = (end-start)//60//60
    mins = ((end-start) - hrs*60*60)//60
    secs = int((end-start) - mins*60 - hrs*60*60)

    return hrs, mins, secs

def info(output_messages):
    '''
    This function logs info level messages and prints them
    Input:  
        output_messages (string or list of strings) = messages to be added to the log and printed on the terminal
    '''
    if isinstance(output_messages, str):    #if output message is a single string
        logger.info(output_messages)
        print(output_messages)
    else:    
        for message in output_messages:
            logger.info(message)
            print(message)

def error(output_messages):
    '''
    This function logs error level messages and prints them
    Input:  
        output_messages (string or list of strings) = errors to be added to the log and printed on the terminal
    '''
    if isinstance(output_messages, str):    #if error message is a single string
        logger.error(output_messages)
        print(output_messages)
    else:    
        for message in output_messages:
            logger.error(message)
            print(message)