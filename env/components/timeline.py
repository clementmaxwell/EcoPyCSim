# inspired by Decima++: https://github.com/ArchieGertsman/spark-sched-sim

import heapq
import itertools
from enum import Enum, auto
from dataclasses import dataclass

@dataclass
class TimelineEvent:
  class Type(Enum):
    JOB_ARRIVAL = auto()
    TASK_ARRIVAL = auto()
    TASK_DEPARTURE = auto()
    
  type: Type
  data: dict
 
# heap-based timeline
class Timeline:
  def __init__(self):
    self.priority_queue = []
    
    self.counter = itertools.count()
  
  def __len__(self):
    return len(self.priority_queue)
  
  @property
  def empty(self):
    return len(self) == 0
  
  def peek(self):
    try:
      key, _, item = self.priority_queue[0]
      return key, item
    except IndexError:
      return None, None
  
  def push(self, key, item):
    heapq.heappush(self.priority_queue, (key, next(self.counter), item))
  
  def pop(self):
    if len(self.priority_queue) > 0:
      key, _, item = heapq.heappop(self.priority_queue)
      return key, item
    else:
      return None, None
  
  def reset(self):
    self.priority_queue.clear()
    self.counter = itertools.count()
  
  def events(self):
    return (event for (*_, event) in self.priority_queue)
  
  def print_queue(self):
    for key, count, item in self.priority_queue:
      print(f"Key: {key}, Count: {count}, Item: {item}")