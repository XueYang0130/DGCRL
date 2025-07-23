import warnings as _warnings
import copy
from typing import List, Any
from collections import deque
import sys

try:
    from dataclasses import dataclass, field as datafield


    def copyfield(data):
        return datafield(default_factory=lambda: copy.deepcopy(data))
except ModuleNotFoundError:
    _warnings.warn('dataclasses not found. To get it, use Python 3.7 or pip install dataclasses')


@dataclass
class DemoInfo:
    demo: List[Any] = copyfield([])
    horizon: int = 0
    disc_ret: List[Any] = copyfield([])
    updated: bool = False
    key: int = 0

if __name__ == '__main__':
    d = DemoInfo()
    print(d.demo)