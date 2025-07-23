import _warnings
import copy
from typing import List, Any

from util.dataclass import copyfield


try:
    from dataclasses import dataclass, field as datafield


    def copyfield(data):
        return datafield(default_factory=lambda: copy.deepcopy(data))
except ModuleNotFoundError:
    _warnings.warn('dataclasses not found. To get it, use Python 3.7 or pip install dataclasses')


@dataclass
class DEMO:
    demo: List[Any] = copyfield([])
    updated: bool = False