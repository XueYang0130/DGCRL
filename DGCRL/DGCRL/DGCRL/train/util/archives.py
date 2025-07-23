from typing import Dict, Any
from util.dataclass import CellInfo
import numpy as np

near_zero_1 = 1e-3
near_zero_2 = 1e-6


class DeterministicArchive:
    def __init__(self):
        self.archive: Dict[Any, Dict[Any, Any]] = {}

    def update_cell(self, utility_key, cell_key, cell_info: CellInfo):
        """
        This method add a cell to the archive. If there is no such cell in the archive, add the cell.

        :param cell_key: the cell to be updated
        :param cell_info: the info to update
        :return:
        """
        if utility_key not in self.archive.keys():
            self.archive[utility_key] = {}
        if cell_key in self.archive[utility_key].keys():
            self.archive[utility_key][cell_key].num_of_visit += 1

            if self.archive[utility_key][cell_key].score < cell_info.score:
                self.archive[utility_key][cell_key].cell_traj = cell_info.cell_traj
                self.archive[utility_key][cell_key].score = cell_info.score

                for cell_key in list(self.archive[utility_key].keys()):
                    cell = self.archive[utility_key][cell_key]
                    cell.num_of_visit = 1

        else:  # new trajectory, update archive
            self.add_cell(utility_key, cell_key)
            cell_info = CellInfo(cell_traj=cell_info.cell_traj,
                                 pos_traj=cell_info.pos_traj,
                                 num_of_visit=1,
                                 score=cell_info.score,
                                 reward_vec=cell_info.reward_vec,
                                 terminal=cell_info.terminal)
            self.archive[utility_key][cell_key] = cell_info

    def get_state(self):
        raise NotImplementedError('get_state not implemented')

    def get_new_cell_info(self):
        return CellInfo()

    def add_cell(self, utility_key, cell_key: Any):
        cell = self.get_new_cell_info()
        self.archive[utility_key][cell_key] = cell


if __name__ == '__main__':
    deterministicArchive = DeterministicArchive()
    archive = deterministicArchive.archive
    archive[(0, 1)] = {(0, 2): CellInfo(cell_traj=([(0, 2)]), num_of_visit=1, score=5),
                       (0, 3): CellInfo(cell_traj=([(9, 3)]), num_of_visit=1, score=10)}
    for k in archive.keys():
        for c in archive[k].keys():
            print(f"k:{k}\t")
            print(f"c:{c}\t")
            print(f"v:{archive[k][c]}")
    deterministicArchive.update_cell(utility_key=(0, 1), cell_key=(0, 2),
                                     cell_info=CellInfo(cell_traj=([(1, 2), (2, 3)]), num_of_visit=1, score=100))
    for k in archive.keys():
        for c in archive[k].keys():
            print(f"k:{k}\t")
            print(f"c:{c}\t")
            print(f"v:{archive[k][c]}")
    deterministicArchive.update_cell(utility_key=(0, 1), cell_key=(0, 2),
                                     cell_info=CellInfo(cell_traj=([(1, 2), (2, 3)]), num_of_visit=1, score=100))
    for k in archive.keys():
        for c in archive[k].keys():
            print(f"k:{k}\t")
            print(f"c:{c}\t")
            print(f"v:{archive[k][c]}")
