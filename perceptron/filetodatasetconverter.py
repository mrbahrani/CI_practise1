import numpy as np
from math import inf


class FileToDatasetConverter:
    def __init__(self):
        self.dataset = list

    def convert(self, filename):
        initialDataset = np.loadtxt(filename, dtype='float', delimiter=',')
        initialDataset = initialDataset.tolist()
        dataset = list()
        for itr in initialDataset:
            dataset.append(itr[:2]+[1.0]+[2*itr[-1]-1])
        self.dataset = dataset
        return self.dataset

    def normalizedDataset(self):
        normalizedDataset = list()
        for itr in self.dataset:
            normalizedDataset.append(itr[:])
        numberOfAtts = len(self.dataset[0])-1
        maximums = [-inf for ctr in range(numberOfAtts)]
        minimums = [inf for ctr in range(numberOfAtts)]
        for attribute in range(numberOfAtts):
            for itr in normalizedDataset:
                if (itr[attribute]>maximums[attribute]): maximums[attribute] = itr[attribute]
                if (itr[attribute] < minimums[attribute]): minimums[attribute] = itr[attribute]

        for attribute in range(numberOfAtts):
            for itr in normalizedDataset:
                if not(minimums[attribute]==maximums[attribute]):
                    itr[attribute] = (itr[attribute]-minimums[attribute])/(maximums[attribute]-minimums[attribute])

        return normalizedDataset