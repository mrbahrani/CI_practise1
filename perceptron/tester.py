class Tester:
    def __init__(self):
        self.dataset = list()

    def setDataset(self, dataset):
        self.dataset = dataset

    def getDataset(self):
        return self.dataset

    def testPerceptron(self, perceptron):
        success = 0
        for itr in self.dataset:
            res = perceptron.binaryOutput(itr[:len(itr)-1])
            if res == itr[-1]:
                success += 1
        return success/len(self.dataset)
