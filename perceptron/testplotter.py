import matplotlib.pyplot as plt
from perceptron.perceptron import Perceptron
from perceptron.tester import Tester

class TestPlotter:
    def __init__(self, dataset):
        self.dataset = dataset
        self.tester = Tester()

    def setDataset(self, dataset):
        self.dataset = dataset

    def getDataset(self):
        return self.dataset

    def testPlot(self):
        size = len(self.dataset)
        step = size // 20
        start = (size % 20)+1

        for to in range(start,size,step):
            per = Perceptron([0.1 for i in range(len(self.dataset[0])-1)])
            per.train(self.dataset[:to],0.6)
            self.tester.setDataset(self.dataset)
            sr = self.tester.testPerceptron(per)
            plt.scatter(to, sr, color="black")

        plt.show()
