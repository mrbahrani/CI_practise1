from matplotlib import pyplot as plt

class PeceptronPlotter:
    def __init__(self, perceptron = None, dataset= None):
        self.perceptron = perceptron
        self.dataset = dataset

    def setPeceptron(self, per):
        self.perceptron = per

    def getPerceptron(self):
        return self.perceptron

    def setDataset(self, dataset):
        self.dataset = dataset

    def getDataset(self):
        return self.dataset

    def show(self):
        if len(self.perceptron.weights)==3:
            x = -self.perceptron.weights[2] / self.perceptron.weights[1]
            y = -self.perceptron.weights[2] / self.perceptron.weights[0]
            plt.plot([0,x],[y,0])
            for itr in self.dataset:
                if itr[-1] ==1:
                    plt.scatter(itr[0], itr[1], None, color="green")
                else:
                    plt.scatter(itr[0], itr[1], None, color="red")
            plt.show()
        else:
            print("Not drawable!")