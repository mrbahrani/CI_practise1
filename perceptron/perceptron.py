import numpy as np
class Perceptron:
    def __init__(self, initialWeights, threshold=0, activatedThreshold=False):
        self.weights = initialWeights
        self.oldWeights = [0 for itr in range(len(initialWeights))]
        self.threshold = threshold
        self.activatedThreshold = activatedThreshold

    def setWeight(self, index, value):
        self.oldWeights[index] = self.weights[index]
        self.weights[index] = value

    def getWeight(self, index):
        return self.weights[index]

    def getThreshold(self):
        return self.threshold

    def setThreshold(self, newValue):
        self.threshold = newValue

    def activateThreshold(self, activate):
        if activate:
            self.activatedThreshold = True

    def train(self, dataset, eta=0.1):
        for record in dataset:
            xlist = record[:len(record)-1]
            check = record[-1]
            y = self.output(xlist)
            for itr in range(len(xlist)):
                self.setWeight(itr,self.getWeight(itr)+eta*xlist[itr]*(check - y))



    def output(self, xlist):
        result = np.array(self.weights).dot(np.array(xlist))
        if self.activatedThreshold:
            if result>self.threshold:
                return result
            else:
                return 0

        else:
            return result


    def binaryOutput(self, xlist):
        if self.output(xlist) >= 0:
            return 1
        else:
            return -1