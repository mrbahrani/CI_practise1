#And Implementation by training
#Third attribute is bias (always one)

from perceptron.perceptron import Perceptron
from perceptron.tester import Tester

# Initial weights are way off.
perAnd = Perceptron([0.3, -0.2, 0.6])

# The perceptron is not working before training
print("Before Training:")
print(perAnd.binaryOutput([-1, -1, 1]))
print(perAnd.binaryOutput([-1, 1, 1]))
print(perAnd.binaryOutput([1, -1, 1]))
print(perAnd.binaryOutput([1, 1, 1]))

# As the states are limited, we use them several times to train the perceptron.
trainingDataset = [[-1,-1,1,-1],[-1,1,1,-1],[1,-1,1,-1],[1,1,1,1],
                   [-1, -1, 1, -1], [-1, 1, 1, -1], [1, -1, 1, -1], [1, 1, 1, 1],
                   [-1, -1, 1, -1], [-1, 1, 1, -1], [1, -1, 1, -1], [1, 1, 1, 1],
                   [-1, -1, 1, -1], [-1, 1, 1, -1], [1, -1, 1, -1], [1, 1, 1, 1],
                   [-1, -1, 1, -1], [-1, 1, 1, -1], [1, -1, 1, -1], [1, 1, 1, 1],
                   [-1, -1, 1, -1], [-1, 1, 1, -1], [1, -1, 1, -1], [1, 1, 1, 1]]
perAnd.train(trainingDataset)

print("After Training:")
print("w1: %f, w2: %f, bias: %f" %(perAnd.getWeight(0), perAnd.getWeight(1), perAnd.getWeight(2)))
print(perAnd.binaryOutput([-1, -1, 1]))
print(perAnd.binaryOutput([-1, 1, 1]))
print(perAnd.binaryOutput([1, -1, 1]))
print(perAnd.binaryOutput([1, 1, 1]))

tester = Tester()
tester.setDataset([[-1,-1,1,-1],[-1,1,1,-1],[1,-1,1,-1],[1,1,1,1]])
print("Success rate: ",tester.testPerceptron(perAnd))