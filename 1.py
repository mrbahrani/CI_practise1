#And Implementation
#Third attribute is bias (always one)
from perceptron.perceptron import Perceptron
from perceptron.tester import Tester

# The weights are set manually.
perAnd = Perceptron([0.5,0.5,-0.6])

# All states are tested.
print(perAnd.binaryOutput([-1,-1,1]))
print(perAnd.binaryOutput([-1,1,1]))
print(perAnd.binaryOutput([1,-1,1]))
print(perAnd.binaryOutput([1,1,1]))

# Tester shows the success rate of the classification. (In this easy case it is 1.0 (100%).
tester = Tester()
tester.setDataset([[-1,-1,1,-1],[-1,1,1,-1],[1,-1,1,-1],[1,1,1,1]])
print("Success rate:",tester.testPerceptron(perAnd))