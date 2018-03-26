import sys
from perceptron import perceptron
from perceptron.perceptron import Perceptron
from perceptron.filetodatasetconverter import FileToDatasetConverter
from perceptron.tester import Tester
from perceptron.perceptronplotter import PeceptronPlotter
from perceptron.testplotter import TestPlotter

# At first, the weights are arbitrary.
per = Perceptron([0.1, 0.1, 0.1])

# File converter turns file into a dataset suitable for training
fileConverter = FileToDatasetConverter()
fileConverter.convert(sys.argv[1])

# We use a normalized dataset. Because otherwise bias effect goes wrong.
# Later, we can change it to the main dataset.
dataset = fileConverter.normalizedDataset()
per.train(dataset, 0.6)
per.train(dataset, 0.6)
tester = Tester()
tester.setDataset(dataset)
print("Success rate: ", tester.testPerceptron(per))

# The first diagram is the decision line.
plotter = PeceptronPlotter(per,dataset)
plotter.show()

# This diagram shows the effect of training on a perceptron.
t = TestPlotter(dataset)
t.testPlot()