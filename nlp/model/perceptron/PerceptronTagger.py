from nlp.model.perceptron.InstanceConsumer import InstanceConsumer
from nlp.model.perceptron.model.StructurePerceptron import StructurePerceptron

class PerceptronTagger(InstanceConsumer):
    def __init__(self, model = None):
        print(model)
        assert model != None, "模型不能为空"

        self.model = model if isinstance(model, StructurePerceptron) else StructurePerceptron(model.featureMap, model.parameter)

    def getModel(self):
        return self.model


