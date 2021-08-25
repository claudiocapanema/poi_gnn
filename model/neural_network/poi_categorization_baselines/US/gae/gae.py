from .model import GCNModelAE

class GAE:

    def __init__(self, model_name):
        self.model_name = model_name
        self.build()

    def build(self):
        if self.model_name == "gae":
            GCNModelAE()
