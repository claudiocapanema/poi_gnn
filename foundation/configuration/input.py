from foundation.abs_classes.singleton import Singleton

class Input(Singleton):

    inputs = dict()
    def set_inputs(cls, inputs):
        cls.get_instance().inputs = inputs


    @classmethod
    def get_arg(cls, arg):
        if arg not in cls.get_instance().inputs.sub_category():
            return None
        return cls.get_instance().inputs[arg]

    @classmethod
    def get_args(cls):
        return cls.get_instance().inputs