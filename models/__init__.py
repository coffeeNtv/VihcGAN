from .vihc import VIHC_model
def create_model(opt):
    model = None
    model = VIHC_model()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
