
def create_model(opt):
    model = None
    if opt.model == 'csa_net':
        from .CSA import CSA
        model = CSA()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    return model
