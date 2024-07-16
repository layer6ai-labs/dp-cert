import foolbox as fb

def create_attack(model, name, kwargs=None):
    fmodel = fb.PyTorchModel(model, bounds=(0,1))
    if name.lower() == "fgsm": 
        return fmodel, fb.attacks.FGSM(), "linf"
    elif name.lower() == "pgd":
        return fmodel, fb.attacks.PGD(rel_stepsize=kwargs.get("PGD_stepsize", 2/255) , 
                                        steps=kwargs.get("PGD_iterations", 40)),  "linf"
    elif name.lower() == "pgdl2":
        return fmodel, fb.attacks.L2PGD(rel_stepsize=kwargs.get("PGDL2_stepsize", 0.2),
                                        steps=kwargs.get("PGDL2_iterations", 40)) , "l2"
    elif name.lower() == "cw":
        return fmodel, fb.attacks.L2CarliniWagnerAttack(binary_search_steps=kwargs.get("CW_c_iterations", 9), 
                                                        steps=kwargs.get("CW_iterations", 10000), 
                                                        stepsize=kwargs.get("CW_stepsize", 0.01), 
                                                        confidence=kwargs.get("CW_confidence", 0), 
                                                        initial_const=kwargs.get("CW_c", 1)), "l2" # abort_early=True
    elif name.lower() == "deepfooll2":
        return fmodel, fb.attacks.L2DeepFoolAttack(steps=kwargs.get("DeepFool_iterations", 50),
                                                    overshoot=kwargs.get("DeepFool_overshoot", 0.02),
                                                    loss=kwargs.get("DeepFool_loss", 'logits')), "l2" # loss='crossentropy'
    elif name.lower() == "deepfoollinf":
        return fmodel, fb.attacks.LinfDeepFoolAttack(steps=kwargs.get("DeepFool_iterations", 50),
                                                    overshoot=kwargs.get("DeepFool_overshoot", 0.02),
                                                    loss=kwargs.get("DeepFool_loss", 'logits')),  "linf" # loss='crossentropy'
    elif name.lower() == "boundary":
        return fmodel, fb.attacks.BoundaryAttack(init_attack=fb.attacks.L2FMNAttack(), # SaltAndPepperNoiseAttack()
                                                steps=kwargs.get("BA_iterations", 25000)), "l2"
    
    # HopSkipJumpAttack, PointwiseAttack
    

# TODO: FAB
