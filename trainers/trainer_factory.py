from .trainer import BaseTrainer, DpsgdTrainer, DpsgdGlobalTrainer, DpsgdGlobalAdaptiveTrainer, \
    DpsgdAutoClipTrainer, DPSGDAugmentedDataTrainer, RegularAugmentedDataTrainer, DpsgdSmoothAdvTrainer, \
        DpsgdAutoClipSmoothAdvTrainer, DPSGDAdvTrainer, DpsgdAutoClipAugmentTrainer

def create_trainer(
        train_loader,
        valid_loader,
        test_loader,
        model,
        optimizer,
        evaluator,
        privacy_engine,
        writer,
        device,
        config
):
    kwargs = {
        'method': config['method'],
        'max_epochs': config['max_epochs'],
        'lr': config['lr'],
        'seed': config['seed'],
        'evaluate_adversarial_loss': config['evaluate_adversarial_loss'],
        'physical_batch_size': config['physical_batch_size'],
        'clip': config['clip'],
        'add_noise': config['add_noise']
    }

    if config["method"] == "regular":
        trainer = BaseTrainer(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            **kwargs
        )
    elif config["method"] == "regular-augment":
        trainer = RegularAugmentedDataTrainer(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            num_augmentations=config["num_augmentations"],
            **kwargs
        )
    elif config["method"] == "dpsgd":
        trainer = DpsgdTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            **kwargs
        )
    elif config["method"] == "dpsgd-adv":
        trainer = DPSGDAdvTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            pgd_steps=config["pgd_steps"],
            max_norm=config["max_norm"],
            warmup=config["warmup"],
            **kwargs
        )
    elif config["method"] == "dpsgd-global":
        trainer = DpsgdGlobalTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            strict_max_grad_norm=config["strict_max_grad_norm"],
            **kwargs
        )
    elif config["method"] == "dpsgd-global-adapt":
        trainer = DpsgdGlobalAdaptiveTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            strict_max_grad_norm=config["strict_max_grad_norm"],
            bits_noise_multiplier=config["bits_noise_multiplier"],
            lr_Z=config["lr_Z"],
            threshold=config["threshold"],
            **kwargs
        )
    elif config["method"] == "dpsgd-auto-clip":
        trainer = DpsgdAutoClipTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            gamma=config["gamma"],
            psac=config["psac"],
            **kwargs
        )
    elif config["method"] == "dpsgd-augment":
        trainer = DPSGDAugmentedDataTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            num_augmentations=config["num_augmentations"],
            augment_noise_std=config["augment_noise_std"],
            consistency=config["consistency"],
            macer=config["macer"],
            stability=config["stability"],
            trades=config["trades"],
            **kwargs
        )
    elif config["method"] == "dpsgd-adv-smooth":
        trainer = DpsgdSmoothAdvTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            pgd_steps=config["pgd_steps"],
            max_norm=config["max_norm"],
            num_augmentations=config["num_augmentations"],
            augment_noise_std=config["augment_noise_std"],
            warmup=config["warmup"],
            no_grad=config["no_grad"],
            include_original=config["include_original"],
            consistency=config["consistency"],
            trades=config["trades"],
            stability=config["stability"],
            **kwargs
        )
    elif config["method"] == "dpsgd-auto-clip-adv-smooth":
        trainer = DpsgdAutoClipSmoothAdvTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            pgd_steps=config["pgd_steps"],
            max_norm=config["max_norm"],
            num_augmentations=config["num_augmentations"],
            augment_noise_std=config["augment_noise_std"],
            warmup=config["warmup"],
            gamma=config["gamma"],
            psac=config["psac"],
            no_grad=config["no_grad"],
            include_original=config["include_original"],
            consistency=config["consistency"],
            trades=config["trades"],
            stability=config["stability"],
            **kwargs
        )
    elif config["method"] == "dpsgd-augment-auto-clip":
        trainer = DpsgdAutoClipAugmentTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            num_augmentations=config["num_augmentations"],
            augment_noise_std=config["augment_noise_std"],
            gamma=config["gamma"],
            psac=config["psac"],
            consistency=config["consistency"],
            trades=config["trades"],
            stability=config["stability"],
            **kwargs
        )
    else:
        raise ValueError("Training method not implemented")
    try:
        trainer.load_checkpoint("latest")
    except FileNotFoundError:
        print("Did not find checkpoint to load model from.")

    return trainer
