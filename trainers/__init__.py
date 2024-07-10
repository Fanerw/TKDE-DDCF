from .ae_ode import AEODETrainer

TRAINERS = {
    AEODETrainer.code():AEODETrainer
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
