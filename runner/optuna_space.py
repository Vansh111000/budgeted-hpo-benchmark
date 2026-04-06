# runner/optuna_space.py
def suggest_from_configspace(trial, cs):
    cfg = {}
    for hp in cs.get_hyperparameters():
        name = hp.name
        cls = hp.__class__.__name__
        # categorical
        if hasattr(hp, "choices"):
            cfg[name] = trial.suggest_categorical(name, list(hp.choices))
            continue
        # float
        if hasattr(hp, "lower") and hasattr(hp, "upper") and "Float" in cls:
            log = "Log" in cls
            cfg[name] = trial.suggest_float(name, float(hp.lower), float(hp.upper), log=log)
            continue
        # int
        if hasattr(hp, "lower") and hasattr(hp, "upper") and "Integer" in cls:
            cfg[name] = trial.suggest_int(name, int(hp.lower), int(hp.upper))
            continue
        raise ValueError(f"Unsupported hyperparameter type: {cls} ({hp})")
    return cfg