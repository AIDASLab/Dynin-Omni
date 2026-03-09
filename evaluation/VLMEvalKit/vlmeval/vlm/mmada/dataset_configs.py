
DEFAULT_KWARGS = {
    "max_new_tokens": 2,
    "steps": 2,
    "block_length": 2,
}

DATASET_CONFIGS = {
    "MathVista_MINI": {
        "max_new_tokens": 96,
        "steps": 96,
        "block_length": 48,
    },
    
    "MathVerse_MINI_Vision_Only": {
        "max_new_tokens": 256,
        "steps": 128,
        "block_length": 32,
    },
    
    "MMVet": {
        "max_new_tokens": 512,
        "steps": 256,
        "block_length": 128,
    },
}


def get_dataset_config(dataset_name):
    return DATASET_CONFIGS.get(dataset_name, {})


def merge_configs(*configs):
    result = DEFAULT_KWARGS.copy()
    for config in configs:
        if config:
            result.update(config)
    return result 