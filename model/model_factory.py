# multi-path paradigm
# from model.clip_multi_path import CLIP_Multi_Path
# from model.coop_multi_path import COOP_Multi_Path
from model.cia import CIA

def get_model(config, attributes, classes, offset):
    if config.model_name == 'CIA':
        model = CIA(config, attributes=attributes, classes=classes, offset=offset)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(
                config.model_name
            )
        )


    return model