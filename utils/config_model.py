import torch
import torch.optim as optim

from models.image_extractor import get_image_extractor
from models.graph_method import GraphFull
from models.model import CZSL


def configure_model(args, dataset):
    device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu'
    image_extractor = None
    is_open = False

    model = CZSL(dataset, args)
    model = model.to(device)

    if args.update_features:
        print('Learnable image_embeddings')
        image_extractor = get_image_extractor(arch = args.image_extractor, pretrained = True)
        image_extractor = image_extractor.to(device)
    # configuring optimizer

    model_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optim_params = [{'params':model_params}]
    if args.update_features:
        ie_parameters = [param for name, param in image_extractor.named_parameters()]
        optim_params.append({'params': ie_parameters,
                            'lr': args.lrg})
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)

    model.is_open = is_open

    return image_extractor, model, optimizer

def optimizer_builder(args,model,image_extractor):
    model_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optim_params = [{'params':model_params}]
    if args.update_features:
        ie_parameters = [param for name, param in image_extractor.named_parameters()]
        optim_params.append({'params': ie_parameters,
                            'lr': args.lrg})
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)
    return optimizer