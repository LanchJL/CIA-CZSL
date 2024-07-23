#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
import datetime as dt
from data import dataset as dset
from models.common import Evaluator
from utils.utils import save_args, load_args
from utils.config_model import configure_model,optimizer_builder
from flags import parser, DATA_FOLDER
import random
import numpy as np

best_auc = 0
best_hm = 0
best_attr = 0
best_obj = 0
best_seen = 0
best_unseen = 0
latest_changes = 0
compose_switch = True


def main():
    # Get arguments and start logging
    now_time = dt.datetime.now().strftime('%F %T')
    print('-----------{}-----------'.format(now_time))
    args = parser.parse_args()
    load_args(args.config, args)
    logpath = os.path.join(args.cv_dir, args.name+'_'+now_time)
    os.makedirs(logpath, exist_ok=True)
    save_args(args, logpath, args.config)

    device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu'
    # random_seed = random.randint(0, 10000)
    random_seed = 3407
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    print('-----------Using Dataset: {}-----------'.format(args.dataset))
    print('-----------Update Features: {}-----------'.format(args.update_features))
    print('-----------Word Embedding: {}-----------'.format(args.emb_init))
    print('-----------eta: {}-----------'.format(args.eta))
    print('-----------tau: {}-----------'.format(args.tem))
    print('Loading Models......')

    # Get dataset
    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        split=args.splitname,
        model =args.image_extractor,
        num_negs=args.num_negs,
        pair_dropout=args.pair_dropout,
        update_features = args.update_features,
        train_only= args.train_only,
        open_world=args.open_world
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)
    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase=args.test_set,
        split=args.splitname,
        model =args.image_extractor,
        subset=args.subset,
        update_features = args.update_features,
        open_world=args.open_world
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)
    args.extractor = image_extractor
    train = train_normal
    evaluator_val = Evaluator(testset, model)
    start_epoch = 0

    model.freeze_model()
    print('-----------Train with Frozen Py, Vy-----------')
    for epoch in tqdm(range(start_epoch, 20), desc = 'Current epoch'):
        optimizer = optimizer_builder(args, model, image_extractor)
        train(epoch, image_extractor, model, trainloader, optimizer, device, prior = True)
        if epoch % args.eval_val_every == 0:
            with torch.no_grad(): # todo: might not be needed
                test(epoch, image_extractor, model, testloader, evaluator_val, args, device)

    best_val_auc = 0
    model.unfreeze_model()
    print('-----------Train All Layers-----------')
    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc = 'Current epoch'):
        model.unfreeze_model()
        optimizer = optimizer_builder(args, model, image_extractor)
        model._init_cliques()
        train(epoch, image_extractor, model, trainloader, optimizer, device, prior = False)
        if epoch % args.eval_val_every == 0:
            with torch.no_grad():  # todo: might not be needed
                AUC = test(epoch, image_extractor, model, testloader, evaluator_val, args, device)

        if AUC > best_val_auc:
            best_val_auc = AUC
            embedding_save_path = os.path.join(logpath, 'Best_AUC_Embedding.pth')
            torch.save(model.state_dict(), embedding_save_path)

    print(
        f'AUC: {best_auc * 100:.2f} HM: {best_hm * 100:.2f} S: {best_seen * 100:.2f} U: {best_unseen * 100:.2f}')

def mean_nonzero(matrix):
    non_zero_mask = matrix != 0
    mean_values = np.sum(matrix * non_zero_mask, axis=0) / np.sum(non_zero_mask, axis=0)
    return mean_values

def train_normal(epoch, image_extractor, model, trainloader, optimizer, device, prior):
    '''
    Runs training for an epoch
    '''

    if image_extractor:
        image_extractor.train()
    model.train() # Let's switch to training

    train_loss = 0.0
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc = 'Training'):
        data  = [d.to(device) for d in data]
        if image_extractor:
            data[0] = image_extractor(data[0])
        loss, poster = model(data, prior)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if prior:
            poster_p_yx = poster['p_yx']
            poster_p_cx = poster['p_cx']
            if idx == 0:
                pred_attr = poster_p_yx[0].detach().cpu().numpy()
                pred_objs = poster_p_yx[1].detach().cpu().numpy()

                _pred_attr = poster_p_cx[0].detach().cpu().numpy()
                _pred_objs = poster_p_cx[1].detach().cpu().numpy()
            else:
                pred_attr = np.concatenate((pred_attr, poster_p_yx[0].detach().cpu().numpy()))
                pred_objs = np.vstack([pred_objs, poster_p_yx[1].detach().cpu().numpy()])

                _pred_attr = np.concatenate((_pred_attr, poster_p_cx[0].detach().cpu().numpy()))
                _pred_objs = np.vstack([_pred_objs, poster_p_cx[1].detach().cpu().numpy()])

        train_loss += loss.item()
    train_loss = train_loss/len(trainloader)
    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))

    if prior:
        pred_attr = mean_nonzero(pred_attr)
        pred_objs = mean_nonzero(pred_objs)
        _pred_attr = mean_nonzero(_pred_attr)
        _pred_objs = mean_nonzero(_pred_objs)
        model.prob['attr'] = torch.from_numpy(_pred_attr)
        model.prob['objs'] = torch.from_numpy(_pred_objs)
        model.prob['attr_y'] = torch.from_numpy(pred_attr)
        model.prob['objs_y'] = torch.from_numpy(pred_objs)
        print('update poster probilities')

def test(epoch, image_extractor, model, testloader, evaluator, args, device):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm, best_obj,best_attr,best_seen,best_unseen,latest_changes
    if image_extractor:
        image_extractor.eval()

    model.eval()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) for d in data]
        if image_extractor:
            data[0] = image_extractor(data[0])
        predictions = model(data)

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)

    stats['a_epoch'] = epoch
    print(f'Test Epoch: {epoch}')
    if stats['AUC'] > best_auc or stats['best_hm'] > best_hm or stats['best_seen'] > best_seen or stats['best_unseen'] > best_unseen:
        if stats['AUC'] > best_auc:
            best_auc = stats['AUC']
        if stats['best_hm'] > best_hm:
            best_hm = stats['best_hm']
        if stats['best_seen'] > best_seen:
            best_seen = stats['best_seen']
        if stats['best_unseen'] > best_unseen:
            best_unseen = stats['best_unseen']
        latest_changes = epoch
        print(
            f'AUC: {stats["AUC"]*100:.2f} HM: {stats["best_hm"]*100:.2f} S: {stats["best_seen"]*100:.2f} U: {stats["best_unseen"]*100:.2f}')
    return stats['AUC']
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(
            f'AUC: {best_auc*100:.2f} HM: {best_hm*100:.2f} S: {best_seen*100:.2f} U: {best_unseen*100:.2f}')
        print('Latest Improved Epoch is', latest_changes)