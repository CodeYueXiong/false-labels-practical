import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import mlflow

from nets.ssd import get_ssd
from nets.ssd_training import LossHistory, MultiBoxLoss, weights_init
from utils.config import Config
from utils.dataloaders import SSDDataset, ssd_dataset_collate

warnings.filterwarnings("ignore")

log_flag = True

if log_flag:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/yue.xiong@stat.uni-muenchen.de/ssd300_w_false")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, criterion, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    loc_loss = 0
    conf_loss = 0
    loc_loss_val = 0
    conf_loss_val = 0

    net.train()
    print('Start Train')
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            # ----------------------#
            #   Forward propagation
            # ----------------------#
            out = net(images)
            # ----------------------#
            #   Clear gradient
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   Calculate the loss
            # ----------------------#
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            # ----------------------#
            #   Backpropagation
            # ----------------------#
            loss.backward()
            optimizer.step()

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            # mlflow.log_metric(key="ssd300_train_loss", value=loss_l.item()+loss_c.item(), step=epoch)
            # log train loss with mlflow
            # if iteration % 10 == 0:
            #   mlflow.log_metric(key="ssd300_train_loc_loss", value=loss_l.item())
            #   mlflow.log_metric(key="ssd300_train_conf_loss", value=loss_c.item())

            pbar.set_postfix(**{'loc_loss': loc_loss / (iteration + 1),
                                'conf_loss': conf_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    # evaluation
    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

                out = net(images)
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)

                loc_loss_val += loss_l.item()
                conf_loss_val += loss_c.item()

                # mlflow.log_metric(key="ssd300_val_loss", value=loss_l.item()+loss_c.item(), step=epoch)
                # log val loss with mlflow
                # if iteration % 10 == 0:
                #   mlflow.log_metric(key="ssd300_val_loc_loss", value=loss_l.item())
                #   mlflow.log_metric(key="ssd300_val_conf_loss", value=loss_c.item())

                pbar.set_postfix(**{'loc_loss': loc_loss_val / (iteration + 1),
                                    'conf_loss': conf_loss_val / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    total_loss = loc_loss + conf_loss
    val_loss = loc_loss_val + conf_loss_val
    # mlflow log with train and val loss
    mlflow.log_metric(key="ssd300_train_loss", value=total_loss / (epoch_size + 1), step=epoch)
    mlflow.log_metric(key="ssd300_val_loss", value=val_loss / (epoch_size_val + 1), step=epoch)

    loss_history.append_loss(total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    print('Saving state, iter:', str(epoch + 1))

    torch.save(model.state_dict(), 'logs_ssd300_update/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    return val_loss / (epoch_size_val + 1)


if __name__ == "__main__":
    # -------------------------------#
    #   Whether to use Cuda
    # -------------------------------#
    Cuda = True
    # --------------------------------------------#
    #   Implemented ssd based on mobilenetv2 and vgg respectively
    #   The backbone network can be selected by modifying the backbone variable
    #   vgg or mobilenet
    # ---------------------------------------------#
    backbone = "vgg"

    model = get_ssd("train", Config["num_classes"], backbone)
    weights_init(model)
    # ------------------------------------------------------#
    #   For weight file, please download in advance.
    # ------------------------------------------------------#
    model_path = "model_data/ssd_weights.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    with open(train_annotation_path) as train_f:
        train_lines = train_f.readlines()
    num_train = len(train_lines)

    with open(val_annotation_path) as val_f:
        val_lines = val_f.readlines()
    num_val = len(val_lines)

    # val_split = 0.1
    # with open(annotation_path) as f:
    #     lines = f.readlines()
    # np.random.seed(10101)
    # np.random.shuffle(lines)
    # np.random.seed(None)
    # num_val = int(len(lines)*val_split)
    # num_train = len(lines) - num_val

    criterion = MultiBoxLoss(Config['num_classes'], 0.5, True, 0, True, 3, 0.5, False, Cuda)
    loss_history = LossHistory("logs_ssd300/")

    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # ------------------------------------------------------#
    #   The main feature extraction network feature is general, freezing training can speed up the training speed
    #   It can also prevent the weights from being destroyed in the early stage of training.
    #   Init_Epoch is the initial epochs setting
    #   Freeze_Epoch is the epochs of freeze training
    #   Unfreeze_Epoch is the total training epochs
    #   Prompt OOM or insufficient memory, please reduce Batch_size
    # ------------------------------------------------------#
    with mlflow.start_run(run_name="ssd300 w falselabels Colab log_epoch"):
        if True:
            lr = 5e-4
            Batch_size = 32
            Init_Epoch = 0
            Freeze_Epoch = 50

            optimizer = optim.Adam(net.parameters(), lr=lr)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7,
                                                                verbose=True)

            train_dataset = SSDDataset(train_lines[:num_train], (Config["min_dim"], Config["min_dim"]), True)
            val_dataset = SSDDataset(val_lines[:num_val], (Config["min_dim"], Config["min_dim"]), False)

            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=ssd_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=ssd_dataset_collate)

            if backbone == "vgg":
                for param in model.vgg.parameters():
                    param.requires_grad = False
            else:
                for param in model.mobilenet.parameters():
                    param.requires_grad = False

            epoch_size = num_train // Batch_size
            epoch_size_val = num_val // Batch_size

            if epoch_size == 0 or epoch_size_val == 0:
                raise ValueError("The data set is too small for training. Please expand the data set.")

            for epoch in range(Init_Epoch, Freeze_Epoch):
                val_loss = fit_one_epoch(net, criterion, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch,
                                         Cuda)
                lr_scheduler.step(val_loss)

        if True:
            lr = 1e-4
            Batch_size = 16
            Freeze_Epoch = 50
            Unfreeze_Epoch = 100

            optimizer = optim.Adam(net.parameters(), lr=lr)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7,
                                                                verbose=True)

            train_dataset = SSDDataset(train_lines[:num_train], (Config["min_dim"], Config["min_dim"]), True)
            val_dataset = SSDDataset(val_lines[:num_val], (Config["min_dim"], Config["min_dim"]), False)

            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=ssd_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=ssd_dataset_collate)

            if backbone == "vgg":
                for param in model.vgg.parameters():
                    param.requires_grad = True
            else:
                for param in model.mobilenet.parameters():
                    param.requires_grad = True

            epoch_size = num_train // Batch_size
            epoch_size_val = num_val // Batch_size

            if epoch_size == 0 or epoch_size_val == 0:
                raise ValueError("The data set is too small for training. Please expand the data set.")

            for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
                val_loss = fit_one_epoch(net, criterion, epoch, epoch_size, epoch_size_val, gen, gen_val,
                                         Unfreeze_Epoch, Cuda)
                lr_scheduler.step(val_loss)