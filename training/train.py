import argparse
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from dataset_new import read_json_file

# from dataset import CustomDataset, ModelInput
from dataset_new import ModelInput, CustomDataset
from model_utils import DiceLoss, FocalLoss, FocalTverskyLoss
from R2AttU_Net import R2AttU_Net
from R2U_Net import R2U_Net
from U_Netpp import UnetPP
from densenet import create_mobilenet

import logging

# logging.basicConfig(level=logging.INFO,
#     format='%(asctime)s %(levelname)-8s %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     filename="training.log", 
#     filemode="w"
# )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_save_path = os.path.join("training", "models")
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help="training dataset creater")
parser.add_argument("-d", "--data", required=True)
parser.add_argument("-m", "--model", choices=["MobNet", "R2U", "R2AttU"], required=True)
parser.add_argument("-e", "--epochs", required=True, type=int)

args = parser.parse_args()

if args.model == "MobNet":
    model = create_mobilenet()
    NETWORK = model.to(DEVICE)
elif args.model == "R2U":
    NETWORK = R2U_Net().to(DEVICE)
else:
    NETWORK = R2AttU_Net().to(DEVICE)

OPTIMIZER = optim.Adam(NETWORK.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, 'min')

model_save_path = os.path.join(model_save_path, args.model)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

fires = read_json_file("./files/Filtered_WFIGS_Interagency_Perimeters.json")

training_list = []
validation_list = []

for fire_type in os.listdir(os.path.join(args.data)):
    for fire_id in os.listdir(os.path.join(args.data, fire_type, "patches")):

        rnd_choice = random.choice(range(1, 10))

        curr_fire = None
        for fire in fires:
            if int(fire_id) == fire.id and fire.fire is not None:
               curr_fire = fire 

        if curr_fire is None:
            continue

        if len(curr_fire.fire)!=0:
            fire_id_path = os.listdir(os.path.join(args.data, fire_type, "patches", fire_id))
            fire_id_paths = [sorted(fire_id_path)[i] for i in curr_fire.fire]

            for date in fire_id_paths:
                date_dir = os.path.join(args.data, fire_type, "patches", fire_id, date)
                if len(os.listdir(date_dir)) == 17:
                    if rnd_choice == 1:
                        validation_list.append(ModelInput(date_dir, True))
                    else:
                        training_list.append(ModelInput(date_dir, True))
                    
                else:
                   logging.warning(f"Bands missing in {date_dir}. Skipping image set from this timestamp")

        else:
            fire_id_path = os.listdir(os.path.join(args.data, fire_type, "patches", fire_id))

            for date in fire_id_path:
                date_dir = os.path.join(args.data, fire_type, "patches", fire_id, date)
                if len(os.listdir(date_dir)) == 17:
                    if rnd_choice == 1:
                        validation_list.append(ModelInput(date_dir, False))
                    else:
                        training_list.append(ModelInput(date_dir, False))
                else:
                   logging.warning(f"Bands missing in {date_dir}. Skipping image set from this timestamp")

transform = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=False),
    ]
)

target_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=False),
    ]
)

train_dataset = CustomDataset(
    training_list, transforms=transform, target_transforms=target_transform
)

validation_dataset = CustomDataset(
    validation_list, transforms=transform, target_transforms=target_transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

best_vloss = 1_000_000
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.)).to(DEVICE)
focal_loss = FocalLoss().to(DEVICE)
ft_loss = FocalTverskyLoss().to(DEVICE)
dice_loss = DiceLoss().to(DEVICE)

v_loss, t_loss       = [], []
f_loss_t, f_loss_v   = [], []
d_loss_t, d_loss_v   = [], []
ft_loss_t, ft_loss_v = [], []


for epoch in tqdm(range(args.epochs)):
    running_loss = 0
    running_floss_t = 0
    running_ftloss_t = 0
    running_dloss_t = 0

    NETWORK.train()

    for batch in (pbar := tqdm(train_loader, leave=False)):
        images, labels = batch
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        OPTIMIZER.zero_grad()

        if args.model == "MobNet":
            outputs = NETWORK(images)['out']

        else:
            outputs = NETWORK(images)

        # Compute the loss and its gradients

        loss = loss_fn(outputs, labels)

        out_cut = outputs.detach().clone()
        # out_cut[out_cut < 0.5] = 0.0
        # out_cut[out_cut >= 0.5] = 1.0

        fl_train = focal_loss(out_cut, labels)
        running_floss_t += fl_train.to("cpu").item()

        d_train = dice_loss(out_cut, labels)
        running_dloss_t += d_train.to("cpu").item()

        flt_train = ft_loss(out_cut, labels)
        running_ftloss_t += flt_train.to("cpu").item()

        pbar.set_description(f"Loss -> {loss}")
        loss.backward()

        # Adjust learning weights
        OPTIMIZER.step()

        # Gather data and report
        running_loss += loss.to("cpu").item()

    t_loss.append(running_loss / len(train_loader))
    f_loss_t.append(running_floss_t / len(train_loader))
    d_loss_t.append(running_dloss_t / len(train_loader))
    ft_loss_t.append(running_ftloss_t / len(train_loader))

    running_vloss = 0.0
    running_floss_v = 0.0
    running_ftloss_v = 0.0
    running_dloss_v = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    NETWORK.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(DEVICE), vlabels.to(DEVICE)

            if args.model == "MobNet":
                voutputs = NETWORK(vinputs)['out']
            else:
                voutputs = NETWORK(vinputs)

            vloss = loss_fn(voutputs, vlabels)

            out_cut = voutputs.detach().clone()
            # out_cut[out_cut < 0.5] = 0.0
            # out_cut[out_cut >= 0.5] = 1.0

            fl_val = focal_loss(out_cut, vlabels)
            running_floss_v += fl_val.to("cpu").item()

            d_val = dice_loss(out_cut, vlabels)
            running_dloss_v += d_val.to("cpu").item()

            flt_val = ft_loss(out_cut, vlabels)
            running_ftloss_v += flt_val.to("cpu").item()

            running_vloss += vloss.to("cpu").item()

    avg_vloss = running_vloss / len(validation_loader)
    scheduler.step(avg_vloss)
    v_loss.append(avg_vloss)
    f_loss_v.append(running_floss_v/ len(validation_loader))
    ft_loss_v.append(running_ftloss_v/ len(validation_loader))
    d_loss_v.append(running_dloss_v/ len(validation_loader))

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = os.path.join(model_save_path, f"model{epoch + 1}_{avg_vloss}.pth")
        torch.save(NETWORK, model_path)

xs = [x for x in range(args.epochs)]

plt.plot(xs, t_loss, label="t_loss")
plt.plot(xs, v_loss, "-.", label="v_loss")
plt.plot(xs, f_loss_t, label="f_loss_t")
plt.plot(xs, f_loss_v, label="f_loss_v")
plt.plot(xs, ft_loss_t, label="fl_loss_t")
plt.plot(xs, ft_loss_v, label="fl_loss_v")
plt.plot(xs, d_loss_t, label="d_loss_t")
plt.plot(xs, d_loss_v, label="d_loss_v")

plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Model loss")
plt.savefig(os.path.join(model_save_path, "fig.png"))
