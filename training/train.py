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

from dataset import CustomDataset, ModelInput
from model_utils import DiceLoss, FocalLoss, FocalTverskyLoss
from R2AttU_Net import R2AttU_Net
from R2U_Net import R2U_Net
from U_Netpp import UnetPP

import logging

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename="training.log", 
    filemode="w"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_save_path = os.path.join("training", "models")
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help="training dataset creater")
parser.add_argument("-d", "--data", required=True)
parser.add_argument("-r", "--ratio", required=True, type=float)
parser.add_argument("-m", "--model", choices=["unet", "R2U", "R2AttU"], required=True)
parser.add_argument("-t", "--threshold", required=False, type=float, default=0.5)
parser.add_argument("-e", "--epochs", required=True, type=int)

args = parser.parse_args()

if args.model == "unet":
    NETWORK = UnetPP().to(DEVICE)
elif args.model == "R2U":
    NETWORK = R2U_Net().to(DEVICE)
else:
    NETWORK = R2AttU_Net().to(DEVICE)

OPTIMIZER = optim.Adam(NETWORK.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, 'min')

model_save_path = os.path.join(model_save_path, args.model)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

boxes = [i for i in os.listdir(args.data) if i != "tmp"]

inputs_list = []

for box in boxes:
    for fire in os.listdir(os.path.join(args.data, str(box))):
        inputs = ModelInput(os.path.join(args.data, str(box), fire))
        if len(inputs.inputs) == 6:
            inputs_list.append(inputs)
        else:
            logging.warning(f"Bands missing in {inputs.in_dir}. Skipping image set from this timestamp")

random.shuffle(inputs_list)
split = int(len(inputs_list) * args.ratio)
training_list = inputs_list[:split]
validation_list = inputs_list[split:]

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
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=True)

best_vloss = 1_000_000
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(100.)).to(DEVICE)
focal_loss = FocalLoss().to(DEVICE)
ft_loss = FocalTverskyLoss().to(DEVICE)

v_loss, t_loss       = [], []
f_loss_t, f_loss_v   = [], []
ft_loss_t, ft_loss_v = [], []


for epoch in tqdm(range(args.epochs)):
    running_loss = 0
    running_floss_t = 0
    running_ftloss_t = 0

    NETWORK.train()

    for batch in (pbar := tqdm(train_loader, leave=False)):
        images, labels = batch
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = NETWORK(images)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)

        out_cut = outputs.detach().clone()
        out_cut[out_cut < 0.5] = 0.0
        out_cut[out_cut >= 0.5] = 1.0

        fl_train = focal_loss(out_cut, labels)
        running_floss_t += fl_train.to("cpu").item()

        flt_train = ft_loss(out_cut, labels)
        running_ftloss_t += flt_train.to("cpu").item()

        pbar.set_description(f"Loss -> {loss}")
        loss.backward()

        # Adjust learning weights
        OPTIMIZER.step()

        # Gather data and report
        running_loss += loss.to("cpu").item()

    t_loss.append(running_loss)
    f_loss_t.append(running_floss_t)
    ft_loss_t.append(running_ftloss_t)

    running_vloss = 0.0
    running_floss_v = 0.0
    running_ftloss_v = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    NETWORK.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(DEVICE), vlabels.to(DEVICE)
            voutputs = NETWORK(vinputs)
            vloss = loss_fn(voutputs, vlabels)

            out_cut = voutputs.detach().clone()
            out_cut[out_cut < 0.5] = 0.0
            out_cut[out_cut >= 0.5] = 1.0

            fl_val = focal_loss(out_cut, vlabels)
            running_floss_v += fl_val.to("cpu").item()

            flt_val = ft_loss(out_cut, vlabels)
            running_ftloss_v += flt_val.to("cpu").item()

            running_vloss += vloss.to("cpu").item()

    avg_vloss = running_vloss / len(validation_loader)
    scheduler.step(avg_vloss)
    v_loss.append(avg_vloss)
    f_loss_v.append(running_floss_v)
    ft_loss_v.append(running_ftloss_v)

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

plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Model loss")
plt.savefig(os.path.join(model_save_path, "fig.png"))
