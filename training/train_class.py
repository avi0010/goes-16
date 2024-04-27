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

from class_Net import create_resnet18_model
from dataset_class import ModelInput, CustomDataset

import logging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_save_path = os.path.join("training", "models")
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help="training dataset creater")
parser.add_argument("-d", "--data", required=True)
parser.add_argument("-r", "--ratio", required=True, type=float)
parser.add_argument("-t", "--threshold", required=False, type=float, default=0.5)
parser.add_argument("-e", "--epochs", required=True, type=int)

args = parser.parse_args()

NETWORK = create_resnet18_model().to(DEVICE)

OPTIMIZER = optim.Adam(NETWORK.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, 'min')

model_save_path = os.path.join(model_save_path, "resnet18")
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

inputs_list = []

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
                        validation_list.append(ModelInput(date_dir, True))
                    else:
                        training_list.append(ModelInput(date_dir, True))
                else:
                   logging.warning(f"Bands missing in {date_dir}. Skipping image set from this timestamp")

transform = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=False),
    ]
)

train_dataset = CustomDataset(training_list, transforms=transform)
validation_dataset = CustomDataset(validation_list, transforms=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

loss_fn = nn.BCEWithLogitsLoss(torch.tensor(10.)).to(DEVICE)
v_loss, t_loss       = [], []


for epoch in tqdm(range(args.epochs)):

    running_loss = 0
    NETWORK.train()

    for batch in (pbar := tqdm(train_loader, leave=False)):
        images, labels = batch
        images, labels = images.to(DEVICE), labels.to(DEVICE).to(torch.float)

        OPTIMIZER.zero_grad()

        outputs = torch.sigmoid(NETWORK(images)).reshape(-1)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)

        pbar.set_description(f"Loss -> {loss}")
        loss.backward()

        # Adjust learning weights
        OPTIMIZER.step()

        # Gather data and report
        running_loss += loss.to("cpu").item()

    t_loss.append(running_loss / len(train_loader))

    running_vloss = 0.0

    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    NETWORK.eval()

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(DEVICE), vlabels.to(DEVICE)

            voutputs = torch.sigmoid(NETWORK(vinputs))

            vloss = loss_fn(voutputs, vlabels)

            running_vloss += vloss.to("cpu").item()

    avg_vloss = running_vloss / len(validation_loader)
    scheduler.step(avg_vloss)
    v_loss.append(avg_vloss)

xs = [x for x in range(args.epochs)]
plt.plot(xs, t_loss, label="t_loss")
plt.plot(xs, v_loss, "-.", label="v_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Model loss")
plt.savefig(os.path.join(model_save_path, "fig.png"))
