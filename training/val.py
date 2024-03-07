import argparse
import os
from tqdm import tqdm

import torch
from torchvision.transforms import v2
from torchvision.utils import save_image

from dataset import CustomDataset, ModelInput

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help="training dataset creater")
parser.add_argument("-d", "--data", required=True)
parser.add_argument("-m", "--model", required=True)

args = parser.parse_args()

MODEL = torch.load(args.model, map_location=torch.device("cpu"))

boxes = [i for i in os.listdir(args.data) if i != "tmp"]

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=False),
    ]
)

for box in boxes:
    for fire in tqdm(os.listdir(os.path.join(args.data, str(box)))):
        inputs = ModelInput(os.path.join(args.data, str(box), fire))
        assert len(inputs.inputs) == 6

        dataset = CustomDataset([inputs], transforms=transform, target_transforms=transform)
        inputs, labels = next(iter(dataset))
        inputs = torch.unsqueeze(inputs, 0)
        MODEL.eval()
        with torch.no_grad():
            outputs = MODEL(inputs)
            # outputs[outputs < 0.5] = 0.0
            # outputs[outputs >= 0.5] = 1.0
        outputs = torch.hstack((torch.squeeze(outputs, 0), labels))
        save_path = os.path.join(args.data, str(box), fire, "box.png")
        save_image(outputs, save_path)



# img = (np.array(Image.open("./DATA/83/2024-01-31 00:01:17.400000/output.tiff")) * 255).astype(np.uint8)
# img = Image.fromarray(img)
# img.save("imm.png")

# for input in inputs_list:
#     pass
