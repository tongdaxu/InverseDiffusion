import argparse
import yaml
import os

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from inversediffusion.utils.utils import build_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipe", type=str)
    parser.add_argument("--op", type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0")

    with open(args.pipe) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(args.op) as f:
        config_op = yaml.load(f, Loader=yaml.FullLoader)

    # data loader
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    config["dataset"]["kwargs"]["transforms"] = transform
    dataset = build_cfg(config["dataset"])
    dataloader = DataLoader(dataset, 1, num_workers=8)

    # model
    model = build_cfg(config["model"])
    model.eval()
    model = model.to(device)

    # scheduler
    scheduler = build_cfg(config["scheduler"])
    scheduler.set_timesteps(config["global"]["timesteps"], device=device)

    # pipeline
    pipeline = build_cfg(config["pipeline"])

    # save dir
    save_dir = config["global"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    for img_dir in ["input", "recon", "progress", "label", "input2"]:
        os.makedirs(os.path.join(save_dir, img_dir), exist_ok=True)

    # operator
    operator = build_cfg(config_op["operator"])
    operator.eval()
    operator = operator.to(device)

    # distance
    distance = build_cfg(config_op["distance"])
    distance.eval()
    distance = distance.to(device)

    # run DIS
    piperun_args = {"save_dir": save_dir}
    for i, (x, name) in enumerate(dataloader):
        x = x.to(device)
        y = operator.init(x)
        x_hat = pipeline(model, scheduler, operator, y, distance, **piperun_args)
        y_hat = operator.init(x_hat)
        # save
        for j in range(len(name)):
            save_image((x[j] + 1.0) / 2.0, os.path.join(save_dir, "label", name[j]))
            save_image((y[j] + 1.0) / 2.0, os.path.join(save_dir, "input", name[j]))
            save_image((x_hat[j] + 1.0) / 2.0, os.path.join(save_dir, "recon", name[j]))
            save_image(
                (y_hat[j] + 1.0) / 2.0, os.path.join(save_dir, "input2", name[j])
            )


if __name__ == "__main__":
    main()
