from glob import glob
from PIL import Image
from torchvision.datasets import VisionDataset


class ImageDataset(VisionDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + "/**/*.png", recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img, fpath.split("/")[-1]
