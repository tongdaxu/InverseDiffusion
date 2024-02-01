from .pipeline import PipelineInterface
from diffusers.utils.torch_utils import randn_tensor
import torch
from tqdm import tqdm


class DPSPipeline(PipelineInterface):
    def __init__(self, zeta):
        super().__init__()
        self.zeta = zeta

    def __call__(self, model, scheduler, operator, y, distance, **kwargs):
        if isinstance(model.config.sample_size, int):
            image_shape = (
                1,
                model.config.in_channels,
                model.config.sample_size,
                model.config.sample_size,
            )
        else:
            image_shape = (1, model.config.in_channels, *model.config.sample_size)

        image = randn_tensor(image_shape, device=y.device)

        pbar = tqdm(scheduler.timesteps)
        for t in pbar:
            # 1. predict noise model_output
            with torch.enable_grad():
                image = image.requires_grad_()
                model_output = model(image, t).sample

                # 2. compute previous image x'_{t-1} and original prediction x0_{t}
                scheduler_out = scheduler.step(model_output, t, image)
                image_pred, origi_pred = (
                    scheduler_out.prev_sample,
                    scheduler_out.pred_original_sample,
                )

                # 3. compute y'_t = f(x0_{t})
                y_pred = operator(origi_pred)

                # 4. compute loss = d(y, y'_t-1)
                loss = distance(y, y_pred)

                loss.backward()

            pbar.set_postfix({"dist: ": loss.item()}, refresh=False)

            with torch.no_grad():
                image_pred = image_pred - self.zeta * image.grad
                image = image_pred.detach()
        image = torch.clip(image, -1, 1)
        return image
