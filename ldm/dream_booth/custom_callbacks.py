from pytorch_lightning.callbacks import Callback
from ldm.dream_booth.dream_booth_util import load_model_from_config
import torch
from tqdm import tqdm
from torch import autocast
import os
import time
from imwatermark import WatermarkEncoder
from contextlib import contextmanager, nullcontext
from itertools import islice
from ldm.dream_booth.plms import PLMSSampler
from omegaconf import OmegaConf



# ---------------
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from einops import rearrange
from PIL import Image
import numpy as np
import cv2
from torchvision.utils import make_grid
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img
# -----------------



class PriorPreservation(Callback):
    def __init__(self, batch_frequency, prior_config, opt):
        self.opt = opt
        self.prior_config = OmegaConf.load(f'{prior_config}')
        self.prior_model = load_model_from_config(self.prior_config, opt.ckpt, prior=True)
        self.batch_freq = batch_frequency
        print("just to check if instance has been created")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataset_idx):
        z = self.get_priors()
        pl_module.prior_z = z

    @staticmethod
    def chunk(it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def get_priors(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.prior_model = self.prior_model.to(device)

        if self.opt.plms:
            sampler = PLMSSampler(self.prior_model)
        else:
            raise NotImplementedError("no other sampler is implemented for dreambooth so far!! :(")

        batch_size = self.opt.n_samples
        os.makedirs(self.opt.outdir, exist_ok=True)
        outpath = self.opt.outdir

        self.opt.outdirbatch_size = self.opt.n_samples

        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))


        n_rows = self.opt.n_rows if self.opt.n_rows > 0 else batch_size
        if not self.opt.from_file:
            prompt = self.opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {self.opt.from_file}")
            with open(self.opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(self.chunk(data, batch_size))

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1
        start_code = None

        # if self.opt.fixed_code:
        #     start_code = torch.randn([self.opt.n_samples, self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f], device=device)

        precision_scope = autocast if self.opt.precision == "autocast" else nullcontext

        with torch.no_grad():
            with precision_scope("cuda"):
                with self.prior_model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    # for n in trange(self.opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if self.opt.scale != 1.0:
                            uc = self.prior_model.get_learned_conditioning(batch_size * [self.opt.neg_prompt])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = self.prior_model.get_learned_conditioning(prompts)
                        shape = [self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f]
                        samples_ddim, _ = sampler.sample(S=self.opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=self.opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=self.opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=self.opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim =  self.prior_model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not self.opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1

                        if not self.opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                if not self.opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

        self.prior_model = self.prior_model.to('cpu')
        torch.cuda.empty_cache()
        return samples_ddim.detach()
