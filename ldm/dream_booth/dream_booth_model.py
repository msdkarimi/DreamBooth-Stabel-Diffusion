import os
import torch
from itertools import islice
import torch.nn as nn
import numpy as np
import time
from PIL import Image
from torch import autocast
from einops import rearrange
from torchvision.utils import make_grid
from omegaconf import OmegaConf
import pytorch_lightning as pl
from imwatermark import WatermarkEncoder
from tqdm import tqdm, trange
from contextlib import contextmanager, nullcontext


import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from plms import PLMSSampler
from dream_booth_util import load_model_from_config, load_replacement, check_safety, put_watermark



def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


class DreamBooth(pl.LightningModule):

    def __init__(self, *, prior_config: OmegaConf, ldm_config: OmegaConf, opt, device):
        super().__init__()

        self.opt = opt

        self.__init_prior(prior_config, opt)
        self.__init_ldm(ldm_config)


        Zs_prior = self.get_priors(device)
        ## uncomment later.
        # self.init_ldm(ldm_config)



    def __init_prior(self, prior_config, opt):
        self.prior_model = load_model_from_config(prior_config, opt.ckpt, prior=True)

    def __init_ldm(self, ldm_config):
        self.ldm_model = load_model_from_config(ldm_config)


    def get_priors(self, device):

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
                data = list(chunk(data, batch_size))

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1
        start_code = None
        if self.opt.fixed_code:
            start_code = torch.randn([self.opt.n_samples, self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f], device=device)

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
                            uc = self.prior_model.get_learned_conditioning(batch_size * [""])
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

        self.prior_model = self.prior_model.to('cpu')
        torch.cuda.empty_cache()
        return samples_ddim





    def prior_z(self, input_text:str, negative_prompt:str = ""):
        pass


