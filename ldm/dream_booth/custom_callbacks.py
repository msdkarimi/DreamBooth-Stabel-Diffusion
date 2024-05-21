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


class PriorPreservation(Callback):
    def __init__(self, batch_frequency, prior_config):

        self.prior_config = OmegaConf.load(prior_config)
        # self.prior_model = load_model_from_config(prior_config, opt.ckpt, prior=True)
        self.batch_freq = batch_frequency
        print("just to check if instance has been created")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
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
        return samples_ddim.detach()
