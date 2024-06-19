"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
import random
from pytorch_lightning import seed_everything

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=250,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.70,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(1, 234598),
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # dataset_dir =

    # assert os.path.isfile(opt.init_img)
    # init_image = load_img(opt.init_img).to(device)
    # init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    # init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    dict_of_data = ["broken_pipe", "dark_mold", "dark_stain",
                    "light_mold", "light_stain", "liquid_spillage",
                    "peeling", "pipe", "raising"]

    # dict_of_data = {
    #     "broken_pipe": ("a photo of a broken pipe in the wall", 6),
    #     "dark_mold": ("A photo of a ceiling with dark mold in a bathroom", 8),
    #     "dark_stain": ("an image of a ceiling showing a prominent dark stain, suggesting water damage in an aged room", 10),
    #     "light_mold": ("A photo of a wall with light mold growth, indicating moisture damages", 8),
    #     "light_stain": ("a photo of a white wall shows light stain, indicating water damage caused by moisture", 9),
    #     "liquid_spillage": ("A photo of a floor with spilled water", 8),
    #     "peeling": ("A photo of a wall with peeling", 7),
    #     "pipe": ("A photo of a pipe", 5),
    #     "raising": ("a photo of a wall with blister caused by moisture", 7),
    # }

    # A photo of a wooden floor, elevated or lifted due to moisture.

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for _loop in range(55):
                    print(f'-------loop-------{_loop}')
                    for label_folder in dict_of_data:
                        ## we don't want it to be random, otherwise, intra-class imbalance
                        _root = "/content/input_images"
                        # label_dirs = os.listdir(_root)
                        # random_label_index = random.randint(0, len(label_dirs)-1)
                        # label_folder = label_dirs[random_label_index]
                        #

                        list_label_images = [image for image in os.listdir(f'{_root}/{label_folder}') if
                                             image.strip().split(".")[-1] == 'jpg']
                                             # image != '.DS_Store']
                        random_label_image_index = random.randint(0, len(list_label_images) - 1)
                        file_name = list_label_images[random_label_image_index]
                        the_image_path = f'{_root}/{label_folder}/{file_name}'

                        # setting image dir,
                        opt.init_img = the_image_path

                        # moving selected image to latent
                        assert os.path.isfile(opt.init_img)
                        init_image = load_img(opt.init_img).to(device)
                        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                        init_latent = model.get_first_stage_encoding(
                            model.encode_first_stage(init_image))  # move to latent space

                        for n in trange(opt.n_iter, desc="Sampling"):
                            for prompts in tqdm(data, desc="data", disable=True):
                                uc = None
                                if opt.scale != 1.0:
                                    uc = model.get_learned_conditioning(batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)

                                # prompts, index_of_token = dict_of_data[label_folder]

                                _data = file_name.strip().split("_")
                                prompts = _data[0]
                                index_of_token = int(_data[1])

                                c = model.get_learned_conditioning(prompts)

                                # encode (scaled latent)
                                z_enc = sampler.stochastic_encode(init_latent,
                                                                  torch.tensor([t_enc] * batch_size).to(device))
                                # decode it
                                samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         index_of_token=index_of_token, label_folder=label_folder,
                                                         image_name=base_count)

                                x_samples = model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                                if not opt.skip_save:
                                    for x_sample in x_samples:
                                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                        Image.fromarray(x_sample.astype(np.uint8)).save(
                                            # os.path.join(sample_path, f"{base_count:05}.jpg"))
                                            os.path.join(f'/content/_dataset/{label_folder}/images',
                                                         f"{base_count:05}.jpg"))
                                        base_count += 1
                                # all_samples.append(x_samples)

                        # if not opt.skip_grid:
                        #     # additionally, save as grid
                        #     grid = torch.stack(all_samples, 0)
                        #     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        #     grid = make_grid(grid, nrow=n_rows)
                        #
                        #     # to image
                        #     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        #     Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                        #     grid_count += 1

                        # toc = time.time()
    sampler.attn_controller.save_annots()
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
