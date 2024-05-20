import sys, os
import torch
import argparse
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# from dream_booth_util import load_model_from_config
from dream_booth_model import DreamBooth


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )

    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1371,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=False,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--datadir_in_name",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Prepend the final directory in the data_root to the output directory name")

    parser.add_argument(
        "--actual_resume",
        type=str,
        required=False,
        help="Path to model to actually resume from")

    parser.add_argument(
        "--data_root",
        type=str,
        required=False,
        help="Path to directory with training images")

    parser.add_argument(
        "--reg_data_root",
        type=str,
        required=False,
        help="Path to directory with regularization images")

    parser.add_argument(
        "--class_word",
        type=str,
        default="dog",
        help="Placeholder token which will be used to denote the concept in future prompts")


    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )

    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )

    parser.add_argument(
        "--config_prior",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model for prior knowledge",
    )
    parser.add_argument(
        "--config_fine_tune",
        type=str,
        default="configs/latent-diffusion/txt2img-fine-tuning.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )


    return parser


def main(db_parser=None):
    assert db_parser is not None, 'Hey dude! give me my so called white cane :('

    seed_everything()
    opt = parser.parse_args()

    config_prior = OmegaConf.load(f"{opt.config_prior}")
    config_fine_tune = OmegaConf.load(f"{opt.config_fine_tune}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    assert config_fine_tune.data.params.batch_size == opt.n_samples, "number of prior samples expected to be the same as batch size!!"



    db_model = DreamBooth(prior_config=config_prior, ldm_config=config_fine_tune, opt=opt, device=device)



if __name__ == '__main__':
    parser = get_parser()
    main(parser)
    sys.exit(0)


