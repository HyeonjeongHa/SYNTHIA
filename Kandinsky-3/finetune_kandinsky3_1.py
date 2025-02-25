import argparse
import os
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import pandas as pd
import math
from packaging import version
import json
import pickle
import random

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from torchvision import transforms
import transformers
from transformers.utils import ContextManagers
import torchvision.transforms as transforms

import diffusers
from transformers import AutoTokenizer
from diffusers import AutoPipelineForText2Image
from diffusers.optimization import get_scheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from omegaconf import OmegaConf
from copy import deepcopy
from transformers import T5EncoderModel
import wandb
import torch
from torch.optim.optimizer import Optimizer



from einops import repeat

from kandinsky3.model.unet import UNet
from kandinsky3.movq import MoVQ
from kandinsky3.condition_encoders import T5TextConditionEncoder
from kandinsky3.condition_processors import T5TextConditionProcessor
from kandinsky3.model.diffusion import BaseDiffusion, get_named_beta_schedule
from kandinsky3 import get_T2I_pipeline, get_T2I_Flash_pipeline

logger = get_logger(__name__, log_level="INFO")


def get_key_step(step, log_step_interval=50):
    step_d = step // log_step_interval
    return 'loss_' + str(step_d * log_step_interval) + '_' + str((step_d + 1) * log_step_interval)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of finetuning Kandinsky 3.")
    parser.add_argument(
        "--image_resolution",
        type=int,
        default=128,
        required=False,
        help="Image resolution",
    )
    parser.add_argument(
        "--image_data_path",
        type=str,
        default="stage_1_data_5neg_imgs.json",
        required=False,
        help="image_data_path",
    )
    parser.add_argument(
        "--device_num",
        type=int,
        default=5,
        required=False,
        help="Device Number",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        required=False,
        help="Number of Epochs",
    )
    parser.add_argument(
        "--pretrained_kandinsky_path",
        type=str,
        default="kandinsky-community/kandinsky-3",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        required=False,
        help="train batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        required=False,
        help="learning rate",
    )
    parser.add_argument(
        "--rand_train",
        type=bool,
        default=False,
        required=False,
        help="whether to apply curriculum learning",
    )
    parser.add_argument(
        "--loss_beta",
        type=float,
        default=0.5,
        required=False,
        help="the coefficient on the negative loss",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        required=False,
        help="weight decay",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="kandi_3-model-finetuned_random",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=125,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.98, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--loss_type", default="mse", type=str, help="mse/cos")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--pos_only", action="store_true")
    parser.add_argument("--use_f32", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def center_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_data_path, image_resolution=512):
        self.pos_img_paths = []
        self.neg_img_paths = []
        self.pos_prompts = []
        self.neg_prompts = []
        self.image_resolution = image_resolution
        self.transform = transforms.Compose([
            transforms.Resize((self.image_resolution, self.image_resolution)),
            transforms.ToTensor()
        ])
        
        with open(image_data_path, "r") as f:
            all_lines = f.readlines()
            for line in all_lines:
                curr_obj = json.loads(line)
                self.pos_prompts.append("a new design that have functions including " + ", ".join(curr_obj["pos_prompts"]))
                self.neg_prompts.append(", ".join(curr_obj["neg_prompts"]))
                self.pos_img_paths.append(curr_obj["pos_paths"])
                self.neg_img_paths.append(curr_obj["neg_paths"])
        
        
    def __len__(self):
        return len(self.pos_img_paths)

    def _process_img(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img
    
    def __getitem__(self, i):
        rand_ind = random.randint(0, len(self.pos_img_paths[i]) - 1)
        pos_img = self._process_img(self.pos_img_paths[i][rand_ind])
        # pos_img = self._process_img(self.pos_img_paths[i][0])
        all_neg = []
        all_keys = list(self.neg_img_paths[i].keys())
        all_keys.sort()
        # random.shuffle(all_keys)
        # all_keys = all_keys[:3]
        for key in all_keys:
            val = self.neg_img_paths[i][key]
            for ii in range(len(val)):
                neg_img = self._process_img(val[ii]["image_path"])
                all_neg.append(neg_img)
        random.shuffle(all_neg)
        all_neg = np.array(all_neg[0])
        # all_neg = np.stack(all_neg, axis=0)
        pos_prompt = self.pos_prompts[i]
        neg_prompt = self.neg_prompts[i]
        return pos_img, all_neg, pos_prompt, neg_prompt


def main():
    args = parse_args()
    dt = args.image_data_path.split(".")[0]
    exp_name = f"model_ckpt_rand_embed_abl/3.1-lr{args.lr}-grad_acc{args.gradient_accumulation_steps}-ep{args.num_train_epochs}-CL{not args.rand_train}-loss{args.loss_type}-beta{args.loss_beta}_normX_f32{args.use_f32}_datatype_{dt}"
    if args.pos_only:
        exp_name = f"{exp_name}_posOnly"
    if "noise" in args.loss_type:
        exp_name = f"{exp_name}_gamma{args.gamma}"
    if args.use_wandb:
        args_dict = vars(args)
        wandb.init(
            project="ai4design",
            entity="ronas",
            name=exp_name,
            config=args_dict
        )
        
    args.output_dir = os.path.join("checkpoints", exp_name)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    weight_dtype = torch.float32        
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16  


    device_map = torch.device('cuda:' + str(args.device_num))
    dtype_map = {
        'unet': torch.float32 if args.use_f32 else torch.bfloat16,
        'text_encoder': torch.bfloat16,
        'movq': torch.float32 if args.use_f32 else torch.bfloat16,
    }
    t2i_pipe = get_T2I_Flash_pipeline(
        device_map, dtype_map,cache_dir="./cache/"
    )

    t2i_pipe.movq.requires_grad_(False)
    t2i_pipe.t5_encoder.requires_grad_(False)
    t2i_pipe.unet.requires_grad_(True)
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
        
    optimizer = optimizer_cls(
        t2i_pipe.unet.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )
    
    train_dataset = ImageDataset(image_data_path=args.image_data_path, image_resolution=args.image_resolution)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    # Prepare everything with our `accelerator`.
    optimizer, t2i_pipe, lr_scheduler = accelerator.prepare(
        optimizer, t2i_pipe, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Loss type = {args.loss_type}")
    logger.info(f"  Beta = {args.loss_beta}")
    logger.info(f"  Curriculum Learning = {args.rand_train}")
    logger.info(f"  Run Name = {args.output_dir}")
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            first_epoch = 0
    print('global_step =', global_step)
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("training goes brrr")
    guidance_scale = 3.0
    eta = 1.0
    betas = get_named_beta_schedule('cosine', 1000)
    base_diffusion = BaseDiffusion(betas, 0.99)
    for epoch in range(first_epoch, args.num_train_epochs):
        if not args.rand_train:
            if epoch == args.num_train_epochs // 3:
                    train_dataset = ImageDataset(image_data_path=args.image_data_path.replace("stage_1", "stage_2"), image_resolution=args.image_resolution)
                    train_dataloader = torch.utils.data.DataLoader(
                        train_dataset,
                        shuffle=True,
                        batch_size=args.train_batch_size,
                        num_workers=args.dataloader_num_workers,
                    )
            elif epoch == 2 * args.num_train_epochs // 3:
                    train_dataset = ImageDataset(image_data_path=args.image_data_path.replace("stage_1", "stage_3"), image_resolution=args.image_resolution)
                    train_dataloader = torch.utils.data.DataLoader(
                        train_dataset,
                        shuffle=True,
                        batch_size=args.train_batch_size,
                        num_workers=args.dataloader_num_workers,
                    )
            
        t2i_pipe.unet.train()
        train_loss = 0.0
        losses = []
        steps = []
        if "mse" in args.loss_type:
            loss_fn = lambda pred, pos: F.mse_loss(pred, pos, reduction="mean")
        elif "cos" in args.loss_type:
            loss_fn = lambda pred, pos: (1 - F.cosine_similarity(pred, pos, dim=-1)).mean()
        else:
            raise ValueError(f"Unsupported loss type: {args.loss_type}")
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(t2i_pipe.unet):
                times = list(range(999, 0, -1000 // 30))
                if t2i_pipe.gan:
                    times = list(range(979, 0, -250))
                pos_image, neg_image, pos_prompt, neg_prompt = batch
                pos_image, neg_image = pos_image.to(device_map), neg_image.to(device_map)
                condition_model_input, negative_condition_model_input = t2i_pipe.t5_processor.encode(pos_prompt[0], neg_prompt[0])
                for input_type in condition_model_input:
                    condition_model_input[input_type] = condition_model_input[input_type][None].to(
                        t2i_pipe.device_map['text_encoder']
                    )
        
                if negative_condition_model_input is not None:
                    for input_type in negative_condition_model_input:
                        negative_condition_model_input[input_type] = negative_condition_model_input[input_type][None].to(
                            t2i_pipe.device_map['text_encoder']
                        )
                with torch.autocast('cuda:' + str(args.device_num), dtype=t2i_pipe.dtype_map['text_encoder']):
                    context, context_mask = t2i_pipe.t5_encoder(condition_model_input)
                    if negative_condition_model_input is not None:
                        negative_context, negative_context_mask = t2i_pipe.t5_encoder(negative_condition_model_input)
                    else:
                        negative_context, negative_context_mask = None, None

                with torch.autocast('cuda:' + str(args.device_num),dtype=t2i_pipe.dtype_map['unet']):
                    images = base_diffusion.p_sample_loop(
                        t2i_pipe.unet, (1, 4, args.image_resolution // 8, args.image_resolution // 8), times, t2i_pipe.device_map['unet'],
                        context, context_mask, t2i_pipe.null_embedding, guidance_scale, eta,
                        negative_context=negative_context, negative_context_mask=negative_context_mask,
                        gan=t2i_pipe.gan
                    )

                with torch.autocast('cuda:' + str(args.device_num),dtype=t2i_pipe.dtype_map['movq']):
                    pred_image = torch.cat([t2i_pipe.movq.decode(image) for image in images.chunk(2)])
                    pred_image = torch.clip((pred_image + 1.) / 2., 0., 1.)
                
                if args.pos_only:
                    loss_pos = loss_fn(pred_image.float(), pos_image.float())
                    loss = loss_pos
                else:
                    loss_pos = loss_fn(pred_image.float(), pos_image.float())
                    loss_neg = loss_fn(pred_image.float(), neg_image.float())
                    loss = loss_pos - args.loss_beta * loss_neg
                
                if "noise" in args.loss_type:
                    with torch.amp.autocast('cuda:' + str(args.device_num), dtype=t2i_pipe.dtype_map['movq']):
                        pos_latents = t2i_pipe.movq.encode(pos_image * 2.0 - 1.0)

                    t = torch.randint(
                        0, base_diffusion.num_timesteps, 
                        (pos_latents.size(0),), 
                        device=pos_latents.device
                    )
                    noise = torch.randn_like(pos_latents)
                    x_noisy = base_diffusion.q_sample(pos_latents, t, noise)
                    with torch.amp.autocast('cuda:' + str(args.device_num), dtype=t2i_pipe.dtype_map['unet']):
                        pred_noise = t2i_pipe.unet(
                            x_noisy,                
                            t,                      
                            context=context, 
                            context_mask=context_mask.bool()
                        )

                    noise_pred_loss = F.mse_loss(pred_noise, noise)
                    loss = loss + args.gamma*noise_pred_loss
                    
                with torch.no_grad():
                    if "mse" in args.loss_type:
                        losses_l = F.mse_loss(pred_image.float(), pos_image.float(), reduction="none").mean(dim=(1, 2, 3))
                    elif "cos" in args.loss_type:
                        losses_l = (1 - F.cosine_similarity(pred_image.float(), pos_image.float(), dim=1, eps=1e-8)).mean(dim=(1, 2))
                    losses_list = list(losses_l.cpu().detach().numpy())
                    losses += losses_list ###

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(t2i_pipe.unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                
            if args.use_wandb:
                log_dict = {
                    "step_loss": loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0],
                    "loss_pos": loss_pos.item(), 
                    "avg_loss": avg_loss.item(),
                    "train_loss": train_loss, 
                }
                if not args.pos_only:
                    log_dict.update({
                        "loss_neg": loss_neg.item(), 
                    })
                wandb.log(log_dict)
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                log_step_interval = 50
                logs_keys = [get_key_step(i) for i in range(0, 1000, log_step_interval)]
                logs_dict = {key:[] for key in logs_keys}
                for i in range(len(steps)):
                    logs_dict[get_key_step(steps[i])].append(losses[i])
                filtered_logs_keys = [get_key_step(i) for i in range(0, 1000, log_step_interval) if len(logs_dict[get_key_step(i)]) > 0]
                filtered_logs_dict = {key:float(np.array(logs_dict[key]).mean()) for key in filtered_logs_keys}
                filtered_logs_dict["train_loss"] = train_loss
                accelerator.log(filtered_logs_dict, step=global_step)
                train_loss = 0.0
                losses, steps = [], []

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        
                        model_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}/unet_model_checkpoint.pt")
                        torch.save(accelerator.unwrap_model(t2i_pipe.unet).state_dict(), model_save_path)
                        logger.info(f"Model Saved state to {model_save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
    accelerator.end_training()
    if args.use_wandb: wandb.finish()


if __name__ == "__main__":
    main()