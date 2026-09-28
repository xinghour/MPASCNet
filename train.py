import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import sys
sys.path.append("..")
import os
import time
import random
import argparse
import json
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from env import AttrDict, build_env
from data.dataset import MusicCodecDataset, MusicCodecEvalDataset, mag_pha_stft, mag_pha_istft
from models.wapper import MPASCNet
from losses.losses import generator_losses
from utils import load_checkpoint, save_checkpoint


torch.backends.cudnn.benchmark = True


def stft_audio(audio, h):
    B, C, T = audio.shape
    audio = audio.reshape(B * C, T)
    mag, pha, com = mag_pha_stft(audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

    return mag, pha, com


def istft_audio(mag, pha, h, batch_size, channels, length):
    audio = mag_pha_istft(mag, pha, h.n_fft, h.hop_size, h.win_size, h.compress_factor, length=length)
    audio = audio.reshape(batch_size, channels, audio.shape[-1])

    return audio


def cal_sdr(clean_audio, audio_g, eps=1e-8):
    noise = clean_audio - audio_g
    sdr = 10 * torch.log10((clean_audio.pow(2).sum(dim=-1) + eps) / (noise.pow(2).sum(dim=-1) + eps))

    return sdr


def cal_si_snr(clean_audio, audio_g, eps=1e-8):
    clean_audio = clean_audio - clean_audio.mean(dim=-1, keepdim=True)
    audio_g = audio_g - audio_g.mean(dim=-1, keepdim=True)

    target = torch.sum(audio_g * clean_audio, dim=-1, keepdim=True) * clean_audio
    target = target / (torch.sum(clean_audio ** 2, dim=-1, keepdim=True) + eps)
    noise = audio_g - target

    si_snr = 10 * torch.log10((target.pow(2).sum(dim=-1) + eps) / (noise.pow(2).sum(dim=-1) + eps))

    return si_snr


def unwrap_model(model):
    if isinstance(model, DDP):
        return model.module

    return model


def find_resume_checkpoint(checkpoint_path):
    latest_checkpoint = os.path.join(checkpoint_path, "latest")

    if os.path.isfile(latest_checkpoint):
        return latest_checkpoint

    return None


def find_finetune_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    best_checkpoint = os.path.join(checkpoint_path, "best")
    latest_checkpoint = os.path.join(checkpoint_path, "latest")

    if os.path.isfile(best_checkpoint):
        return best_checkpoint

    if os.path.isfile(latest_checkpoint):
        return latest_checkpoint

    return None


def reduce_metrics(metrics, batches, device, distributed):
    names = list(metrics.keys())
    values = [metrics[name] for name in names]
    values.append(batches)

    values = torch.tensor(values, dtype=torch.float64, device=device)

    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)

    batches = values[-1].item()

    return {
        name: (values[i] / batches).item()
        for i, name in enumerate(names)
    }


def restore_training(a, generator, device):
    is_main = not dist.is_initialized() or dist.get_rank() == 0

    training_state = {
        "steps": 0,
        "last_epoch": -1,
        "best_si_snr": -100.,
        "si_snr_no_improve_epochs": 0,
        "wandb_run_id": None,
        "checkpoint": None
    }

    if a.finetune_from is not None:
        checkpoint_path = find_finetune_checkpoint(a.finetune_from)

        if checkpoint_path is None:
            if is_main:
                print("No finetune checkpoint found.")

            return training_state

        if is_main:
            print("Loading pretrained generator from: {}".format(checkpoint_path))

        checkpoint = load_checkpoint(checkpoint_path, device)

        if "generator" in checkpoint:
            generator.load_state_dict(checkpoint["generator"])
        else:
            generator.load_state_dict(checkpoint)

        if is_main:
            print("Loaded pretrained generator.")

        return training_state

    if a.no_resume:
        if is_main:
            print("Resume disabled.")

        return training_state

    resume_path = a.resume_path if a.resume_path is not None else a.checkpoint_path
    checkpoint_path = find_resume_checkpoint(resume_path)

    if checkpoint_path is None:
        if is_main:
            print("No resume checkpoint found.")

        return training_state

    if is_main:
        print("Resuming from checkpoint: {}".format(checkpoint_path))

    checkpoint = load_checkpoint(checkpoint_path, device)
    generator.load_state_dict(checkpoint["generator"])

    training_state["steps"] = checkpoint.get("steps", 0)
    training_state["last_epoch"] = checkpoint.get("epoch", -1)
    training_state["best_si_snr"] = checkpoint.get("best_si_snr", -100.)
    training_state["si_snr_no_improve_epochs"] = checkpoint.get("si_snr_no_improve_epochs", 0)
    training_state["wandb_run_id"] = checkpoint.get("wandb_run_id", None)
    training_state["checkpoint"] = checkpoint

    if is_main:
        print(
            "Resume state: epoch {}, steps {}, best_si_snr {:.4f}".format(
                training_state["last_epoch"] + 1,
                training_state["steps"],
                training_state["best_si_snr"]
            )
        )

    return training_state


def train_worker(rank, world_size, a, h):
    distributed = world_size > 1
    is_main = rank == 0

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    if distributed:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    seed = h.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    os.makedirs(a.checkpoint_path, exist_ok=True)

    use_visqol = h.use_visqol if hasattr(h, "use_visqol") else False
    audio_metrics = None

    if is_main and use_visqol:
        from losses.metrics import AudioMetrics
        audio_metrics = AudioMetrics(sample_rate=h.sampling_rate)

    generator = MPASCNet(h).to(device)

    if is_main:
        generator_params = sum(parameter.numel() for parameter in generator.parameters())

        print("Model Parameters: {:.3f}M".format(generator_params / 1e6))
        print(
            "GPU Number: {}, Batch Size Per GPU: {}, Global Batch Size: {}".format(
                world_size,
                h.batch_size,
                h.batch_size * world_size
            )
        )
        print("checkpoints directory : ", a.checkpoint_path)

    training_state = restore_training(a, generator, device)

    if distributed:
        generator = DDP(generator, device_ids=[rank], output_device=rank)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay)

    checkpoint = training_state["checkpoint"]

    if checkpoint is not None:
        if "optim_g" in checkpoint:
            optim_g.load_state_dict(checkpoint["optim_g"])

        if "scheduler_g" in checkpoint:
            scheduler_g.load_state_dict(checkpoint["scheduler_g"])

    trainset = MusicCodecDataset(
        a.train_dir, h.codec_type, h.codec_options, h.sampling_rate,
        h.segments, h.num_stems, h.snr_range, h.num_samples
    )

    if distributed:
        train_sampler = DistributedSampler(
            trainset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=h.seed, drop_last=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(trainset, batch_size=h.batch_size, sampler=train_sampler, shuffle=train_sampler is None, pin_memory=True, drop_last=True)

    if is_main:
        validset = MusicCodecEvalDataset(a.eval_dir)
        validation_loader = DataLoader(validset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    else:
        validation_loader = None

    run = None

    if is_main:
        wandb_config = dict(h)
        wandb_config["train_dir"] = a.train_dir
        wandb_config["eval_dir"] = a.eval_dir
        wandb_config["checkpoint_path"] = a.checkpoint_path
        wandb_config["training_epochs"] = a.training_epochs
        wandb_config["global_batch_size"] = h.batch_size * world_size

        run = wandb.init(
            project=h.wandb_project, entity=h.wandb_entity, name=h.wandb_name, mode=h.wandb_mode,
            dir=os.path.abspath(a.checkpoint_path), config=wandb_config, id=training_state["wandb_run_id"],
            resume="allow" if training_state["wandb_run_id"] is not None else None
        )

    steps = training_state["steps"]
    best_si_snr = training_state["best_si_snr"]
    si_snr_no_improve_epochs = training_state["si_snr_no_improve_epochs"]
    start_epoch = max(0, training_state["last_epoch"] + 1)

    generator_metric_names = (
        "loss_gen", "loss_mag", "loss_pha", "loss_ip",
        "loss_gd", "loss_iaf", "loss_com", "loss_stft",
        "loss_time"
    )

    metric_names = generator_metric_names

    try:
        for epoch in range(start_epoch, a.training_epochs):
            start = time.time()

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if is_main:
                print("Epoch: {}".format(epoch + 1))

            generator.train()

            train_metrics = {
                name: 0.
                for name in metric_names
            }

            train_bar = tqdm(train_loader, desc="Training", dynamic_ncols=True,disable=not is_main)
            for batch in train_bar:
                clean_audio, noisy_audio = batch
                clean_audio = clean_audio.to(device, non_blocking=True)
                noisy_audio = noisy_audio.to(device, non_blocking=True)

                batch_size, channels, _ = clean_audio.shape

                clean_mag, clean_pha, clean_com = stft_audio(clean_audio, h)
                noisy_mag, noisy_pha, _ = stft_audio(noisy_audio, h)

                mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)
                audio_g = istft_audio(mag_g, pha_g, h, batch_size, channels, clean_audio.shape[-1])
                mag_g_hat, pha_g_hat, com_g_hat = stft_audio(audio_g, h)

                optim_g.zero_grad(set_to_none=True)

                loss_g = generator_losses(
                    clean_audio, audio_g, clean_mag, clean_pha,
                    clean_com, mag_g, pha_g, com_g,
                    com_g_hat, mag_weight=h.mag_weight,
                    pha_weight=h.pha_weight, com_weight=h.com_weight,
                    stft_weight=h.stft_weight, time_weight=h.time_weight
                )

                loss_g["loss_gen"].backward()
                optim_g.step()

                for name in generator_metric_names:
                    train_metrics[name] += loss_g[name].item()

                if is_main:
                    train_bar.set_postfix({
                        "loss": "{:.3f}".format(loss_g["loss_gen"].item()),
                        "mag": "{:.3f}".format(loss_g["loss_mag"].item()),
                        "pha": "{:.3f}".format(loss_g["loss_pha"].item())
                    })

                steps += 1

            train_metrics = reduce_metrics(train_metrics, len(train_loader), device, distributed)
            scheduler_g.step()

            generator.eval()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stop_training = torch.zeros(1 ,dtype=torch.int64, device=device)

            if is_main:
                print(
                    "Epoch {:d}, Loss: {:4.3f}, Magnitude Loss: {:4.3f}, "
                    "Phase Loss: {:4.3f}, Complex Loss: {:4.3f}, "
                    "Time Loss: {:4.3f}, STFT Loss: {:4.3f}".format(
                        epoch + 1, train_metrics["loss_gen"], train_metrics["loss_mag"], train_metrics["loss_pha"],
                        train_metrics["loss_com"], train_metrics["loss_time"], train_metrics["loss_stft"]
                    )
                )

                val_sdr_tot = 0.
                val_si_snr_tot = 0.
                val_visqol_tot = 0.

                valid_bar = tqdm(validation_loader, desc="Validation", dynamic_ncols=True)
                validation_generator = unwrap_model(generator)
                with torch.no_grad():
                    for batch in valid_bar:
                        clean_audio, noisy_audio, _ = batch
                        clean_audio = clean_audio.to(device, non_blocking=True)
                        noisy_audio = noisy_audio.to(device, non_blocking=True)

                        batch_size, channels, _ = clean_audio.shape

                        noisy_mag, noisy_pha, _ = stft_audio(noisy_audio, h)
                        mag_g, pha_g, _ = validation_generator(noisy_mag, noisy_pha)

                        audio_g = istft_audio(mag_g, pha_g, h, batch_size, channels, clean_audio.shape[-1])

                        if use_visqol:
                            score_val = audio_metrics(clean_audio, audio_g)
                            sdr = score_val["sdr"].mean().item()
                            si_snr = score_val["si_snr"].mean().item()
                            visqol = score_val["visqol"].mean().item()

                            val_visqol_tot += visqol

                            valid_bar.set_postfix({
                                "sdr": "{:.3f}".format(sdr),
                                "si_snr": "{:.3f}".format(si_snr),
                                "visqol": "{:.3f}".format(visqol)
                            })
                        else:
                            sdr = cal_sdr(clean_audio, audio_g).mean().item()
                            si_snr = cal_si_snr(clean_audio, audio_g).mean().item()

                            valid_bar.set_postfix({
                                "sdr": "{:.3f}".format(sdr),
                                "si_snr": "{:.3f}".format(si_snr)
                            })

                        val_sdr_tot += sdr
                        val_si_snr_tot += si_snr

                val_sdr = val_sdr_tot / len(validation_loader)
                val_si_snr = val_si_snr_tot / len(validation_loader)

                if use_visqol:
                    val_visqol = val_visqol_tot / len(validation_loader)
                else:
                    val_visqol = 0.

                epoch_time = int(time.time() - start)

                print(
                    "Epoch {:d}, SDR: {:4.3f}, SI-SNR: {:4.3f}, "
                    "ViSQOL: {:4.3f}, Time: {:d} sec".format(
                        epoch + 1,
                        val_sdr,
                        val_si_snr,
                        val_visqol,
                        epoch_time
                    )
                )

                if val_si_snr > best_si_snr:
                    best_si_snr = val_si_snr
                    si_snr_no_improve_epochs = 0

                    best_state = {
                        "generator": unwrap_model(generator).state_dict(),
                        "best_si_snr": best_si_snr,
                        "steps": steps,
                        "epoch": epoch,
                        "wandb_run_id": run.id
                    }

                    save_checkpoint(
                        os.path.join(a.checkpoint_path, "best"),
                        best_state
                    )

                    run.summary["best_si_snr"] = best_si_snr
                    run.summary["best_epoch"] = epoch + 1
                else:
                    si_snr_no_improve_epochs += 1

                log_data = {
                    "epoch": epoch + 1
                }

                log_data.update({
                    "train/{}".format(name): value
                    for name, value in train_metrics.items()
                })

                log_data.update({
                    "validation/sdr": val_sdr,
                    "validation/si_snr": val_si_snr,
                    "validation/visqol": val_visqol,
                    "learning_rate/generator": optim_g.param_groups[0]["lr"],
                    "best_si_snr": best_si_snr,
                    "epoch_time": epoch_time
                })

                run.log(log_data, step=epoch + 1)

                latest_state = {
                    "generator": unwrap_model(generator).state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "scheduler_g": scheduler_g.state_dict(),
                    "steps": steps,
                    "epoch": epoch,
                    "best_si_snr": best_si_snr,
                    "si_snr_no_improve_epochs": si_snr_no_improve_epochs,
                    "wandb_run_id": run.id
                }

                save_checkpoint(
                    os.path.join(a.checkpoint_path, "latest"),
                    latest_state
                )

                print(
                    "Time taken for epoch {} is {} sec\n".format(
                        epoch + 1,
                        epoch_time
                    )
                )

                if si_snr_no_improve_epochs >= 20:
                    print("SI-SNR did not improve for 20 consecutive epochs. Stopping training.")
                    stop_training.fill_(1)

            if distributed:
                dist.broadcast(stop_training, src=0)

            if stop_training.item() == 1:
                break
    finally:
        if run is not None:
            run.finish()

        if distributed:
            dist.destroy_process_group()


def main():
    print("Initializing Training Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--checkpoint_path", default="cp_model")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--training_epochs", default=200, type=int)
    parser.add_argument("--resume_path", default=None, type=str)
    parser.add_argument("--no_resume", default=False, action="store_true")
    parser.add_argument("--finetune_from", default=None, type=str)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    h = AttrDict(json.loads(data))
    gpu_ids = h.gpu_ids if hasattr(h, "gpu_ids") else [0]

    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    build_env(a.config, "config.json", a.checkpoint_path)

    if torch.cuda.is_available() and len(gpu_ids) > 0:
        world_size = len(gpu_ids)

        if torch.cuda.device_count() != world_size:
            raise RuntimeError("GPU configuration error: requested {}, available {}".format( world_size,torch.cuda.device_count()))
    else:
        world_size = 1

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    if world_size > 1:
        mp.spawn(train_worker, args=(world_size, a, h), nprocs=world_size, join=True)
    else:
        train_worker(0, world_size, a, h)


if __name__ == "__main__":
    main()