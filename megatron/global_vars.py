# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""

import argparse
import os
import sys
import torch
import typing

from megatron import dist_signal_handler
from megatron.tokenizer import build_tokenizer
from .microbatches import build_num_microbatches_calculator
from .timers import Timers

_GLOBAL_ARGS = None
_GLOBAL_RETRO_ARGS = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_WANDB_WRITER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None
_GLOBAL_SIGNAL_HANDLER = None


def get_args() -> argparse.Namespace:
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return typing.cast(argparse.Namespace, _GLOBAL_ARGS)


def get_retro_args():
    """Return retro arguments."""
    return _GLOBAL_RETRO_ARGS


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()  # type: ignore


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()  # type: ignore


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples,  # type: ignore
                                               consistency_check)


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_wandb_writer():
    return _GLOBAL_WANDB_WRITER


def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return typing.cast(Timers, _GLOBAL_TIMERS)


def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    return _GLOBAL_SIGNAL_HANDLER


def _set_signal_handler():
    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    _GLOBAL_SIGNAL_HANDLER = dist_signal_handler.DistributedSignalHandler().__enter__()


def set_global_variables(args: argparse.Namespace, build_tokenizer=True) -> None:
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""

    assert args is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    set_args(args)

    _build_num_microbatches_calculator(args)
    if build_tokenizer:
        _ = _build_tokenizer(args)
    _set_tensorboard_writer(args)
    _set_wandb_writer(args)
    _set_adlr_autoresume(args)
    _set_timers(args)

    if args.exit_signal_handler:
        _set_signal_handler()


def set_args(args: argparse.Namespace) -> None:
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def set_retro_args(retro_args):
    global _GLOBAL_RETRO_ARGS
    _GLOBAL_RETRO_ARGS = retro_args


def _build_num_microbatches_calculator(args: argparse.Namespace) -> None:

    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _ensure_var_is_not_initialized(_GLOBAL_NUM_MICROBATCHES_CALCULATOR,
                                   'num microbatches calculator')

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(
        args)


def _build_tokenizer(args: argparse.Namespace):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def rebuild_tokenizer(args: argparse.Namespace):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args: argparse.Namespace):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')

    if hasattr(args, 'tensorboard_dir') and args.tensorboard_dir and args.rank == (args.world_size - 1):
        try:
            from torch.utils.tensorboard.writer import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir,
                max_queue=args.tensorboard_queue_size)
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)


def _set_wandb_writer(args: argparse.Namespace) -> None:
    """Set wandb writer."""
    global _GLOBAL_WANDB_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER, "wandb writer")  # type: ignore

    if (
        hasattr(args, "wandb_name")
        and (args.wandb_name or args.wandb_id)
        and args.rank == (args.world_size - 1)
    ):
        try:
            import wandb
            from datetime import datetime

            now = datetime.now()
            now = now.strftime("%Y-%m-%d-%H-%M-%S")
            exp_name = args.wandb_name + "-" + now
            entity: str = args.wandb_entity
            wandb_input = {
                "entity": entity,
                "name": exp_name,
                "config": args,
                "project": args.wandb_project,
            }
            if args.wandb_id is not None:
                wandb_input["id"] = args.wandb_id
                wandb_input["resume"] = "must"
            wandb.init(**wandb_input)
            _GLOBAL_WANDB_WRITER = True
            print("> wandb ...")
        except ModuleNotFoundError:
            print(
                "WARNING: wandb writing requested but is not "
                "available (are you using PyTorch 1.1.0 or later?), "
                "no wandb logs will be written.",
                flush=True,
            )


def _set_adlr_autoresume(args: argparse.Namespace):
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, 'adlr autoresume')

    if args.adlr_autoresume:
        if args.rank == 0:
            print('enabling autoresume ...', flush=True)
        sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
        try:
            from userlib.auto_resume import AutoResume  # type: ignore
        except BaseException:
            print('ADLR autoresume is not available, exiting ...')
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers(args: argparse.Namespace):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)
