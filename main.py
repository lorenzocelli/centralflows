import json
import os
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal, Optional, Set, Union

import git
import torch
import tyro
from tqdm import trange
from tyro.conf import arg, subcommand

from src import loggers
from src.architectures import CNN, MLP, VIT, LSTM, Mamba, Transformer, Resnet
from src.datasets import CIFAR10, SST2, Sorting, Copying, Moons, Circles, Classification #,#SparseParity, FlattenedMNIST
from src.functional import FunctionalModel
from src.loss_function import SupervisedLossFunction
from src.processes import (
    WeightGroup,
    CentralFlow,
    CentralFlowConfig,
    DiscreteProcess,
    DiscreteProcessConfig,
    MidpointProcess,
    MidpointProcessConfig,
    StableFlow,
    StableFlowConfig,
    EigConfig
)
from src.saving import Checkpointer, DataSaver, LoadOptions
from src.update_rules import GradientDescent, RMSProp, ScalarRMSProp, Muon
from src.utils import convert_dataclasses, create_all_layer_masks, create_layers_mask, print_layer_masks, flatten_pytree

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(True)

ValidOpt = Union[
    Annotated[GradientDescent, subcommand("gd")],
    Annotated[ScalarRMSProp, subcommand("scalar_rmsprop")],
    Annotated[RMSProp, subcommand("rmsprop")],
    Annotated[Muon, subcommand("muon")]
]
ValidData = Union[CIFAR10, SST2, Sorting, Copying, Moons, Circles, Classification]#, FlattenedMNIST, SparseParity]
ValidArch = Union[CNN, MLP, VIT, LSTM, Mamba, Transformer, Resnet]
ValidRuns = Set[Literal["discrete", "midpoint", "central", "stable", "stationary"]]

def main(
    # opt_: ValidOpt,    # which optimization algorithm to use # TODO make optimizer configurable
    data: ValidData,  # which dataset to train on
    arch: ValidArch,  # which architecture to train
    runs: ValidRuns,  # which processes (e.g. discrete alg, central flow, stable flow) to run
    eig_config: Annotated[EigConfig, arg(name="eig")],                        # global config for eigenvalues
    discrete_config: Annotated[DiscreteProcessConfig, arg(name="discrete")],  # config for discrete process
    midpoint_config: Annotated[MidpointProcessConfig, arg(name="midpoint")],  # config for midpoint process
    stable_config: Annotated[StableFlowConfig, arg(name="stable")],           # config for stable flow
    central_config: Annotated[CentralFlowConfig, arg(name="central")],        # config for central flow
    checkpointer: Annotated[Checkpointer, arg(name="checkpoint")],            # settings for saving checkpoints
    load: Annotated[LoadOptions, arg(name="load")],                           # optionally, settings for loading checkoints
    warm_start: int = -1,           # how many steps to warm-start for
    steps: int = 50,                # how many steps to train for
    device: str = "cuda",           # cuda or cpu
    seed: int = 0,                  # random seed
    expid: Optional[str] = None,    # optionally, an experiment id (defaults to a random UUID)
):
    print("loaded cli")
    
    # collect configs that were passed in
    config = convert_dataclasses(locals())
    config["git_hash"] = git.Repo(".").git.rev_parse("HEAD")
    config["cmd"] = " ".join(sys.argv)
    
    # experiment id defaults to random uuid
    expid = expid or uuid.uuid4().hex
    
    # create experiment folder
    folder = _create_experiment_folder(expid)
    print("Saving data to: ", folder, flush=True)
    
    # dump configs to json file
    with open(folder / "config.json", "w") as config_file:
        json.dump(config, config_file, indent=4)
        
    # initialize checkpointer
    checkpointer.init(folder / "checkpoints")

    # set random seed
    torch.manual_seed(seed)
    
    # load the dataset
    print("Loading Data")
    dataset = data.load(device=device)
    
    # instantiate the model as a PyTorch module
    model = arch.create(dataset.input_shape, dataset.output_dim).to(device)
    
    # make the model functional. 'model_fn' is a functional version of the network; 'w' are the initial weights
    w, model_fn = FunctionalModel.make_functional(model)
    
    print("\n=== Parameter Structure ===")
    params_dict = model_fn.unflatten(w)

    def print_param_structure(d, prefix=''):
        """Recursively print parameter names and shapes."""
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, torch.Tensor):
                shape_str = str(tuple(value.shape))
                print(f"{full_key:50s} shape={shape_str:<20s} numel={value.numel():8d}")
            elif isinstance(value, dict):
                print_param_structure(value, full_key)

    print_param_structure(params_dict)
    print("=" * 80 + "\n")

    masks = create_all_layer_masks(w, model_fn)
    print_layer_masks(masks, params_dict)

    # TODO: allow configuring the optimizer from command line
    groups = {}

    # 1. Identify 2D Params and their shapes for Muon
    muon_mask = torch.zeros_like(w, dtype=torch.bool)
    shapes_2d = []

    # Iterate in order of keys to ensure deterministic flattened order
    for key, param in params_dict.items():
        if param.ndim >= 2 and "embed" not in key:  # Muon Criteria
            # Get mask for this param
            param_mask_dict = create_layers_mask([key], params_dict)
            flat_m, _ = flatten_pytree(param_mask_dict)
            muon_mask |= flat_m
            shapes_2d.append(param.shape)

    # 2. Create Groups

    # Muon Group
    if shapes_2d:
        # We must initialize state ONLY on the masked weights
        w_muon = w[muon_mask]

        opt_muon = Muon(lr=0.02, shapes=shapes_2d)
        state_muon = opt_muon.initialize_state(w_muon)  # Pass subset w

        groups["muon"] = WeightGroup(opt=opt_muon, state=state_muon, mask=muon_mask)

    # Non-Muon Group (SGD for 1D params)
    non_muon_mask = ~muon_mask
    if non_muon_mask.any():
        w_other = w[non_muon_mask]
        opt_other = GradientDescent(lr=0.01)
        state_other = opt_other.initialize_state(w_other)
        groups["other"] = WeightGroup(opt=opt_other, state=state_other, mask=non_muon_mask)

    print(f"\n=== Created {len(groups)} Weight Groups ===")
    for name, group in groups.items():
        num_params = group.mask.sum().item()
        print(f"  {name:20s} | optimizer: {type(group.opt).__name__:20s} | params: {num_params:8,}")
    print("=" * 80 + "\n")
    
    # put together loss function
    loss_fn = SupervisedLossFunction(
        model_fn=model_fn, criterion=dataset.criterion_fn, batches=dataset.trainset
    )
    
    # these are loggers that are called on each individual process 
    process_loggers = {
        k: [
            loggers.LossAndAccuracy(model_fn, dataset, split="train"),   # log train loss/acc
            loggers.LossAndAccuracy(model_fn, dataset, split="test"),    # log test loss/acc
            loggers.OutputLogger(model_fn, dataset),                     # log network output
            loggers.EigLogger(),                                         # log top eigenvalues of effective Hessian
            loggers.RawEigLogger(),                                      # log top eigenvalues of "raw" Hessian
            loggers.GradientLogger(),                                    # log gradient sq entries and sq norm
            loggers.StateLogger(),                                       # log optimizer state
            loggers.CentralFlowPredictionLogger(),                       # log central flow predictions for time-averages
        ]
        for k in runs
    }
    
    # these are loggers that are called collectively on the set of all processes
    group_loggers = [
        loggers.DistanceLogger(),        # log distance between each pair of processes
        loggers.EmpiricalDeltaLogger(),  # log delta between the central flow and the other processes
    ]                                    # along the central flow's top eigenvectors
    
    initial_step = 0

    # warm start if appropriate
    if warm_start > 0:
        if load.path is not None:
            print("Skipping warm start because checkpoint file is provided")
        else:
            # TODO: this should be done per-group  
            raise NotImplementedError("Warm starting not yet implemented")
            print(f"Warm starting for {warm_start} steps")
            for _ in trange(warm_start):
                w, state = opt.update(w, state, loss_fn.D(w))
            initial_step = warm_start


    # initialize processes
    processes = {}
    kwargs = dict(loss_fn=loss_fn, w=w, groups=groups, eig_config=eig_config)
    if "discrete" in runs:
        processes["discrete"] = DiscreteProcess(**kwargs, config=discrete_config)
    if "midpoint" in runs:
        processes["midpoint"] = MidpointProcess(**kwargs, config=midpoint_config)
    if "central" in runs:
        processes["central"] = CentralFlow(**kwargs, config=central_config)
    if "stable" in runs:
        processes["stable"] = StableFlow(**kwargs, config=stable_config)

    # load from checkpoint, if appropriate
    if load.path is not None:
        print(f"Loading Checkpoint from {load.path}")
        initial_step = checkpointer.load(load, processes)

    # main training loop - run all processes in parallel
    print(
        f"Running concurrent processes from step {initial_step} to {initial_step + steps}"
    )
    with DataSaver(
        folder / "data.hdf5", initial_step=initial_step, total_steps=steps
    ) as data_saver:
        for i in trange(steps):
            # TODO: this currently is not used anywhere
            # step = initial_step + i
                        
            # save checkpoint if appropriate
            checkpointer.maybe_checkpoint(i, processes)
            
            # collected data will go here
            out = defaultdict(lambda: {})
            
            # for each process: prepare for the step
            for name in processes:
                process_log = processes[name].prepare()
                out[name].update(process_log)
            
            # run group loggers
            for logger in group_loggers:
                out.update(logger.log(processes))
                
            # run individual loggers
            for name in processes:
                for logger in process_loggers[name]:
                    out[name].update(logger.log(processes[name]))
                                        
            # for each process: take the step
            for name in processes:
                processes[name].step()
                
            # save data to disk
            data_saver.save(i, out)


def _create_experiment_folder(expid: int) -> Path:
    """Create the folder where saved data and checkpoints will be stored."""
    experiment_dir = Path(os.environ.get("EXPERIMENT_DIR", "experiments"))
    folder = experiment_dir / expid
    if folder.exists():
        override = input(
            f"Directory {folder} already exists. Do you want to override it? (y/N): "
        ).lower()
        if override == 'y':
            shutil.rmtree(folder)
        else:
            raise ValueError(f"Directory {folder} already exists")
    folder.mkdir(parents=True)
    return folder

if __name__ == "__main__":
    args = tyro.cli(main, config=[tyro.conf.ConsolidateSubcommandArgs])
