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
from src.architectures import CNN, MLP, Linear, UnbiasedMLP, VIT, LSTM, Mamba, Transformer, Resnet
from src.datasets import CIFAR10, SST2, Sorting, Copying, Moons, Circles, Classification, Regression #,#SparseParity, FlattenedMNIST
from src.functional import FunctionalModel
from src.loss_function import SupervisedLossFunction
from src.processes import (
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
from src.update_rules import GradientDescent, RMSProp, ScalarRMSProp, CompositeUpdateRule
from src.utils import convert_dataclasses

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(True)

ValidOpt = Union[
    Annotated[GradientDescent, subcommand("gd")],
    Annotated[ScalarRMSProp, subcommand("scalar_rmsprop")],
    Annotated[RMSProp, subcommand("rmsprop")],
    Annotated[CompositeUpdateRule, subcommand("comp")],
]
ValidData = Union[CIFAR10, SST2, Sorting, Copying, Moons, Circles, Classification, Regression]#, FlattenedMNIST, SparseParity]
ValidArch = Union[CNN, MLP, UnbiasedMLP, Linear, VIT, LSTM, Mamba, Transformer, Resnet]
ValidRuns = Set[Literal["discrete", "midpoint", "central", "stable", "stationary"]]

def main(
    opt: ValidOpt,    # which optimization algorithm to use
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
    
    # initialize optimizer state
    state = opt.initialize_state(w, model_fn.unflatten) 
    
    # put together loss function
    loss_fn = SupervisedLossFunction(
        model_fn=model_fn, criterion=dataset.criterion_fn, batches=dataset.trainset
    )
    
    # these are loggers that are called on each individual process 
    process_loggers = {
        k: [
            loggers.LossAndAccuracy(model_fn, dataset, split="train"),   # log train loss/acc
            loggers.LossAndAccuracy(model_fn, dataset, split="test"),    # log test loss/acc
            loggers.OptimalErrorLogger(model_fn, dataset),               # log distance to optimal weights if known
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
            print(f"Warm starting for {warm_start} steps")
            for _ in trange(warm_start):
                w, state = opt.update(w, state, loss_fn.D(w))
            initial_step = warm_start


    # initialize processes
    processes = {}
    kwargs = dict(loss_fn=loss_fn, w=w, state=state, opt=opt, eig_config=eig_config)
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
