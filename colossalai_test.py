from colossalai.utils import is_using_pp
from colossalai.pipeline.pipelinable import PipelinableContext
from colossalai.logging import get_dist_logger
from colossalai.context import ParallelMode
from titans.model.vit.vit import _create_vit_model
from colossalai.trainer import Trainer, hooks
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import _create_vision_transformer
from tqdm import tqdm

import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import CrossEntropyLoss
from colossalai.nn._ops import *
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.parallel.data_parallel import ColoDDP
from colossalai.tensor import ComputePattern, ComputeSpec, DistSpecManager, ProcessGroup, ShardSpec
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext


def init_1d_row_for_linear_weight_spec(model, world_size: int):
    pg = ProcessGroup(tp_degree=world_size)
    spec = (ShardSpec([-1], [pg.tp_world_size()]),
            ComputeSpec(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        for n, p in model.named_parameters():
            if 'weight' in n and 'norm' not in n and 'patch_embed.proj.weight' not in n:
                p.set_process_group(pg)
                p.set_tensor_spec(*spec)


# Similarly, it's col split for Linear but row split for others.
def init_1d_col_for_linear_weight_bias_spec(model, world_size: int):
    pg = ProcessGroup(tp_degree=world_size)
    spec = (ShardSpec([0], [pg.tp_world_size()]),
            ComputeSpec(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        for n, p in model.named_parameters():
            if ('weight' in n or 'bias' in n) and 'norm' not in n and ('patch_embed.proj.weight' not in n
                                                                       and 'patch_embed.proj.bias' not in n):
                p.set_process_group(pg)
                p.set_tensor_spec(*spec)


def init_spec_func(model, tp_type):
    world_size = torch.distributed.get_world_size()
    if tp_type == 'row':
        init_1d_row_for_linear_weight_spec(model, world_size)
    elif tp_type == 'col':
        init_1d_col_for_linear_weight_bias_spec(model, world_size)
    else:
        raise NotImplemented

from train.pretrain import get_args

from train.pretrain import get_train_and_valid_dataset,build_model,parse_default_args,create_logsys

def get_the_args(ckpt_path):
    
    if len(os.listdir(ckpt_path))==0:
        return 
    args = get_args(os.path.join(ckpt_path,'config.json'))
    args.SAVE_PATH = "debug"
    return args

def train_weather_forecast_colo():

    parser = colossalai.get_default_parser()
    parser.add_argument('--resume_from', default=False, action='store_true')
    parser.add_argument('--dummy_data', default=False, action='store_true')

    args = parser.parse_args()
    colossalai.launch_from_torch(config=args.config)
    use_ddp = gpc.config.USE_DDP

    disable_existing_loggers()

    logger = get_dist_logger()
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    logger.info('Build data loader', ranks=[0])
    atom_args = get_the_args("checkpoints/WeathBench32x64/CK_LgNet/ts_2_pretrain-2D706N_per_6_step/02_24_09_58_53396-seed_73001")
    atom_args.use_wandb="off"
    atom_args.SAVE_PATH = "debug"
    
    atom_args= parse_default_args(atom_args)
    logsys = create_logsys(atom_args)
    atom_args.distributed = True 
    train_dataset, valid_dataset, train_dataloader, test_dataloader = get_train_and_valid_dataset(atom_args)
    
    logger.info('Build model', ranks=[0])
    atom_args.distributed = False 
    with ColoInitContext(device=get_current_device()):
        model = build_model(atom_args)
    
    init_spec_func(model, gpc.config.TP_TYPE)

    world_size = torch.distributed.get_world_size()
    model = ColoDDP(module=model, process_group=ProcessGroup(tp_degree=world_size))
    logger.info('Build criterion, optimizer, lr_scheduler', ranks=[0])
    # optimizer = HybridAdam(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)
    optimizer = torch.optim.SGD(model.parameters(), lr=gpc.config.LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,total_steps=gpc.config.NUM_EPOCHS,
                        warmup_steps=gpc.config.WARMUP_EPOCHS)

    start_epoch = 0


    for epoch in range(start_epoch, gpc.config.NUM_EPOCHS):
        model.train()
        for index, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False):
            x, y = x.cuda().float(), y.cuda().float()
            
            output = model(x)
            loss = criterion(output, y)
            loss = loss / gpc.config.gradient_accumulation
            if use_ddp:
                model.backward(loss)
            else:
                loss.backward()
            if (index + 1) % gpc.config.gradient_accumulation == 0:
                optimizer.step()
                if use_ddp:
                    model.zero_grad()
                else:
                    optimizer.zero_grad()

        logger.info(
            f"Finish Train Epoch [{epoch+1}/{gpc.config.NUM_EPOCHS}] loss: {loss.item():.3f} lr: {optimizer.state_dict()['param_groups'][0]['lr']}",
            ranks=[0])

        model.eval()
        test_loss = 0
        correct = 0
        test_sum = 0
        with torch.no_grad():
            for index, (x, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=False):
                x, y = x.cuda().float(), y.cuda().float()
                output = model(x)
                test_loss += F.cross_entropy(output, y, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                test_sum += y.size(0)

        test_loss /= test_sum
        logger.info(
            f"Finish Test Epoch [{epoch+1}/{gpc.config.NUM_EPOCHS}] loss: {test_loss:.3f} Accuracy: [{correct}/{test_sum}]({correct/test_sum:.3f})",
            ranks=[0])

        lr_scheduler.step()


class DummyDataloader():

    def __init__(self, length, batch_size):
        self.length = length
        self.batch_size = batch_size

    def generate(self):
        data = torch.rand(self.batch_size, 68, 32, 64)
        label = torch.randint(low=0, high=10, size=(self.batch_size,))
        return data, label

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.step < self.length:
            self.step += 1
            return self.generate()
        else:
            raise StopIteration

    def __len__(self):
        return self.length

def train_weather_forecast():
    # launch from torch
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    colossalai.launch_from_torch(config=args.config)

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    use_pipeline = is_using_pp()

    
    atom_args = get_the_args(
        "checkpoints/WeathBench32x64/CK_LgNet/ts_2_pretrain-2D706N_per_6_step/02_24_09_58_53396-seed_73001")
    atom_args.use_wandb = "off"
    atom_args.SAVE_PATH = "debug"

    atom_args = parse_default_args(atom_args)
    logsys = create_logsys(atom_args)
    atom_args.distributed = False 
    if use_pipeline:
        pipelinable = PipelinableContext()
        with pipelinable:
            model = build_model(atom_args)
        pipelinable.to_layer_list()
        pipelinable.policy = "uniform"
        model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))
    else:
        model = build_model(atom_args)
    

    # count number of parameters
    total_numel = 0
    for p in model.parameters():
        total_numel += p.numel()
    if not gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_stage = 0
    else:
        pipeline_stage = gpc.get_local_rank(ParallelMode.PIPELINE)
    logger.info(
        f"number of parameters: {total_numel} on pipeline stage {pipeline_stage}")

    # use synthetic dataset
    # we train for 10 steps and eval for 5 steps per epoch
    atom_args.distributed = False
    #train_dataset, valid_dataset, train_dataloader, test_dataloader = get_train_and_valid_dataset(atom_args)
    train_dataloader = DummyDataloader(
        length=10, batch_size=gpc.config.BATCH_SIZE)
    test_dataloader = DummyDataloader(
        length=5, batch_size=gpc.config.BATCH_SIZE)
    # create loss function
    criterion = CrossEntropyLoss(label_smoothing=0.1)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    # create lr scheduler
    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=gpc.config.NUM_EPOCHS,
                                           warmup_steps=gpc.config.WARMUP_EPOCHS)

    # initialize
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model=model,optimizer=optimizer,criterion=criterion,train_dataloader=train_dataloader,
                                         test_dataloader=test_dataloader)

    logger.info("Engine is built", ranks=[0])

    for epoch in range(gpc.config.NUM_EPOCHS):
        # training
        engine.train()
        data_iter = iter(train_dataloader)

        if gpc.get_global_rank() == 0:
            description = 'Epoch {} / {}'.format(epoch, gpc.config.NUM_EPOCHS)
            progress = tqdm(range(len(train_dataloader)), desc=description)
        else:
            progress = range(len(train_dataloader))
        for _ in progress:
            engine.zero_grad()
            engine.execute_schedule(data_iter, return_output_label=False)
            engine.step()
            lr_scheduler.step()
    gpc.destroy()
if __name__ == '__main__':
    train_weather_forecast()
