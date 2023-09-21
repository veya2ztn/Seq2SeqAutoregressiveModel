import torch
import torch.distributed as dist
def config_of(model):
    unwrapper_model = model
    while hasattr(unwrapper_model, 'module'):
        unwrapper_model = unwrapper_model.module
    return unwrapper_model.config

class NanDetect:
    def __init__(self, logsys, use_amp):
        self.logsys = logsys
        self.nan_count = 0
        self.good_loss_count = 0
        self.default_amp = use_amp
        self.try_upgrade_times = 0

    def nan_diagnose_weight(self, model, loss, scaler):
        skip = torch.zeros_like(loss)
        downgrad_use_amp = torch.zeros_like(loss)
        upgrade_use_amp = torch.zeros_like(loss)
        nan_count = self.nan_count
        logsys = self.logsys
        if torch.isnan(loss):
            self.good_loss_count = 0
            # we will check whether weight has nan
            bad_weight_name = []
            bad_check = False
            for name, p in model.named_parameters():
                if torch.isnan(p).any():
                    bad_check = True
                    bad_weight_name.append(name)
            if bad_check:
                logsys.info(f"the value is nan in weight:{bad_weight_name}")
                raise NotImplementedError  # optuna.TrialPruned()
            else:
                nan_count += 1
                if nan_count > 5 and model.use_amp:
                    downgrad_use_amp += 1
                if nan_count > 10:
                    logsys.info("too many nan happened")
                    raise NotImplementedError  # optuna.TrialPruned()
                logsys.info(
                    f"detect nan, now at {nan_count}/10 warning level, pass....")
                skip += 1
            self.good_loss_count = 0
        else:
            self.good_loss_count += 1
            if self.good_loss_count > 5 and self.default_amp and self.try_upgrade_times < 5 and not model.use_amp:
                upgrade_use_amp += 1
                self.try_upgrade_times += 1
            nan_count = 0
        self.nan_count = nan_count
        if hasattr(model, 'module'):
            dist.all_reduce(skip)  # 0+0+0+0 = 0; 0 + 1 + 0 + 1 =1;
            dist.all_reduce(downgrad_use_amp)  # 0+0+0+0 = 0; 0 + 1 + 0 + 1 =1;
            # 0*0*0*0 = 0; 1*1**1 =1;
            dist.all_reduce(upgrade_use_amp,
                            torch.distributed.ReduceOp.PRODUCT)
        if downgrad_use_amp:
            logsys.info(
                f"detect nan loss during training too many times and we are now at `autograd.amp` mode. so we will turn off amp mode ")
            if model.use_amp:
                model.use_amp = False
                scaler._enabled = False
        elif upgrade_use_amp:
            logsys.info(
                f"detect nan loss during training too many times and we are now at `autograd.amp` mode. so we will turn off amp mode ")
            if model.use_amp:
                model.use_amp = True
                scaler._enabled = True  # notice the default scaler should be activated at initial
        return skip

    def nan_diagnose_grad(self, model, loss, scaler):
        skip = torch.zeros_like(loss)
        downgrad_use_amp = torch.zeros_like(loss)
        logsys = self.logsys
        nan_count = self.nan_count
        # we will check whether weight has nan
        bad_weight_name = []
        bad_check = False
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any():
                bad_check = True
                bad_weight_name.append(name)
        if bad_check:
            logsys.info(f"the value is nan in weight.grad:{bad_weight_name}")
            nan_count += 1
            if nan_count > 10:
                logsys.info("too many nan happened")
                raise
            logsys.info(
                f"detect nan, now at {nan_count}/10 warning level, pass....")
            skip += 1

        if hasattr(model, 'module'):
            dist.all_reduce(skip)  # 0+0+0+0 = 0; 0 + 1 + 0 + 1 =1;
            dist.all_reduce(downgrad_use_amp)  # 0+0+0+0 = 0; 0 + 1 + 0 + 1 =1;
        if downgrad_use_amp:
            if model.use_amp:
                model.use_amp = False
                scaler._enabled = False
                model.use_amp = bool(downgrad_use_amp.item()) and model.use_amp
        return skip