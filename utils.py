import os
import torch
from torch import nn


class AverageMeter:
    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg


def generate_noise(n_samples, dim, type='normal'):
    if type == 'normal':
        return torch.randn(n_samples, dim)
    elif type == 'uniform':
        return torch.rand(n_samples, dim)
    else:
        raise ValueError(f"Invalid type: {type}. Must be one of ['uniform', 'normal']")


def weights_init(m):
    """
    Initialize weights of given model as described in the DCGAN paper.
    Args:
        m (nn.Module): Pytorch model with trainable parameters
    """
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, 0.0, 0.02)


class WandbLogger(object):
    """
    Log training runs, datasets, models, and predictions to Weights & Biases.
    This logger sends information to W&B at wandb.ai.
    By default, this information includes hyperparameters,
    system configuration and metrics, model metrics,
    and basic data metrics and analyses.
    For more information, please refer to:
    https://docs.wandb.ai/guides/track
    """
    def __init__(self,
                 project=None,
                 name=None,
                 id=None,
                 entity=None,
                 save_dir=None,
                 config=None,
                 **kwargs):
        """
        Args:
            project (str): wandb project name.
            name (str): wandb run name.
            id (str): wandb run id.
            entity (str): wandb entity name.
            save_dir (str): save directory.
            config (dict): config dict.
            **kwargs: other kwargs.
        """
        try:
            import wandb
            self.wandb = wandb
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "wandb is not installed."
                "Please install wandb using pip install wandb"
                )

        self.project = project
        self.name = name
        self.id = id
        self.save_dir = save_dir
        self.config = config
        self.kwargs = kwargs
        self.entity = entity
        self._run = None
        self._wandb_init = dict(
            project=self.project,
            name=self.name,
            id=self.id,
            entity=self.entity,
            dir=self.save_dir,
            resume="allow"
        )
        self._wandb_init.update(**kwargs)

        _ = self.run

        if self.config:
            self.run.config.update(self.config)
        self.run.define_metric("epoch")
        self.run.define_metric("val/", step_metric="epoch")

    @property
    def run(self):
        if self._run is None:
            if self.wandb.run is not None:
                logger.info(
                    "There is a wandb run already in progress "
                    "and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()`"
                    "before instantiating `WandbLogger`."
                )
                self._run = self.wandb.run
            else:
                self._run = self.wandb.init(**self._wandb_init)
        return self._run

    def log_metrics(self, metrics, step=None):
        """
        Args:
            metrics (dict): metrics dict.
            step (int): step number.
        """
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v.item()

        if step is not None:
            self.run.log(metrics, step=step)
        else:
            self.run.log(metrics)

    def save_checkpoint(self, save_dir, model_name, is_best):
        """
        Args:
            save_dir (str): save directory.
            model_name (str): model name.
            is_best (bool): whether the model is the best model.
        """
        filename = os.path.join(save_dir, model_name + "_ckpt.pth")
        artifact = self.wandb.Artifact(
            name=f"model-{self.run.id}",
            type="model"
        )
        artifact.add_file(filename, name="model_ckpt.pth")

        aliases = ["latest"]

        if is_best:
            aliases.append("best")

        self.run.log_artifact(artifact, aliases=aliases)

    def finish(self):
        self.run.finish()


def save_checkpoint(model, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")

    ckpt_state = {
        "model": model.state_dict(),
    }
    torch.save(ckpt_state, filename)

