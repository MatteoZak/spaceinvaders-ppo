import os
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Wrapper unificato per W&B e TensorBoard.
    Usa W&B se track=True e wandb è disponibile, altrimenti TensorBoard.
    """
    def __init__(self, config, run_name):
        self.use_wandb = False
        if config.get("track", False):
            try:
                import wandb
                wandb.init(
                    project=config.get("wandb_project", "dlai-ppo"),
                    name=run_name,
                    config=config,
                    sync_tensorboard=True,
                )
                self.use_wandb = True
            except Exception as e:
                print(f"W&B non disponibile ({e}), uso TensorBoard")

        log_dir = os.path.join("runs", run_name)
        self.writer = SummaryWriter(log_dir)
        print(f"Logging su: {log_dir}")

    def log(self, metrics: dict, step: int):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close(self):
        self.writer.close()
        if self.use_wandb:
            import wandb
            wandb.finish()
