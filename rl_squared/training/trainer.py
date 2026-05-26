import os
import platform

import torch
import gymnasium
import wandb
from tqdm import tqdm

import rl_squared.utils.logging_utils as logging_utils
from rl_squared.training.experiment_config import ExperimentConfig
from rl_squared.learners.ppo import PPO

from rl_squared.utils.env_utils import make_vec_envs
from rl_squared.utils.training_utils import (
    sample_meta_episodes,
    save_checkpoint,
    timestamp,
)
from rl_squared.training.meta_batch_sampler import MetaBatchSampler

from rl_squared.networks.stateful.stateful_actor_critic import StatefulActorCritic


class Trainer:
    def __init__(
        self, experiment_config: ExperimentConfig, restart_checkpoint: str = None
    ):
        """
        Initialize an instance of a trainer for PPO.

        Args:
            experiment_config (ExperimentConfig): Params to be used for the trainer.
            restart_checkpoint (str): Checkpoint path from where to restart the experiment.
        """
        self.config = experiment_config

        # private
        self._device = None
        self._log_dir = None

        # restart
        self._restart_checkpoint = restart_checkpoint
        pass

    def train(
        self,
        is_dev: bool,
        enable_wandb: bool = True,
    ) -> None:
        """
        Train an agent based on the configs specified by the training parameters.

        Args:
            is_dev (bool): Whether to log the run statistics as a `dev` run.
            enable_wandb (bool): Whether to log to Wandb, `True` by default.

        Returns:
            None
        """
        # log
        self.save_params()

        if enable_wandb:
            wandb.login()
            project_suffix = "-dev" if is_dev else ""
            wandb.init(project=f"rl-squared{project_suffix}", config=self.config.dict)
            pass

        # seed
        torch.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed_all(self.config.random_seed)

        # clean
        logging_utils.cleanup_log_dir(self.log_dir)

        torch.set_num_threads(1)

        rl_squared_envs = make_vec_envs(
            self.config.env_name,
            self.config.env_configs,
            self.config.random_seed,
            self.config.num_processes,
            self.device,
        )

        actor_critic = StatefulActorCritic(
            rl_squared_envs.observation_space,
            rl_squared_envs.action_space,
            recurrent_state_size=256,
        ).to_device(self.device)

        ppo = PPO(
            actor_critic=actor_critic,
            clip_param=self.config.ppo_clip_param,
            opt_epochs=self.config.ppo_opt_epochs,
            num_minibatches=self.config.ppo_num_minibatches,
            value_loss_coef=self.config.ppo_value_loss_coef,
            entropy_coef=self.config.ppo_entropy_coef,
            actor_lr=self.config.actor_lr,
            critic_lr=self.config.critic_lr,
            eps=self.config.optimizer_eps,
            max_grad_norm=self.config.max_grad_norm,
        )

        current_iteration = 0

        # load
        if self._restart_checkpoint:
            checkpoint = torch.load(self._restart_checkpoint)
            current_iteration = checkpoint["iteration"]
            actor_critic.actor.load_state_dict(checkpoint["actor"])
            actor_critic.critic.load_state_dict(checkpoint["critic"])
            ppo.optimizer.load_state_dict(checkpoint["optimizer"])
            pass

        print(self._run_header(current_iteration, enable_wandb))

        progress = tqdm(
            range(current_iteration, self.config.policy_iterations),
            desc=self.config.env_name,
            unit="iter",
            dynamic_ncols=True,
        )

        for j in progress:
            # anneal
            if self.config.use_linear_lr_decay:
                ppo.anneal_learning_rates(j, self.config.policy_iterations)

            # sample
            meta_episode_batches, meta_train_reward_per_step = sample_meta_episodes(
                actor_critic,
                rl_squared_envs,
                self.config.meta_episode_length,
                self.config.meta_episodes_per_epoch,
                self.config.use_gae,
                self.config.gae_lambda,
                self.config.discount_gamma,
                self.device,
            )

            minibatch_sampler = MetaBatchSampler(meta_episode_batches, self.device)
            ppo_update = ppo.update(minibatch_sampler)

            mean_reward = meta_train_reward_per_step * self.config.meta_episode_length
            wandb_logs = {
                "meta_train/mean_policy_loss": ppo_update.policy_loss,
                "meta_train/mean_value_loss": ppo_update.value_loss,
                "meta_train/mean_entropy": ppo_update.entropy,
                "meta_train/approx_kl": ppo_update.approx_kl,
                "meta_train/clip_fraction": ppo_update.clip_fraction,
                "meta_train/explained_variance": ppo_update.explained_variance,
                "meta_train/mean_meta_episode_reward": mean_reward,
            }

            progress.set_postfix(
                reward=f"{mean_reward:.3f}",
                pi_loss=f"{ppo_update.policy_loss:.3f}",
                v_loss=f"{ppo_update.value_loss:.3f}",
                entropy=f"{ppo_update.entropy:.3f}",
                kl=f"{ppo_update.approx_kl:.4f}",
            )

            # save
            is_last_iteration = j == (self.config.policy_iterations - 1)
            checkpoint_name = str(timestamp()) if self.config.checkpoint_all else "last"

            if j % self.config.checkpoint_interval == 0 or is_last_iteration:
                save_checkpoint(
                    iteration=j,
                    checkpoint_dir=self.config.checkpoint_directory,
                    checkpoint_name=checkpoint_name,
                    actor=actor_critic.actor,
                    critic=actor_critic.critic,
                    optimizer=ppo.optimizer,
                )

            if enable_wandb:
                wandb.log(wandb_logs)

        # end
        if enable_wandb:
            wandb.finish()

    def _run_header(self, current_iteration: int, enable_wandb: bool) -> str:
        device = self.device
        if device.type == "cuda":
            hw = torch.cuda.get_device_name(device.index or 0)
            device_str = f"cuda:{device.index or 0}  {hw}"
        elif device.type == "mps":
            try:
                import subprocess
                chip = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL
                ).decode().strip()
            except Exception:
                chip = "Apple Silicon"
            device_str = f"mps  {chip}"
        else:
            device_str = f"cpu  {platform.processor() or platform.machine()}"

        c = self.config
        w = 52
        sep = "─" * w
        lines = [
            sep,
            f"  env          {c.env_name}",
            f"  device       {device_str}",
            f"  processes    {c.num_processes}  (env workers)",
            f"  iterations   {current_iteration} → {c.policy_iterations}"
            f"  ·  seed {c.random_seed}",
            f"  meta-ep      {c.meta_episode_length} steps"
            f"  ·  {c.meta_episodes_per_epoch} per iter",
            f"  ppo          epochs {c.ppo_opt_epochs}"
            f"  ·  minibatches {c.ppo_num_minibatches}"
            f"  ·  clip {c.ppo_clip_param}",
            f"  lr           actor {c.actor_lr}  ·  critic {c.critic_lr}",
            f"  torch        {torch.__version__}"
            f"  ·  gymnasium {gymnasium.__version__}",
            f"  wandb        {'on' if enable_wandb else 'off'}",
            sep,
        ]
        return "\n".join(lines)

    @property
    def log_dir(self) -> str:
        """
        Returns the path for training logs.

        Returns:
            str
        """
        if not self._log_dir:
            self._log_dir = os.path.expanduser(self.config.log_dir)

        return self._log_dir

    def save_params(self) -> None:
        """
        Save experiment_config to the logging directory.

        Returns:
          None
        """
        self.config.save()
        pass

    @property
    def device(self) -> torch.device:
        """
        Torch device to use for training and optimization.

        Returns:
          torch.device
        """
        if isinstance(self._device, torch.device):
            return self._device

        use_cuda = self.config.use_cuda and torch.cuda.is_available()
        if use_cuda and self.config.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        if use_cuda:
            self._device = torch.device(f"cuda:{self.config.cuda_device_id}")
        elif self.config.use_cuda and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        return self._device
