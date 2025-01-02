from stable_baselines3.common.callbacks import BaseCallback
import tensorflow as tf
import os

class CustomLoggingCallback(BaseCallback):
    """
    Logs hyperparameters and metrics to TensorBoard's HPARAMS tab.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.writer = None

    def _on_training_start(self) -> None:
        # Retrieve tensorboard_log path from the model
        log_dir = self.model.tensorboard_log if self.model.tensorboard_log else "./logs/"

        # Initialize the TensorBoard writer
        self.writer = tf.summary.create_file_writer(log_dir)

        # Log hyperparameters at the start of training
        with self.writer.as_default():
            hparams = {
                'learning_rate': self.model.learning_rate,
                'gamma': self.model.gamma if hasattr(self.model, 'gamma') else 'N/A',
                'policy': self.model.policy.__class__.__name__
            }
            tf.summary.text("Hyperparameters", str(hparams), step=0)
            self.writer.flush()

    def _on_step(self) -> bool:
        # Log rewards at each step
        if self.writer:
            if self.num_timesteps % 100 == 0:
                with self.writer.as_default():
                    tf.summary.scalar("rollout/reward", self.locals['rewards'][0], step=self.num_timesteps)
                    self.logger.dump(self.num_timesteps)
                    self.writer.flush()
        return True

    def _on_training_end(self) -> None:
        # Close the writer to finalize logging
        if self.writer:
            self.writer.close()
