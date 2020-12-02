from muspy_config import config
from muspy_train import Trainer
import os
from create_dataset import NoteRepresentationManager

if __name__ == "__main__":
    data = NoteRepresentationManager(resolution=config.resolution,
                                     tempo=config.tempo,
                                     pad_token=config.pad_token,
                                     sos_token=config.sos_token,
                                     bar_token=config.bar_token,
                                     eos_token=config.eos_token,
                                     num_values=config.num_values,
                                     time_first_token=config.time_first_token,
                                     pitch_first_token=config.pitch_first_token,
                                     duration_first_token=config.duration_first_token,
                                     velocity_first_token=config.velocity_first_token,
                                     use_velocity=config.use_velocity,
                                     log_file=config.dataset_converter_log_file,
                                     reconstruct_programs=config.reconstruct_programs,
                                     max_bar_length=config.max_bar_length)
    trainer = Trainer(model=None,
                      pad_token=config.pad_token,
                      device=config.device,
                      dataset_path=config.dataset_path,
                      test_size=config.test_size,
                      batch_size=config.batch_size,
                      n_workers=config.n_workers,
                      vocab_size=config.vocab_size,
                      n_epochs=config.n_epochs,
                      model_name=config.model_name,
                      plot_name=config.plot_name,
                      dataset=data)
    checkpoint = os.path.join("training_2020-12-02_14-03-00", "checkpoint_epoch_19.pt")
    sampled_dir = os.path.join("training_2020-12-02_14-03-00", "sampled")
    trainer.generate(checkpoint_path=checkpoint, sampled_dir=sampled_dir)
