import glob
import torch
import pytorch_lightning as pl

from utils.data_manager import DataManager
from utils.data_loader import LIDataLoader
from utils.result_logger import save_test_log
from model.define_model import load_pretrained_encoder
from model.model_interface import ModelInterface, define_callback

class ModelTraining:
    def __init__(self, config, project_name, 
                 is_weak=True):
        self.config = config
        self.project_name = project_name
        self.is_weak = is_weak
        if self.is_weak:
            self.aug_path = None
        else:
            self.aug_path = f"./results/aug-data_{self.project_name}/*.jpg"

    def run(self):
        self.prepare_data()
        self.prepare_trainer()
        self.train_and_eval()

    def prepare_data(self):
        self.data_manager = DataManager(
            aug_path=self.aug_path,
            is_sampling=self.config['is_sampling'],
            sampling_rate=self.config['sampling_rate'],
            sampling_random_state=self.config['sampling_random_state'],
            sample_only_pos=self.config['sample_only_pos']
        )
        
        self.train_df, self.valid_df, self.test_df = self.data_manager.run()
        
        self.train_dataloader, self.valid_dataloader, self.test_dataloader, self.valid_transform = LIDataLoader(self.train_df, self.valid_df, self.test_df).run()
        # For scheduling learning rate using cosine annealing 
        self.len_train_dataloader = len(self.train_dataloader)
        
    def prepare_trainer(self):
        self.model = load_pretrained_encoder()
        self.model_interface = ModelInterface(model=self.model, 
                                            learning_rate=self.config['learning_rate'], 
                                            len_train_dataloader=self.len_train_dataloader)
        
        if self.is_weak:
            self.weight_path = self.project_name + "_weak"       
        else:
            self.weight_path = self.project_name + "_strong"
        callbacks = define_callback(self.weight_path)
        
        self.trainer = pl.Trainer(
            max_epochs=self.config['epochs'],
            accelerator=self.config['device'],
            callbacks=callbacks,
            precision=16,
            enable_progress_bar=True
            )
        
    def train_and_eval(self):
        self.trainer.fit(self.model_interface, 
                         self.train_dataloader, self.valid_dataloader)

        test_results = self.trainer.test(self.model_interface, self.test_dataloader, ckpt_path="best")
        
        if self.is_weak:
            results_fname = "results/weak_model_results.csv"
            weight_fname = f"./weights/{self.weight_path}/weak_model.pt"
        else:
            results_fname = "results/strong_model_results.csv"
            weight_fname = f"./weights/{self.weight_path}/strong_model.pt"
            
        save_test_log(results_fname, self.project_name, test_results)
        torch.save(self.model_interface.model.state_dict(), 
                   weight_fname)