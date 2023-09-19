import glob 
import pandas as pd 

from collections import Counter
from sklearn.model_selection import train_test_split

class DataManager:
    def __init__(self, 
                 hard_path="./data/hard_labeled_dataset/*.png", 
                 aug_path=None, valid_ratio=0.2, test_ratio=0.2, 
                 is_sampling=False, sampling_rate=0.5,
                 sampling_random_state=42, sample_only_pos=False):
        
        self.hard_path = hard_path
        self.aug_path = aug_path
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        
        self.is_sampling = is_sampling
        self.sampling_rate = sampling_rate
        self.sampling_random_state = sampling_random_state
        self.sample_only_pos = sample_only_pos

    def run(self):
        self.generate_dataset_df()
        self.split_train_valid_test()
        if self.is_sampling:
            self.sample_train_dataset()
        self.log_dataset()
        
        return self.train_df, self.valid_df, self.test_df
    
    def generate_dataset_df(self):
        flist = glob.glob(self.hard_path)
        df = []
        
        for f in flist:
            f_split = f.split("/")[-1].split("_")
            label = int([1 if f_split[0] == "pos" else 0][0])
            p_id = f_split[1]
            s_id = f_split[2]

            df.append([p_id, s_id, label, f])
            
        df = pd.DataFrame(df, columns=["p_id", "s_id", "label", "fpath"])
        
        if self.aug_path is not None:
            aug_flist = glob.glob(self.aug_path)
            aug_df = []
            
            for aug_f in aug_flist:
                p_id = aug_f.split("/")[-1].split("_")[0]
                s_id = aug_f.split("/")[-1].split("_")[2]
                label = 0
                aug_df.append([p_id, s_id, label, aug_f])
                
            aug_df = pd.DataFrame(aug_df, columns=["p_id", "s_id", "label", "fpath"])
            df = pd.concat([df, aug_df], ignore_index=True)

        self.df = df

    def split_train_valid_test(self):
        unique_ids = self.df.s_id.unique()
        train_id, test_id = train_test_split(unique_ids, test_size=self.test_ratio, random_state=self.sampling_random_state)
        train_id, valid_id = train_test_split(train_id, test_size=self.valid_ratio, random_state=self.sampling_random_state)
        
        self.train_df = self.df[self.df.s_id.isin(train_id)].reset_index(drop=True)
        self.valid_df = self.df[self.df.s_id.isin(valid_id)].reset_index(drop=True)
        self.test_df = self.df[self.df.s_id.isin(test_id)].reset_index(drop=True)
    
    def sample_train_dataset(self):
        if self.sample_only_pos:
            pos_df = self.train_df[(self.train_df.label == 1)].reset_index(drop=True)
            neg_df = self.train_df[(self.train_df.label == 0)].reset_index(drop=True)
            
            pos_df_sampled = pos_df.sample(int(len(pos_df) * self.sampling_rate), random_state=self.sampling_random_state).reset_index(drop=True)
            self.train_df = pd.concat([pos_df_sampled, neg_df], axis=0).reset_index(drop=True)
        else:
            _, self.train_df = train_test_split(self.train_df, test_size=self.sampling_rate, random_state=self.sampling_random_state).reset_index(drop=True)
            self.train_df = self.train_df.reset_index(drop=True)
            
    def log_dataset(self):
        print("Dataset configurations ... ")
        print(f"# of patches\t train: {len(self.train_df)} valid: {len(self.valid_df)} test: {len(self.test_df)}")
        print(f"# of labels\t train: {Counter(self.train_df.label.values)} valid: {Counter(self.valid_df.label.values)} test: {Counter(self.test_df.label.values)}") 
        print()
        
if __name__ == "__main__":
    weak_data_manager = DataManager()
    weak_data_manager.run()