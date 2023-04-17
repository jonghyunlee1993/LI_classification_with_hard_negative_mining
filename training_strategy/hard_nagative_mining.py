
import os
import cv2
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_loader import PatchDataset

class ListPatchDataset(PatchDataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
    
    def __getitem__(self, idx):
        fname = self.df[idx]
        x = cv2.imread(fname)
        x = self.transform(image=x)['image']
                                
        return fname.split("/")[-1], x

class HardNegativeMining:
    def __init__(self, project_name, model, 
                 valid_transform, config, 
                 number_of_query=20):
        self.project_name = project_name
        self.device = config['device']
        self.model = model.to(self.device)
        self.model.eval()
        
        self.valid_transform = valid_transform
        self.number_of_query = number_of_query
        
        ref_path = f"./results/false-positive_{self.project_name}/*.png"
        self.ref_flist = glob.glob(ref_path)
        
        weakly_path = "./data/weakly_labeled_dataset/*.jpg"
        self.weakly_flist = glob.glob(weakly_path)
    
    def run(self):
        print("\nStart hard negative mining ... ")
        self.get_false_positive_features()
        self.get_weakly_labeled_features()
        self.make_results_directory()
        self.query_hard_negatives()
    
    def get_false_positive_features(self):
        self.ref_feats = torch.tensor([])
        
        for ref in self.ref_flist:
            ref_image = cv2.imread(ref)
            ref_image = self.valid_transform(image=ref_image)['image']
            if self.device != "cpu":
                ref_image = ref_image.to(self.device)
                
            ref_hidden = self.model.forward_features(ref_image.unsqueeze(0))
            pooled_ref_feat = torch.mean(ref_hidden.view(ref_hidden.size(0), ref_hidden.size(1), -1), dim=2).detach().cpu()
        
            self.ref_feats = torch.cat([self.ref_feats, pooled_ref_feat], dim=0)

    def get_weakly_labeled_features(self):
        sample_batch_size = 32
        cand_dataset = ListPatchDataset(self.weakly_flist, self.valid_transform)
        cand_dataloader = DataLoader(cand_dataset, batch_size=sample_batch_size, 
                                     num_workers=32, drop_last=True, shuffle=True)
        
        self.weakly_feats = []
        with tqdm(total=len(self.ref_feats), desc="Ref Image") as outer:
            with tqdm(total=len(cand_dataloader), desc="Query Images") as inner:
                for i, ref_feat in enumerate(self.ref_feats):
                    result = {"fname": [], "dist": []}
                    inner.reset()
                    
                    for batch in cand_dataloader:
                        try:
                            if self.device != "cpu":
                                x = batch[1].to(self.device)
                            else:
                                x = batch[1]
                                
                            cand_feat = self.model.forward_features(x).reshape(sample_batch_size, -1).detach().cpu()
                            dist = nn.PairwiseDistance(p=2)(ref_feat, cand_feat).detach().cpu().numpy().tolist()
                            result["fname"].extend([b for b in batch[0]])
                            result["dist"].extend(dist)
                        except Exception as e:
                            result["fname"].extend([b for b in batch[0]])
                            result["dist"].extend([9999] * sample_batch_size)
                                
                        inner.update()

                    result = pd.DataFrame(result)
                    self.weakly_feats.extend(result.sort_values("dist", ascending=True).head(self.number_of_query).fname.values.tolist())
                    outer.update()
                    inner.refresh()
                    
        self.weakly_feats = np.unique(self.weakly_feats).tolist()
    
    def make_results_directory(self):
        if not os.path.exists("./results"):
            os.makedirs("./results")
        
        path = os.path.join("./results", "aug-data_" + self.project_name)
        if not os.path.exists(path):
            os.makedirs(path)
    
    def query_hard_negatives(self):
        for f in self.weakly_feats:
            shutil.copyfile(f"./data/weakly_labeled_dataset/{f.split('/')[-1]}",
                            os.path.join("./results/aug-data_" + self.project_name, f.split('/')[-1]))
            