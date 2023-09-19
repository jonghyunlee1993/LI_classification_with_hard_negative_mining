import os
import cv2
import shutil
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F


class QueryFalsePositives:
    def __init__(self, project_name, model, valid_df, valid_transform, config):
        self.device = config["device"]
        self.project_name = project_name

        self.model = model.to(self.device)
        self.model.eval()

        self.valid_df = valid_df
        self.valid_transform = valid_transform

    def run(self):
        print("Start false positive query ... ")
        self.compute_probability()
        self.make_results_directory()
        self.query_false_positives()

    def compute_probability(self):
        pred_df = []
        for i in tqdm(range(len(self.valid_df))):
            p_id, fname, pred, label = self.predict(i)
            prob = F.softmax(pred, dim=1)
            pred = torch.argmax(prob, dim=1)

            result = [
                p_id,
                fname,
                pred.detach().cpu().numpy()[0],
                prob.detach().cpu().numpy()[0][1],
                label,
            ]
            pred_df.append(result)

        pred_df = pd.DataFrame(
            pred_df, columns=["p_id", "fname", "pred", "prob", "label"]
        )
        self.pred_df = pred_df

    def predict(self, index):
        p_id = self.valid_df.loc[index, "p_id"]
        fname = self.valid_df.loc[index, "fpath"]
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        label = self.valid_df.loc[index, "label"]
        input_tensor = self.valid_transform(image=image)["image"].unsqueeze(0)
        if self.device is not "cpu":
            input_tensor = input_tensor.to(self.device)
        pred = self.model(input_tensor)

        return p_id, fname, pred, label

    def make_results_directory(self):
        if not os.path.exists("./results"):
            os.makedirs("./results")

        path = os.path.join("./results", "false-positive_" + self.project_name)
        if not os.path.exists(path):
            os.makedirs(path)

    def query_false_positives(self, threshold=0.1):
        false_positive_df = self.pred_df[
            (self.pred_df.label == 0) & (self.pred_df.prob >= threshold)
        ].reset_index(drop=True)

        for _, line in false_positive_df.iterrows():
            source = line.fname
            dest = source.replace(
                "data/hard_labeled_dataset/",
                f"results/false-positive_{self.project_name}/",
            )
            shutil.copy(source, dest)
