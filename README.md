# LI classification with hard negative mining

This is the author implementation of `Robust model training strategy via hard negative mining in the weakly labeled dataset for lymphatic invasion in gastric cancer`.

## Abstract
Gastric cancer poses a significant public health concern, emphasizing the need for accurate evaluation of lymphatic invasion (LI) for determining prognosis and treatment options. However, this task is time-consuming, labor-intensive, and prone to intra- and inter-observer variability. Furthermore, the scarcity of annotated data presents a challenge, particularly in the field of digital pathology. Therefore, there is a demand for an accurate and objective method to detect LI using a small dataset, benefiting pathologists. In this study, we trained convolutional neural networks (CNNs) to classify LI using a four-step training process: (1) weak model training; (2) identification of false positives; (3) hard negative mining in a weakly labeled dataset; and (4) strong model training. To overcome the lack of annotated datasets, we applied a hard negative mining approach in a weakly labeled dataset, which contained only final diagnostic information, resembling the typical data found in hospital databases, and improved classification performance. Ablation studies were performed to simulate the lack of datasets and severely unbalanced datasets, further confirming the effectiveness of our proposed approach. Notably, our results demonstrated that despite the small number of annotated datasets, efficient training was achievable, with the potential to extend to other image classification approaches used in the field of medicine.

![image](https://github.com/jonghyunlee1993/LI_classification_with_hard_negative_mining/assets/37280722/71e0a3f8-dc8c-43c8-aa70-c9a81d9114d2)

## How to run?
To utilize this repository, simply add the desired configuration file to the "config" folder. We recommend using the Template config file as a starting point for modifications. Once you have made the necessary changes, execute the following commands to run the modified configuration file for the learning process.
1. Install requiring packages `pip install -r requirements.txt`
2. Run a script `python run.py -i my_config.yaml`

## Dataset hierarchy
```
project/
├─ config/
├─ data/
│  ├─ hard_labeled_dataset/
│  │  ├─ pos_example_1.png
│  │  ├─ neg_example_1.png
│  ├─ weakly_labeled_dataset/
│  │  ├─ example_1.png
├─ utils/
├─ training_strategy/
├─ model/
├─ run.py
```

## Data availability
You can freely access dataset with following link: https://zenodo.org/records/10020633
