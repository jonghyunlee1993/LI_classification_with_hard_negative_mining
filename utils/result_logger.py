import os
import pandas as pd


def save_test_log(fname, project_name, log):
    if not os.path.exists(fname):
        df = pd.DataFrame(log)
        df.loc[0, "project_name"] = project_name
    elif os.path.exists(fname):
        new_line = pd.DataFrame(log)
        new_line.loc[0, "project_name"] = project_name
        df = pd.read_csv(fname)
        df = pd.concat([df, new_line], axis=0)

    df.to_csv(fname, index=False)
