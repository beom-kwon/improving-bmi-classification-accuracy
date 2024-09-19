import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    dataset_path = "./kinect gait raw dataset/"

    df = pd.read_csv("./person-data.csv")
    df = df.dropna(axis=0, how="all")
    df = df[["Individual", "Body Mass Index"]]
    df = df[df["Body Mass Index"] != 0.0]

    num_joints = 20  # Kinect Version 1

    x, y, pid, rep = [], [], [], []
    for person, bmi in zip(df["Individual"].to_numpy(), df["Body Mass Index"].to_numpy()):    
        walk_list = os.listdir(dataset_path + person)
        for walk in walk_list:
            file_path = dataset_path + person + '/' + walk
            txt = pd.read_csv(file_path, sep=';', header=None).iloc[:, 1:]
            txt = np.array(txt)
            txt = txt.reshape(int(txt.shape[0] / num_joints), num_joints * 3)

            x.append(txt)
            y.append(bmi)
            pid.append(person)  # Person ID
            rep.append(walk[0]) # Repetition

    np.savez("dataset.npz", x=np.array(x, dtype="object"), y=np.array(y), pid=np.array(pid), rep=np.array(rep))
