import numpy as np
import matplotlib.pyplot as plt


def LabelGeneration(y):
    # BMI < 25.0: Normal Weight(label=0)
    # BMI >= 25.0: Overweight(label=1)
    # BMI >= 30.0: Obesity(label=2)
    y[y < 25.0] = 0
    y[(25.0 <= y) & (y < 30.0)] = 1
    y[30.0 <= y] = 2
    return y


if __name__ == "__main__":
    data = np.load("dataset.npz", allow_pickle=True)
    y = LabelGeneration(data['y'])
    
    _, counts = np.unique(y, return_counts=True)
    
    plt.figure()
    plt.bar(range(3), counts, color=["green", "orange", "red"])
    plt.xticks(range(3), ["Normal Weight", "Overweight", "Obesity"])
    plt.ylabel("Count")
    plt.grid()
    plt.show()
