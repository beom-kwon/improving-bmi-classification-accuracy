import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = np.load("dataset.npz", allow_pickle=True)
    x = data['x']
    
    # Control Parameters
    idx = 6  # Sample Index
    w = 8    # Window Size Used in Spatiotemporal Feature Extraction
    
    right_ankle = x[idx][:, 48:51]  # Right Ankle (Joint Number: 17)
    left_ankle = x[idx][:, 51:54]   # Left Ankle (Joint Number: 18)
    diff = []
    for frm in range(0, x[idx].shape[0], 1):
        diff.append(np.linalg.norm(right_ankle[frm, :] - left_ankle[frm, :]))
        
    
    local_maxima = []
    for t in range(w, len(diff) - w):
        if (False not in (diff[(t - w):t] < diff[t])) and \
                (False not in (diff[(t + 1):(t + w)] < diff[t])):
            local_maxima.append(t)
        
        
    plt.figure()
    plt.plot(diff, color="black")
    plt.scatter(local_maxima, np.array(diff)[local_maxima], color="red")
    plt.xlabel("Frame")
    plt.ylabel("Difference between the right and left ankles")
    plt.grid()
    plt.show()
