import numpy as np

def SkeletonInfo():
    num_joints = 20
    limb_info = [[2, 3],    # 01. shoulder right
                 [3, 5],    # 02. arm right
                 [5, 7],    # 03. forearm right
                 [7, 9],    # 04. hand right
                 [12, 13],  # 05. hip right
                 [13, 15],  # 06. thigh right
                 [15, 17],  # 07. leg right
                 [17, 19],  # 08. foot right
                 [1, 2],    # 09. neck
                 [2, 11],   # 10. upper spine
                 [11, 12],  # 11. lower spine
                 [2, 4],    # 12. shoulder left
                 [4, 6],    # 13. arm left
                 [6, 8],    # 14. forearm left
                 [8, 10],   # 15. hand left
                 [12, 14],  # 16. hip left
                 [14, 16],  # 17. thigh left
                 [16, 18],  # 18. leg left
                 [18, 20]]  # 19. foot left

    return num_joints, np.array(limb_info)


def LabelGeneration(y):
    # BMI < 25.0: Normal Weight(label=0)
    # BMI >= 25.0: Overweight(label=1)
    # BMI >= 30.0: Obesity(label=2)
    y[y < 25.0] = 0
    y[(25.0 <= y) & (y < 30.0)] = 1
    y[30.0 <= y] = 2
    return y


def AnthropometricFeatureExtraction(motion, num_joints, limb_info):
    feature = []
    for frm in range(0, motion.shape[0], 1):
        pose = motion[frm, :].reshape(num_joints, 3)
        
        vec = []
        # Calculate the Length of Each Body Part (19 Dim.)
        for i in range(0, limb_info.shape[0], 1):
            vec.append(np.linalg.norm(pose[limb_info[i, 0] - 1, :] - pose[limb_info[i, 1] - 1, :]))

        # Calculate the Height of Subject (1 Dim.)
        # height = neck + upper spine + lower spine
        #        + avg(right hip, left hip)
        #        + avg(right thigh, left thigh)
        #        + avg(right leg, left leg)
        neck = vec[9 - 1]                            # neck
        upper_spine = vec[10 - 1]                    # upper_spine
        lower_spine = vec[11 - 1]                    # lower_spine
        avg_hips = (vec[5 - 1] + vec[16 - 1]) / 2    # average of hips
        avg_thighs = (vec[6 - 1] + vec[17 - 1]) / 2  # average of thighs
        avg_legs = (vec[7 - 1] + vec[18 - 1]) / 2    # average of legs
        subj_height = neck + upper_spine + lower_spine + avg_hips + avg_thighs + avg_legs
        vec.append(subj_height)

        feature.append(vec)
        
    feature = np.array(feature)
    
    # Remove Outliers and Recalculate Means
    mean_f = np.mean(feature, axis=0)
    std_f = np.std(feature, axis=0)
    
    recalc_means = []
    for att in range(0, feature.shape[1], 1):
        vec = []
        for frm in range(0, feature.shape[0], 1):
            if feature[frm, att] < mean_f[att] - 2 * std_f[att]:
                pass
            elif feature[frm, att] > mean_f[att] + 2 * std_f[att]:
                pass
            else:
                vec.append(feature[frm, att])
                
        recalc_means.append(np.mean(vec))
    
    return recalc_means


def SpatiotemporalFeatureExtraction(right_ankle, left_ankle, w):
    diff = []
    for frm in range(0, right_ankle.shape[0], 1):
        diff.append(np.linalg.norm(right_ankle[frm, :] - left_ankle[frm, :]))
        
    local_maxima = []
    for t in range(w, len(diff) - w):
        if (False not in (diff[(t - w):t] < diff[t])) and \
                (False not in (diff[(t + 1):(t + w)] < diff[t])):
            local_maxima.append(t)

    SetOfStepLengths = np.array(diff)[local_maxima]
    stepLength = np.mean(SetOfStepLengths)
    strideLength = 2 * stepLength  # Feature 1
    
    SetOfStrideLengths = []
    for j in range(1, SetOfStepLengths.size, 1):
        SetOfStrideLengths.append(SetOfStepLengths[j - 1] + SetOfStepLengths[j])
    
    avgStrideLength = np.mean(SetOfStrideLengths)  # Feature 2
    
    cyclePeriod = []
    for j in range(1, len(local_maxima), 1):
        cyclePeriod.append(local_maxima[j] - local_maxima[j - 1])
    
    avgCyclePeriod = np.mean(cyclePeriod)
    cycleTime = avgCyclePeriod / 30.0       # Feature 3
    velocity = avgStrideLength / cycleTime  # Feature 4

    return [strideLength, avgStrideLength, cycleTime, velocity]


# def KinematicFeatureExtraction(...):
#     return [...]


if __name__ == "__main__":
    num_joints, limb_info = SkeletonInfo()

    data = np.load("dataset.npz", allow_pickle=True)

    x = data['x']
    y = LabelGeneration(data['y'])
    pid = data["pid"]

    a_f, s_f, ans_f = [], [], []
    for j in range(0, x.shape[0], 1):
        # 1) Anthropometric Feature Extraction (Total of 20 Dim.)
        afe = AnthropometricFeatureExtraction(x[j], num_joints, limb_info)

        # 2) Spatiotemporal Feature Extraction (Total of 4 Dim.)
        rightAnkle = x[j][:, 48:51]  # Right Ankle (Joint Number: 17)
        leftAnkle = x[j][:, 51:54]   # Left Ankle (Joint Number: 18)
        sfe = SpatiotemporalFeatureExtraction(rightAnkle, leftAnkle, 8)

        # 3) Kinematic Feature Extraction (Total of 56 Dim.)
        # k = KinematicFeatureExtraction()

        a_f.append(afe)
        s_f.append(sfe)
        ans_f.append(afe + sfe)


    np.savez("A_Features.npz", x=np.array(a_f), y=y, pid=pid)
    np.savez("S_Features.npz", x=np.array(s_f), y=y, pid=pid)
    np.savez("AnS_Features.npz", x=np.array(ans_f), y=y, pid=pid)
