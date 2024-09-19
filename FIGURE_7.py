import matplotlib.pyplot as plt

if __name__ == "__main__":
    as1_acc = [0.3982, 0.3186, 0.3982, 0.6018, 0.5664, 0.4956]
    methods = ["U1+GB", "U2+kNN", "U3+RF", "O1+HGB", "O2+SVM", "O3+SVM"]
    
    plt.figure()
    bar = plt.bar(range(0, 6, 1), as1_acc, \
                  color=["lightblue", "dodgerblue", "blue", "thistle", "plum", "purple"])
    
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, \
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.xticks(range(0, 6, 1), methods)
    plt.ylabel("Classification Accuracy")
    plt.ylim([0, 1.0])
    plt.grid()
    plt.show()
