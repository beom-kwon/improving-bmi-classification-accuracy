import matplotlib.pyplot as plt

if __name__ == "__main__":
    as1_acc = [0.6991, 0.5310, 0.6903, 0.9292, 0.9027, 0.9027]
    methods = ["U1+kNN", "U2+SVM", "U3+SVM", "O1+kNN", "O2+RF", "O3+kNN"]
    
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
