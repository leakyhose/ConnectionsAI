"""
Generates the outcomes used to evaluate algorithms, as well as the data used for making the graphs.

"""

from game_master import play
import numpy as np
import matplotlib.pyplot as plt
import statistics

DATA_MODEL = "fasttext"
SIZE = len(np.load(DATA_MODEL + "/data.npy", allow_pickle=True))

cheat = []


def create_outcomes(SIZE, DATA_MODEL, weights):
    """
    Simulate the game for a given number of iterations (SIZE) and store the outcomes.
    Saves the results to a .npy file.
    """
    data = []
    for i in range(SIZE):
        data.append(play(i, DATA_MODEL, weights))
        print(str(i) + ":" + str(data[i]))

    return data


def load_outcomes(DATA_MODEL):
    """
    Load the outcomes data from a previously saved .npy file.
    """
    return np.load(DATA_MODEL + "_outcomes.npy")


def graph(data):
    """
    Generate and display a histogram of the outcomes data.
    The histogram shows the frequency of different outcome values.
    """
    min_val = min(data)
    max_val = max(data)
    bins = np.arange(min_val, max_val + 2)
    x, bins, patches = plt.hist(data, bins=bins, edgecolor="black", align="left")

    for i in range(len(patches)):
        bin_center = (bins[i] + bins[i + 1]) / 2
        plt.text(
            bin_center,
            -0.5,
            str(int(bins[i])),
            ha="center",
            va="top",
            color="black",
            fontsize=8,
        )

    plt.title("Connections AI Outputs Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.ylim(bottom=-1)
    plt.show()


def analyze(numbers):
    count_under_4 = sum(1 for num in numbers if num < 4)
    mean = statistics.mean(numbers)
    return count_under_4 + (1 - (mean * 0.01))


def analyze_full(numbers):
    """
    Analyze the data to calculate:
    - The mean value of the outcomes.
    - The mode (most frequent) value of the outcomes.
    - The number and percentage of outcomes that are less than 4.
    """
    count_under_4 = sum(1 for num in numbers if num < 4)
    mean = statistics.mean(numbers)
    mode = statistics.mode(numbers)
    percentage_under_4 = count_under_4 / len(numbers)

    print(f"Mean: {mean}")
    print(f"Mode: {mode}")
    print(f"Amount of numbers under or equal to 4: {count_under_4}")
    print(f"Percentage of numbers under: {percentage_under_4:.2f}%")


# data = create_outcomes(SIZE, DATA_MODEL, (0.7441864013671875, 0.06005859375))
# analyze_full(data)
# graph(data)
