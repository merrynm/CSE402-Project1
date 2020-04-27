#####################################################################
#                                                                   #
# Project 1 - CSE 402: Biometrics and Pattern Recognition           #
#                                                                   #
# Calculates the following based on the datasets from two matchers: #
# Fingerprint and Hand.                                             #
#                                                                   #
# - Genuine and impostor score counts                               #
# - Maximum and minimum scores for each matcher                     #
# - Mean and variance for each matchers' set of genuine and         #
#   impostor scores                                                 #
# - Histogram displaying genuine and impostor score distribution on #
#   the same graph                                                  #
# - FNMR and FMR given an input eta value for each matcher          #
# - ROC curve, as well as AUC and EER for each matcher              #
# - Using values plotted in the ROC curve, determines the FNMR      #
#   at a given FMR value (10%, 5%, 1%)                              #
#                                                                   #
#####################################################################

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as ROC_plt
import sklearn.metrics


def newline_remove(a_file):
    """Opens a data file, removes new lines, converts each point to a float and returns as a list of values"""
    a_list = open(a_file).readlines()
    return [float(item.rstrip('\n')) for item in a_list]


def find_max(a_list):
    """Finds the maximum value in a list of numbers"""
    a_list.sort()
    return a_list[-1]


def find_min(a_list):
    """Finds the minimum value in a list of numbers"""
    a_list.sort()
    return a_list[0]


def find_mean(a_list):
    """Finds the mean value in a list of numbers"""
    length = len(a_list)
    total = sum(a_list)

    return total/length


def d_prime(gen, imp):
    """Given a list of genuine and impostor scores, calculates and returns d' value of the matcher"""
    gen_mean = find_mean(gen)
    imp_mean = find_mean(imp)

    numerator = math.fabs(gen_mean - imp_mean)

    sum_vars = np.var(gen) + np.var(imp)
    denominator = math.sqrt(sum_vars/2)

    return numerator/denominator


def FMR(eta, a_list):
    """Calculates the False Match Rate given an eta value and list of impostor scores"""
    false_matches = 0
    total_scores = len(a_list)

    for item in a_list:
        if item > eta:
            false_matches += 1

    return false_matches/total_scores


def FNMR(eta, a_list):
    """Calculates the False Non-Match Rate given an eta value and a list of genuine scores"""
    false_non_matches = 0
    total_scores = len(a_list)

    for item in a_list:
        if item < eta:
            false_non_matches += 1

    return false_non_matches / total_scores


def plot_ROC(gen_data, imp_data, title):
    """Plots the ROC curve of a matcher"""
    FNMR_list = []
    FMR_list = []
    ROC_title = "ROC Curve: " + title

    for eta_value in range(1001):
        FNMR_list.append(FNMR(eta_value, gen_data))
        FMR_list.append(FMR(eta_value, imp_data))

    # giving a title to my graph
    ROC_plt.title(ROC_title)

    # naming the x axis
    ROC_plt.xlabel("FMR")
    # naming the y axis
    ROC_plt.ylabel("FNMR")

    ROC_plt.plot(FMR_list, FNMR_list)

    ROC_plt.show()

    print()

    AUC = sklearn.metrics.auc(FMR_list, FNMR_list)
    print("Area under the curve for the %s: %.3f" % (title, AUC))

    EER_val = EER(FNMR_list, FMR_list)
    print("Equal Error Rate (EER) for the %s: %.1f" % (title, EER_val))


def EER(FNMR_vals, FMR_vals):
    """Calculates the Equal Error Rate of a matcher's ROC curve"""
    EER_val = None

    for i in range(1001):
        FMR = round(FMR_vals[i], 1)
        FNMR = round(FNMR_vals[i], 1)

        if FMR == FNMR:
            EER_val = FMR

    return EER_val


def FNMR_at_FMR(FMR_val, gen_data, imp_data):
    """Calculates a matcher's FNMR at a given FMR value"""
    FNMR_list = []
    FMR_list = []

    FNMR_val = 0

    for eta_value in range(1001):
        FNMR_list.append(FNMR(eta_value, gen_data))
        FMR_list.append(FMR(eta_value, imp_data))

    for loc, score in enumerate(FMR_list):
        if round(score, 2) == FMR_val:
            FNMR_val = float(FNMR_list[loc])

    FMR_val *= 100
    FNMR_val *= 100

    print("The FNMR value when FMR is %.2f%% is: %.2f%%." % (FMR_val, FNMR_val))


# All score sets for each matcher
finger_gen_scores = newline_remove("proj01_q1_match_scores/finger_genuine.score")
finger_imp_scores = newline_remove("proj01_q1_match_scores/finger_impostor.score")
hand_gen_scores = newline_remove("proj01_q1_match_scores/hand_genuine.score")
hand_imp_scores = newline_remove("proj01_q1_match_scores/hand_impostor.score")


# Code relating to problems 1 through 3
finger_scores = finger_imp_scores + finger_gen_scores
hand_scores = hand_imp_scores + hand_gen_scores

print("1. NUMBER OF SCORES")
print()
print("Fingerprint Matcher:")
print("Number of Genuine Scores: %d" % len(finger_gen_scores))
print("Number of Impostor Scores: %d" % len(finger_imp_scores))
print()
print("Hand Matcher:")
print("Number of Genuine Scores: %d" % len(hand_gen_scores))
print("Number of Impostor Scores: %d" % len(hand_imp_scores))
print()

print()
print("2. MAXIMUM AND MINIMUM SCORES")
print()
print("Fingerprint Matcher:")
print(" - Maximum: %.3f" % find_max(finger_scores))
print(" - Minimum: %.3f" % find_min(finger_scores))
print()
print("Hand Matcher:")
print(" - Maximum: %.3f" % find_max(hand_scores))
print(" - Minimum: %.3f" % find_min(hand_scores))
print()

print()
print('3. MEAN AND VARIANCE')
print()
print("Fingerprint Matcher")
print("Genuine Scores:")
print(" - Mean: %.3f" % find_mean(finger_gen_scores))
print(" - Variance: %.3f" % np.var(finger_gen_scores))
print()
print("Impostor Scores:")
print(" - Mean: %.3f" % find_mean(finger_imp_scores))
print(" - Variance: %.3f" % np.var(finger_imp_scores))
print()
print("Hand Matcher")
print("Genuine Scores:")
print(" - Mean: %.3f" % find_mean(hand_gen_scores))
print(" - Variance: %.3f" % np.var(hand_gen_scores))
print()
print("Impostor Scores:")
print(" - Mean: %.3f" % find_mean(hand_imp_scores))
print(" - Variance: %.3f" % np.var(hand_imp_scores))
print()

print()
print("4. D-PRIME VALUES")
print()
print("Fingerprint Matcher: %.3f" % d_prime(finger_gen_scores, finger_imp_scores))
print("Hand Matcher: %.3f" % d_prime(hand_gen_scores, hand_imp_scores))

# Code relating to problem 5: Histogram plotting
print()
print("5. SEE HISTOGRAMS")
# Set number of bins; constant among each matchers' scores
num_bins = [i for i in range(0, 1000, 10)]

# Plotting fingerprint matcher's genuine and impostor scores on same graph
plt.figure(0)
plt.title("Genuine and Impostor Scores for Fingerprint Matcher")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.hist(finger_gen_scores, bins=num_bins, alpha=0.5, label='Genuine scores', color='r')
plt.hist(finger_imp_scores, bins=num_bins, alpha=0.5, label='Impostor scores', color='b')
plt.legend(loc='upper right')
plt.show()

# Plotting fingerprint matcher's genuine and impostor scores on same graph
plt.figure(1)
plt.title("Genuine and Impostor Scores for Hand Matcher")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.hist(hand_gen_scores, bins=num_bins, alpha=0.5, label='Genuine scores', color="y")
plt.hist(hand_imp_scores, bins=num_bins, alpha=0.5, label='Impostor scores', color="b")
plt.legend(loc='upper right')
plt.show()

# Code relating to problem 6
print()
print("6. FMR and FNMR")
print()

# Inputs for eta
try:
    eta_fingerprint = int(input("Input a threshold value for the fingerprint matcher: \n"))
    eta_hand = int(input("Input a threshold value for the hand matcher: \n"))
    print()

# input is not a number
except ValueError:
    print("Please try another input (numeric value 0 - 1000)")
    eta_fingerprint = int(input("Input a threshold value for the fingerprint matcher: \n"))
    eta_hand = int(input("Input a threshold value for the hand matcher: \n"))
    print()

# Format and print data
print("FINGERPRINT MATCHER")
print("FMR: %.3f" % FMR(eta_fingerprint, finger_imp_scores))
print("FNMR: %.3f" % FNMR(eta_fingerprint, finger_gen_scores))
print()
print(".............................................................")
print()
print("HAND MATCHER")
print("FMR: %.3f" % FMR(eta_hand, hand_gen_scores))
print("FNMR: %.3f" % FNMR(eta_hand, hand_imp_scores))

# Code relating to problems 7 and 8
print()
print("7. ROC CURVES, AUC & EER")

# ROC curve for finger matcher
plot_ROC(finger_gen_scores, finger_imp_scores, "Fingerprint Matcher")

print()

# for hand matcher, swap impostor and genuine scores, for dissimilarity
plot_ROC(hand_imp_scores, hand_gen_scores, "Hand Matcher")

print()

# FMR values we are testing at
percent_vals = [.10, .05, .01]

print("8. FNMR VALUES AT SPECIFIC FMR VALUES")
print("Fingerprint Matcher")
for val in percent_vals:
    FNMR_at_FMR(val, finger_gen_scores, finger_imp_scores)
print()
print("Hand Matcher")
for val in percent_vals:
    FNMR_at_FMR(val, hand_imp_scores, hand_gen_scores)
