import warnings

warnings.filterwarnings("ignore")
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


# All enroll are enrolled
def evaluate_offline(
    featurePath="./pickle/record_feature_second.pkl", enrollPath="./pickle/enrolled_user.pkl", scorePath="./pickle/score.pkl", scoreMethodPath="./pickle/score_method.pkl", showPlot=True
):
    if not os.path.exists(featurePath) or not os.path.exists(enrollPath):
        raise FileNotFoundError("Second session feature or enrolled feature not found")
    else:
        with open(featurePath, "rb") as f:
            record_feature = pickle.load(f)
        with open(enrollPath, "rb") as f:
            enroll_feature = pickle.load(f)

        if len(record_feature) == 0 or len(enroll_feature) == 0:
            raise ValueError("Empty second session feature or enrolled feature")

    evaluate_score = {}
    score_method = {}
    num_of_method = len(enroll_feature)

    for method, item in enroll_feature.items():
        # Iterate every user
        y_score = np.array([])
        y_true = np.array([])
        scores_match = scores_imposter = np.array([])
        for enrollUser, enrollFeatures in item.items():
            genuine_list = np.array([])
            imposter_list = np.array([])

            # Iterate every featrue to be verified
            for verifyUser, verifyFeatures in record_feature.items():
                # verifyFeatures (150,512)
                for verifyFeature in verifyFeatures:
                    # print(enrollUser,verifyUser)
                    max_score = -1
                    # Iterate every feature in every method's every template list
                    for enrollFeature in enrollFeatures:
                        score = sum(verifyFeature * enrollFeature)
                        if score > max_score:
                            max_score = score
                    # 001_left_index  in 001_left_index_2
                    if enrollUser in verifyUser:
                        genuine_list = np.append(genuine_list, max_score)
                    else:
                        imposter_list = np.append(imposter_list, max_score)
            try:
                evaluate_score[method].update({enrollUser: {"genuine": genuine_list, "imposter": imposter_list}})
            except KeyError:
                evaluate_score.update({method: {enrollUser: {"genuine": genuine_list, "imposter": imposter_list}}})
            y_score = np.append(y_score, np.append(genuine_list, imposter_list))
            y_true = np.append(y_true, (np.append(np.ones(genuine_list.shape), np.zeros(imposter_list.shape))))
            scores_match = np.append(scores_match, genuine_list)
            scores_imposter = np.append(scores_imposter, imposter_list)
            score_method.update({method: {"y_score": y_score, "y_true": y_true, "scores_match": scores_match, "scores_imposter": scores_imposter}})
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        metrics, metrics_thred = compute_roc_metrics(fpr, tpr, thresholds)

    # Draw Roc
    plt.figure(figsize=(10, 6), dpi=120)
    for method, item in score_method.items():
        fpr, tpr, thresholds = roc_curve(item["y_true"], item["y_score"])
        metrics, metrics_thred = compute_roc_metrics(fpr, tpr, thresholds)
        draw_roc(method, metrics[0], fpr, tpr)
    plt.savefig("./evaluation_result/Evaluate_ROC.svg")
    if showPlot:
        plt.show()

    # Draw Hist
    index = 1
    plt.figure(figsize=(20, num_of_method // 2 * 5), dpi=120)
    for method, item in score_method.items():
        print("*" * 80)
        print(method)
        fpr, tpr, thresholds = roc_curve(item["y_true"], item["y_score"])
        metrics, metrics_thred = compute_roc_metrics(fpr, tpr, thresholds, printMetrics=True)
        scores_match = item["scores_match"]
        scores_imposter = item["scores_imposter"]
        print("score_match min", np.amin(scores_match), "score_match max", np.amax(scores_match))
        print("scores_imposter min", np.amin(scores_imposter), "scores_imposter max", np.amax(scores_imposter))
        plot_histogram(scores_match, scores_imposter, metrics, metrics_thred, method, index, num_of_method=num_of_method)
        index = index + 1
    plt.savefig("./evaluation_result/Evalute_Histogram.svg")
    if showPlot:
        plt.show()

    with open(scorePath, "wb") as f:
        pickle.dump(evaluate_score, f)
    with open(scoreMethodPath, "wb") as f:
        pickle.dump(score_method, f)


def plot_histogram(scores_match, scores_imposter, metrics=None, metrics_threds=None, methodName=None, index=1, title=None, y_lim=None, x_lim=90, save_fig=True, num_of_method=1):
    # show histogram
    f = 180 / np.pi
    bins = np.linspace(0, np.pi, 300) * f
    angles_match = np.arccos(np.clip(scores_match, -1, 1)) * f
    angles_imposter = np.arccos(np.clip(scores_imposter, -1, 1)) * f
    plt.subplot(num_of_method // 2, 2, index)
    plt.hist(angles_match, bins, alpha=0.5, fill=False, density=True, histtype="step", stacked=False, label="Genuine, mean={:.2f}".format(np.mean(angles_match)))
    plt.hist(angles_imposter, bins, alpha=0.5, fill=False, density=True, histtype="step", stacked=False, label="Imposter, mean={:.2f}".format(np.mean(angles_imposter)))
    if metrics_threds is not None:
        plt.axvline(np.arccos(metrics_threds[0]) * f, color="red", linestyle="-", linewidth=1, label="eer threshold=%.2f" % (np.arccos(metrics_threds[0]) * f))
    plt.axvline(np.mean(angles_match), color="k", linestyle="dashed", linewidth=1)
    plt.axvline(np.mean(angles_imposter), color="k", linestyle="dashed", linewidth=1)

    plt.ylim([0, y_lim])
    plt.xlim([0, x_lim])
    plt.xlabel("Angle(degree)")
    plt.ylabel("Probability")
    plt.legend(loc="upper left")
    # "Normalized " +
    title = methodName + " EER:{:.2f}%".format(metrics[0] * 100)
    plt.title(title)
    # plt.show()


def draw_roc(label, eer, fpr, tpr):
    lw = 0.5
    plt.semilogx(fpr, tpr, label=label + " EER:{:.2f}%".format(eer * 100))
    # plt.xlim([0, 0.6])
    # plt.ylim([0.4, 1.0])
    plt.xlabel("False Acceptance Rate")
    plt.ylabel("True Acceptance Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")


def compute_roc_metrics(fpr, tpr, thresholds, printMetrics=False):
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    fpr100_idx = sum(fpr <= 0.01) - 1
    fpr1000_idx = sum(fpr <= 0.001) - 1
    fpr0_idx = sum(fpr <= 0.0) - 1

    # compute EER, FRR@FAR=0.01, FRR@FAR=0.001, FRR@FAR=0
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    fpr100 = fnr[fpr100_idx]
    fpr1000 = fnr[fpr1000_idx]
    fpr0 = fnr[fpr0_idx]

    metrics = (eer, fpr100, fpr1000, fpr0)
    metrics_thred = (thresholds[eer_idx], thresholds[fpr100_idx], thresholds[fpr1000_idx], thresholds[fpr0_idx])
    # print(dt(), 'Performance evaluation...')
    if printMetrics:
        print("EER:%.2f%%, FRR@FAR=0.01: %.2f%%, FRR@FAR=0.001: %.2f%%, FRR@FAR=0: %.2f%%, Aver: %.2f%%" % (eer * 100, fpr100 * 100, fpr1000 * 100, fpr0 * 100, np.mean(metrics) * 100))
    return metrics, metrics_thred


if __name__ == "__main__":

    evaluate_offline()