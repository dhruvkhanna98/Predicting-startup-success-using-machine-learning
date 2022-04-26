from sklearn import metrics
import matplotlib.pyplot as plt

# From lecture notes
def plot_roc(mod, X_test, y_test, label):
    # predicted_probs is an N x 2 array, where N is number of observations
    # and 2 is number of classes
    predicted_probs = mod.predict_proba(X_test)

    # keep the second column, for label=1
    predicted_prob1 = predicted_probs[:, 1]

    fpr, tpr, _ = metrics.roc_curve(y_test, predicted_prob1)

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot([0, 1], [0, 1], "k--")
    ax.plot(fpr, tpr, label=f"{label} (AUC score={round(metrics.roc_auc_score(y_test, mod.predict(X_test)), 2)})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.set_title(f"ROC Curve for {label}")
    
    return fig, ax
