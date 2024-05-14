# Multiple functions to create confusion Matrices and Plots for evaluation

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from torch.utils.data import DataLoader
from model import CustomModel
from datasets import load_dataset
from builder import MCSBuilder
import os
import copy
import tqdm


def generate_cfm_separate_classes(labels, predictions, num_classes=2):
    """
    Generate a confusion matrix for multi-class classification problems.

    :param labels: numpy array of true labels
    :param predictions: numpy array of predicted labels
    :param num_classes: number of classes
    :return: tensor representing the confusion matrix
    """

    # Initialize confusion matrix
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int)

    # Iterate over each sample
    for i in range(labels.shape[0]):
        cm[int(labels[i]), int(predictions[i])] += 1

    return cm


def plot_confusion_matrix(
    cm, class_names, title="Confusion Matrix", cmap="viridis", save_path=None
):
    """
    Plots the confusion matrix using seaborn's heatmap function.

    :param cm: the confusion matrix to be plotted
    :param class_names: array of class names
    :param title: title of the graph
    :param cmap: colormap (default is viridis which is a perceptually uniform colormap)
    :param save_path: optional name of where to save the confusion matrix
    """
    plt.figure(figsize=(12, 7))
    with torch.no_grad():  # Context-manager that disabled gradient calculation
        norm_cm = cm.float() / cm.sum(dim=1, keepdim=True)
        norm_cm[torch.isnan(norm_cm)] = 0  # Convert NaNs to 0

    # Convert to NumPy for plotting
    norm_cm = norm_cm.numpy()
    cm = cm.numpy()

    # Create annotations with percentage and count
    annot = np.array(
        [
            [
                f"{cm[i, j]}\n({val*100:.2f}%)" if cm[i, j] != 0 else "0"
                for j, val in enumerate(row)
            ]
            for i, row in enumerate(norm_cm)
        ]
    )
    sns.heatmap(
        norm_cm,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Save the plot
    if save_path:
        plt.savefig(
            save_path, format="png", dpi=300
        )  # You can adjust the format and DPI

    plt.close()


def binary_classification_metrics(labels, predictions):
    """
    Compute binary classification metrics (TP, TN, FP, FN) using PyTorch tensors.

    :param labels: PyTorch tensor of true labels (multi-class)
    :param predictions: PyTorch tensor of predicted labels (multi-class)
    :return: dictionary with TP, TN, FP, FN
    """
    # Convert multi-class labels to binary labels (0 is negative, all else is positive)
    binary_labels = (labels != 0).int()
    binary_predictions = (predictions != 0).int()

    TP = torch.sum((binary_labels == 1) & (binary_predictions == 1))
    TN = torch.sum((binary_labels == 0) & (binary_predictions == 0))
    FP = torch.sum((binary_labels == 0) & (binary_predictions == 1))
    FN = torch.sum((binary_labels == 1) & (binary_predictions == 0))

    return {"TP": TP.item(), "TN": TN.item(), "FP": FP.item(), "FN": FN.item()}


def plot_classification_metrics(metrics, save_path=None):
    """
    Plot a 2x2 heatmap of TP, TN, FP, FN using Seaborn with counts and percentages.

    :param metrics: dictionary containing TP, TN, FP, FN
    """
    # Create a 2x2 matrix for the heatmap using PyTorch
    total = sum(metrics.values())
    data = torch.tensor(
        [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]],
        dtype=torch.float32,
    )

    # Calculate percentages using PyTorch
    percentages = data / total * 100

    # Create annotations array with both counts and percentages
    labels = np.array(
        [
            [
                f"True Negative: {metrics['TN']}\n({percentages[0, 0]:.2f}%)",
                f"False Positive: {metrics['FP']}\n({percentages[0, 1]:.2f}%)",
            ],
            [
                f"False Negative: {metrics['FN']}\n({percentages[1, 0]:.2f}%)",
                f"True Positive: {metrics['TP']}\n({percentages[1, 1]:.2f}%)",
            ],
        ]
    )

    # Convert PyTorch tensor to NumPy array for plotting
    data_np = data.numpy()

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data_np,
        annot=labels,
        fmt="",
        cmap="coolwarm",
        cbar=True,
        linewidths=2,
        linecolor="black",
        xticklabels=["Predicted Negative", "Predicted Positive"],
        yticklabels=["Actual Negative", "Actual Positive"],
    )

    plt.title("Classification Metrics: TP, TN, FP, FN")
    plt.xlabel("Predicted Condition")
    plt.ylabel("Actual Condition")

    if save_path:
        plt.savefig(save_path, format="png", dpi=300)

    plt.close()


def manual_precision_recall_tensor(
    y_true, y_scores, num_thresholds=100, save_path=None
):
    """
    Compute precision and recall for a series of thresholds using PyTorch tensors.

    Parameters:
    - y_true: PyTorch tensor of true binary labels
    - y_scores: PyTorch tensor of scores or probabilities for the positive class
    - num_thresholds: number of thresholds to evaluate

    Returns:
    - precision: list of precision values
    - recall: list of recall values
    - thresholds: numpy array of used thresholds
    """
    y_scores_np = y_scores.numpy()

    precision = []
    recall = []

    y_true_np = y_true.numpy()

    y_scores_np = 1 / (1 + np.exp(-1 * y_scores_np))

    sorted_indices = np.argsort(y_scores_np)
    y_scores_np = y_scores_np[sorted_indices]
    y_true_np = y_true_np[sorted_indices]

    for threshold in np.linspace(0, 1, num_thresholds):
        # Convert scores to binary predictions
        preds = (y_scores_np >= threshold).astype(int)
        print(preds)
        TP = int(sum((y_true_np == 1) & (preds == 1)))
        FP = int(sum((y_true_np == 0) & (preds == 1)))
        FN = int(sum((y_true_np == 1) & (preds == 0)))

        prec = TP / (TP + FP) if TP + FP > 0 else 1
        rec = TP / (TP + FN) if TP + FN > 0 else 1

        precision.append(prec)
        recall.append(rec)

    # Plotting the Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker=".", label="Precision-Recall Curve")
    plt.title("Precision-Recall Curve for Custom Thresholds")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    recall, precision = zip(*sorted(zip(recall, precision)))
    auc = np.trapz(y=precision, x=recall)
    plt.text(0.7, 0.1, "AUC = {:.5f}".format(auc), fontsize=12, ha="center")
    plt.grid(True)
    plt.ylim([0.0, 1.05])  # Slightly above 1 to see top boundary clearly
    plt.xlim([0.0, 1.05])
    plt.legend()

    plt.savefig(save_path)

    plt.close()

    return precision, recall


def prec_recall_premade(logits, labels, category, save_path=None):
    predics = torch.sigmoid(logits)
    predics_np = predics.numpy()
    labels_np = labels.numpy()

    predics_order, labels_order = zip(*sorted(zip(predics_np, labels_np)))

    prec, recall, thresh = precision_recall_curve(labels_order, predics_order)

    disp = PrecisionRecallDisplay(precision=prec, recall=recall)
    disp.plot()
    recall, prec = zip(*sorted(zip(recall, prec)))
    auc = np.trapz(y=prec, x=recall)

    plt.text(0.7, 0.1, "AUC = {:.5f}".format(auc), fontsize=12, ha="center")
    plt.grid(True)
    plt.title(f"Precision-Recall Curve for {category}")
    if save_path:
        plt.savefig(save_path)

    plt.close()

    return prec, recall


def eval_collate_fn(batch):
    return {
        "pixel_values": batch["pixel_values"],
        "labels": torch.vstack(batch["labels"]).to(dtype=torch.float).T,
    }


def eval_model(
    path_to_state,
    experiment_dir,
    data_dir,
    views,
    label_type,
    is_testing,
    can_skip,
    on_server,
    num_samples,
    transform,
):
    model = CustomModel()
    model.load_state_dict(torch.load(path_to_state))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataset = load_dataset(
        "builder.py",
        "builder.py",
        data_dir=data_dir,
        label_type=label_type,
        views=views,
        is_testing=is_testing,
        can_skip=can_skip,
        on_server=on_server,
        num_samples=num_samples,
        split="test",
        trust_remote_code=True,
    ).with_transform(transform)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    threshold = 0.5

    all_logits = []
    all_labels = []
    with torch.no_grad():  # No need to track gradients for evaluation
        for batch in tqdm.tqdm(test_loader):
            # Assuming that the dataset returns a dictionary
            batch = eval_collate_fn(batch)

            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values, labels)
            logits = outputs["logits"]

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    predicted_labels = (all_logits > threshold).long()
    all_labels = all_labels.long()

    figures_dir = os.path.join(experiment_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Saving outputs to {experiment_dir}.")

    with open(os.path.join(experiment_dir, "predicted_labels.npy"), "wb") as f:
        np.save(f, predicted_labels.numpy())

    with open(os.path.join(experiment_dir, "all_labels.npy"), "wb") as f:
        np.save(f, all_labels.numpy())

    with open(os.path.join(experiment_dir, "all_logits.npy"), "wb") as f:
        np.save(f, all_logits.numpy())

    classes = MCSBuilder.classes

    for i, lab in enumerate(classes):
        pred = predicted_labels[:, i]
        label = all_labels[:, i]
        cm = generate_cfm_separate_classes(label, pred, num_classes=2)
        plot_confusion_matrix(
            cm,
            ["Not Found", lab],
            title=f"Confusion for {lab}",
            cmap="viridis",
            save_path=os.path.join(figures_dir, f"{lab}_confusion.png"),
        )

    # Plot TP vs TN binary classification based on no finding and based on all the labels
    # First based on no finidng assuming it is the 8th index
    no_finding_path = os.path.join(figures_dir, "binary_no_findings.png")
    metrics = binary_classification_metrics(all_labels[:, 8], predicted_labels[:, 8])
    plot_classification_metrics(metrics, save_path=no_finding_path)

    # Now true postive and negative based on any other predicitons being postivie
    binary_all_path = os.path.join(figures_dir, "binary_all_class.png")
    preds_modified = copy.deepcopy(predicted_labels)
    preds_modified[:, 8] = 0  # zero out the no findings
    preds_binary = torch.any(preds_modified == 1, dim=1).int()
    labs_modified = copy.deepcopy(all_labels)
    labs_modified[:, 8] = 0  # zero out the no findings
    labs_binary = torch.any(labs_modified == 1, dim=1).int()
    metrics = binary_classification_metrics(labs_binary, preds_binary)
    plot_classification_metrics(metrics, save_path=binary_all_path)

    for i, lab in enumerate(classes):
        log = all_logits[:, i]
        label = all_labels[:, i]

        file_name = f"{lab}_precision_manual_recall.png"
        precision_path = os.path.join(figures_dir, file_name)

        precision, recall = manual_precision_recall_tensor(
            label, log, num_thresholds=1000, save_path=precision_path
        )
        file_name2 = f"{lab}_precision_premade_recall.png"
        precision_path_premade = os.path.join(figures_dir, file_name2)
        precision2, recall2 = prec_recall_premade(
            log, label, lab, precision_path_premade
        )
