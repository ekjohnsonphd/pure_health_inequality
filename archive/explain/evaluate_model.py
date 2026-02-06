import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    PrecisionRecallDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import os


def evaluate_model(
    model, X_test, y_test, id_test, output_dir, threshold=0.5, metadata=None
):
    os.makedirs(output_dir, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    if metadata:
        metrics.update(metadata)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, "test_metrics.csv"), index=False
    )
    
    preds_df = pd.DataFrame({
        "pnr": id_test.values,
        "y_test": y_test.values if hasattr(y_test, "values") else y_test,
        "y_proba":y_proba,
        "y_pred": y_pred
    })
    preds_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # precision recall curve
    display = PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.figure()
    display.plot()
    plt.title("Precision-recall curve")
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.pdf"))
    plt.close()

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = ["actual_0", "actual_1"]
    pred_labels = ["pred_0", "pred_1"]
    
    cm_df = pd.DataFrame(cm, index=labels, columns=pred_labels)
    
    cm_df["Total"] = cm_df.sum(axis=1)
    total_row = cm_df.sum(axis=0)
    total_row_name = "Total"
    cm_df = pd.concat([cm_df, total_row.to_frame().T])
    
    cm_df.to_csv(os.path.join(output_dir,"confusion_matrix.csv"))
    
    # cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    # plt.figure()
    # cm_display.plot()
    # plt.title("Confusion matrix")
    # plt.savefig(os.path.join(output_dir, "confusion_matrix.pdf"))
    # plt.close()

    return metrics
