"""Dependency-light multiclass evaluation metrics."""


def classification_report(y_true, y_pred, classes):
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    size = len(classes)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for truth, prediction in zip(y_true, y_pred):
        if not (0 <= truth < size and 0 <= prediction < size):
            raise ValueError("Class index is outside the configured class range")
        matrix[truth][prediction] += 1
    per_class = {}
    for index, name in enumerate(classes):
        tp = matrix[index][index]
        predicted = sum(matrix[row][index] for row in range(size))
        support = sum(matrix[index])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    total = len(y_true)
    accuracy = sum(matrix[i][i] for i in range(size)) / total
    macro = {metric: sum(row[metric] for row in per_class.values()) / size
             for metric in ("precision", "recall", "f1")}
    return {"accuracy": accuracy, "macro_avg": macro, "per_class": per_class,
            "confusion_matrix": matrix, "confusion_matrix_labels": list(classes), "samples": total}
