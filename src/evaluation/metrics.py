""" Evaluation metrics """


import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def soft_cross_entropy(
    teacher_probabilities: np.ndarray,
    student_probabilities: np.ndarray,
) -> float:
    """ Compute soft cross-entropy between student and teacher probabilities. """
    student_probabilities = np.clip(
        student_probabilities,
        1e-15,
        1.0,
    )

    # Loss = - sum(teacher_probabilities * log(student_probabilities))
    return float(
        -np.mean(
            np.sum(
                teacher_probabilities
                * np.log(student_probabilities),
                axis=1,
            )
        )
    )


def evaluate_window(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    inference_seconds: float,
    teacher_probabilities: np.ndarray | None = None,
) -> dict:
    """  Evaluate predictions for one window. 
    Args:
        true_labels: Ground-truth labels
        probabilities: Model prediction probabilities
        classes: Classes
        inference_seconds: Inference time
        teacher_probabilities: Teacher prediction probabilities
    """
    # Predict labels by taking the maximum probability.
    predicted_labels = classes[
        np.argmax(probabilities, axis=1)
    ]

    result = {
        # Accuracy
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),

        # Multiclass macro-F1.
        "macro_f1": f1_score(
            true_labels,
            predicted_labels,
            labels=classes,
            average="macro",
            zero_division=0,
        ),

        # Inference time
        "inference_seconds": inference_seconds,

        # Throughput
        "throughput": (
            len(true_labels) / inference_seconds
        ),
    }

    if teacher_probabilities is not None:
        teacher_labels = classes[
            np.argmax(
                teacher_probabilities,
                axis=1,
            )
        ]

        # Student-teacher prediction agreement.
        result["teacher_fidelity"] = np.mean(
            predicted_labels == teacher_labels
        )   

        # Soft cross-entropy between teacher and student probabilities.
        result["soft_cross_entropy"] = (
            soft_cross_entropy(
                teacher_probabilities,
                probabilities,
            )
        )

    return result
