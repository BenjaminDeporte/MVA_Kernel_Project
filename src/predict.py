import numpy as np
import pandas as pd
from src.data_loader import load_dataset

def predict_test_labels(models):
    """
    Predict test labels using trained models.

    Args:
        models (dict): Trained SVM models.

    Returns:
        submission_df (DataFrame): Final submission dataframe.
    """
    submission = []

    for k in range(3):
        print(f"\nPredicting on test dataset k={k}")

        _, _, X_test = load_dataset(k)

        model = models[k]
        Y_pred = model.predict(X_test)
        Y_pred = np.array(Y_pred >= 0, dtype=int)

        ids = [1000 * k + i for i in range(len(Y_pred))]

        submission.append(pd.DataFrame({"Id": ids, "Bound": Y_pred}))

    submission_df = pd.concat(submission)
    submission_df.to_csv("submission.csv", index=False)
    print("\nSubmission file 'submission.csv' created!")

    return submission_df
