from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.data_loader import load_dataset
from src.methods import KernelSVCBen, KernelSVCLilian, KernelLR

def train_model(k, kernel, method='SVM', test_size=0.2, random_state=42):
    """
    Train a Kernel SVM model on dataset k and evaluate on a validation set.

    Args:
        k (int): Dataset index (0,1,2).
        kernel (function): Kernel function (linear or RBF).
        C (float): Regularization parameter.
        method (str): 'ben' or 'lilian' (choose SVM implementation).
        test_size (float): Fraction of data to use for validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        model (object): Trained model.
        val_accuracy (float): Validation accuracy.
    """
    print(f"\nTraining on dataset k={k}")

    X, Y, _ = load_dataset(k)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    if method == 'SVM':
        model = KernelSVCLilian(C=1.0, kernel=kernel)
    else:
        model = KernelLR(kernel=kernel, lmbda=0.01, iters=1000, tol=1.e-5)

    model.fit(X_train, Y_train)

    Y_pred_train = model.predict(X_train)
    Y_pred_val = model.predict(X_val)

    train_acc = accuracy_score(Y_train, Y_pred_train)
    val_acc = accuracy_score(Y_val, Y_pred_val)
    
    print(f"Training Accuracy for k={k}: {train_acc:.4f}")
    print(f"Validation Accuracy for k={k}: {val_acc:.4f}")

    return model, val_acc

def train_all_models(kernel, method='SVM'):
    """
    Train models for all datasets (k=0,1,2) with a validation set.

    Args:
        kernel (function): Kernel function (linear or RBF).
        C (float): Regularization parameter.
        method (str): 'ben' or 'lilian'.

    Returns:
        models (dict): Trained models for k=0,1,2.
        accuracies (dict): Validation accuracies for k=0,1,2.
    """
    models = {}
    accuracies = {}

    for k in range(3):
        model, val_acc = train_model(k, kernel, method)
        models[k] = model
        accuracies[k] = val_acc

    return models, accuracies

