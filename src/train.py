from sklearn.metrics import accuracy_score
from src.data_loader import load_dataset
from src.methods import KernelSVCBen, KernelSVCLilian

def train_model(k, kernel, C=1.0, method='ben'):
    """
    Train a Kernel SVM model on dataset k.

    Args:
        k (int): Dataset index (0,1,2).
        kernel (function): Kernel function (linear or RBF).
        C (float): Regularization parameter.
        method (str): 'ben' or 'lilian' (choose SVM implementation).

    Returns:
        model (object): Trained model.
        accuracy (float): Training accuracy.
    """
    print(f"\nTraining on dataset k={k}")

    X_train, Y_train, _ = load_dataset(k)

    if method == 'ben':
        model = KernelSVCBen(C=C, kernel=kernel)
    else:
        model = KernelSVCLilian(C=C, kernel=kernel)

    model.fit(X_train, Y_train)

    Y_pred_train = model.predict(X_train)
    acc = accuracy_score(Y_train, Y_pred_train)
    print(f"Training Accuracy for k={k}: {acc:.4f}")

    return model, acc

def train_all_models(kernel, C=1.0, method='ben'):
    """
    Train models for all datasets (k=0,1,2).

    Args:
        kernel (function): Kernel function (linear or RBF).
        C (float): Regularization parameter.
        method (str): 'ben' or 'lilian'.

    Returns:
        models (dict): Trained models for k=0,1,2.
        accuracies (dict): Training accuracies for k=0,1,2.
    """
    models = {}
    accuracies = {}

    for k in range(3):
        model, acc = train_model(k, kernel, C, method)
        models[k] = model
        accuracies[k] = acc

    return models, accuracies
