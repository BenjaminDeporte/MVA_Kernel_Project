from src.train import train_all_models
from src.predict import predict_test_labels
from src.kernels import KernelSpectrum, KernelMismatch, FastKernelMismatch

def main():
    """
    Automates the full pipeline.
    Steps:
    1. Train SVM models for k=0,1,2 using kernel methods.
    2. Predict test labels.
    3. Save results in submission.csv.
    """
    print("\nStarting the pipeline...")

    #kernel_method = KernelSpectrum(k=7)
    kernel_method = FastKernelMismatch(k=6, m=1)

    models, _ = train_all_models(kernel=kernel_method.k_matrix, method='KLR')

    predict_test_labels(models)

    print("\nProcess complete.")

if __name__ == "__main__":
    main()


