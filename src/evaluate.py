import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze().cpu().numpy()
        truth = y.cpu().numpy()

        print("RMSE:", np.sqrt(mean_squared_error(truth, preds)))
        print("MAE:", mean_absolute_error(truth, preds))
        print("R² Score:", r2_score(truth, preds))

        plt.figure(figsize=(10,5))
        plt.plot(truth[:200], label="Actual")
        plt.plot(preds[:200], label="Predicted")
        plt.legend()
        plt.title("Bandwidth Prediction for Indian Locations")
        plt.savefig("prediction_plot.png")
        plt.show()
