import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def get_sample_weights(y):
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(y), y=y,
        )

    sample_weights = ((np.ones(y.shape) - y)*class_weights[0]
                      + y*class_weights[1])

    return sample_weights


class NeuralNetworkClassifier(nn.Module):
    def __init__(self, input_dim, layers,):
        super(NeuralNetworkClassifier, self).__init__()
        self.layers = []
        self.input_dim = input_dim

        # first layer
        self.layers.append(nn.Linear(self.input_dim, layers[0]))
        self.layers.append(nn.ReLU())

        # hidden layers
        for idx, nodes in enumerate(layers):
            if idx == 0:
                continue
            else:
                self.layers.append(nn.Linear(layers[idx-1], nodes))
                self.layers.append(nn.ReLU())

        # final layer
        self.layers.append(nn.Linear(layers[-1], 1))
        self.layers.append(nn.Sigmoid())

        self.model_stack = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model_stack(x)


class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, layers,
                 learning_rate=0.001, epochs=100, batch_size=32,
                 validation_fraction=0.1, split_seed=42, patience=5,
                 weight_decay=0.0, input_size=None,
                 tol=1e-7):

        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.split_seed = split_seed
        self.patience = patience
        if self.patience == None:
            self.patience = self.epochs
        self.tol = tol
        self.weight_decay = weight_decay
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.train_losses = None
        self.val_losses = None
        if input_size is not None:
            self.input_size = input_size
            self.model = NeuralNetworkClassifier(
                self.input_size, self.layers
                ).to(self.device)
        else:
            self.input_size = None

    def fit(self, X, y, sample_weights=None):
        X, y = check_X_y(X, y)

        if sample_weights is None:
            sample_weights = torch.ones(X.shape[0])

        if self.validation_fraction is not None:
            if sample_weights == "balanced":
                (X_train, X_val, y_train, y_val) = train_test_split(
                    X, y, test_size=self.validation_fraction,
                    random_state=self.split_seed)

                weights_train = get_sample_weights(y_train)
                weights_val = get_sample_weights(y_val)
            else:
                (X_train, X_val, y_train, y_val,
                 weights_train, weights_val) = train_test_split(
                     X, y, sample_weights, test_size=self.validation_fraction,
                     random_state=self.split_seed)

            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val),
                torch.FloatTensor(weights_val)
            )

            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=True
            )

            best_val_loss = float('inf')
            self.val_losses = []

        else:
            X_train = X
            y_train = y

            if sample_weights == "balanced":
                weights_train = get_sample_weights(y_train)
            else:
                weights_train = sample_weights

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
            torch.FloatTensor(weights_train)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.input_size = X_train.shape[1]
        self.model = NeuralNetworkClassifier(
            self.input_size, self.layers
        ).to(self.device)

        criterion = nn.functional.binary_cross_entropy

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        patience_counter = 0
        self.train_losses = []

        for epoch in range(self.epochs):
            running_loss = 0.0
            self.model.train()
            for inputs, labels, weights in train_loader:
                inputs, labels, weights = (
                    inputs.to(self.device),
                    labels.to(self.device).reshape(-1, 1),
                    weights.to(self.device).reshape(-1, 1))

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels, weight=weights)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            self.train_losses.append(running_loss/len(train_loader))

            if self.validation_fraction is None:
                print((f"Epoch {epoch+1}/{self.epochs}, "
                       f"Loss: {running_loss/len(train_loader)}"))

                continue

            running_val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for inputs, labels, weights in val_loader:
                    inputs, labels, weights = (
                        inputs.to(self.device),
                        labels.to(self.device).reshape(-1, 1),
                        weights.to(self.device).reshape(-1, 1))

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels, weight=weights)
                    running_val_loss += loss.item()

                tmp_val_loss = running_val_loss/len(val_loader)

            self.val_losses.append(tmp_val_loss)

            if tmp_val_loss < (best_val_loss - self.tol):
                best_val_loss = tmp_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            print((f"Epoch {epoch+1}/{self.epochs}, "
                   f"Loss: {running_loss/len(train_loader)}, "
                   f"Val Loss: {tmp_val_loss}"))

            if patience_counter >= self.patience:
                print((f"Early stopping at epoch {epoch+1} "
                       "due to lack of improvement in validation loss."))
                break

        if self.validation_fraction is not None:
            self.model.load_state_dict(best_model_state)

        return self

    def predict_proba(self, X):
        #check_is_fitted(self, "model")
        X = check_array(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)

        return outputs.cpu().numpy().flatten()
