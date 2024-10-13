import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        # X ist der Eingabedatensatz, y die Labels
        n_samples, n_features = X.shape
        # Initialisieren der Gewichte und des Bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training über mehrere Epochen
        for _ in range(self.n_epochs):
            for idx, x_i in enumerate(X):
                # Lineare Kombination der Eingaben
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Vorhersage mit der Aktivierungsfunktion (Schritt-Funktion)
                y_pred = 1 if linear_output >= 0 else 0

                # Berechnung des Fehlers
                update = self.learning_rate * (y[idx] - y_pred)
                # Aktualisieren der Gewichte und des Bias
                self.weights += update * x_i
                self.bias += update

    def infer(self, X):
        # Vorhersage für neue Daten
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

# Beispiel: Training eines Perzeptrons
if __name__ == "__main__":
    # Beispielhafte Eingabedaten (Punktmenge in der Ebene)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Beispielhafte Labels (AND-Operation)
    y = np.array([0, 0, 0, 1])

    # Erstelle und trainiere das Perzeptron
    perceptron = Perceptron(learning_rate=0.1, n_epochs=10)
    perceptron.train(X, y)

    # Vorhersagen für neue Eingabedaten
    print(perceptron.infer(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])))
