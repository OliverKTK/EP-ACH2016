import numpy as np
import pandas as pd
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1. - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Inicialização mais estável dos pesos
        self.weights_input_hidden = np.random.randn(n_inputs, n_hidden) * 0.1
        self.bias_hidden = np.zeros((1, n_hidden))
        self.weights_hidden_output = np.random.randn(n_hidden, n_outputs) * 0.1
        self.bias_output = np.zeros((1, n_outputs))
        self.learning_rate = learning_rate

    def forward(self, X):
        self.input = X
        self.hidden = sigmoid(np.dot(self.input, self.weights_input_hidden) + self.bias_hidden)
        self.output = softmax(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    def backward(self, y_true):
        output_error = self.output - y_true
        output_delta = output_error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_deriv(self.hidden)
        self.weights_hidden_output -= self.hidden.T.dot(output_delta) * self.learning_rate
        self.bias_output -= np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden -= self.input.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden -= np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
        return cross_entropy(y_true, self.output)

    def train(self, X, y, epochs, save_error_path=None, decay=0.99, patience=10):
        errors = []
        best_error = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            self.forward(X)
            error = self.backward(y)
            errors.append(error)
            if error < best_error:
                best_error = error
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                self.learning_rate *= decay
                patience_counter = 0
            # print(f"Época {epoch}, erro médio: {error}, lr: {self.learning_rate:.6f}")  # <-- Agora imprime a cada época
        if save_error_path:
            with open(save_error_path, 'w') as f:
                for e in errors:
                    f.write(f"{e}\n")
        return errors
        
    def train_sem_parada(self, X, y, epochs, save_error_path=None):
        """
        Treina a rede neural sem mecanismo de parada antecipada.
        Nesta versão, a taxa de aprendizado permanece constante durante
        todo o treinamento e todas as épocas serão executadas.
        """
        errors = []
        
        for epoch in range(epochs):
            self.forward(X)
            error = self.backward(y)
            errors.append(error)
            
            # print(f"Época {epoch}, erro médio: {error}, lr: {self.learning_rate:.6f}")
            
        if save_error_path:
            with open(save_error_path, 'w') as f:
                for e in errors:
                    f.write(f"{e}\n")
        return errors

    def predict(self, X):
        return self.forward(X)
    
def one_hot_encode(y):
    """
    Converte rótulos categóricos em vetores para uso em redes neurais.
    
    A codificação one-hot transforma cada categoria em um vetor binário onde apenas
    uma posição contém o valor 1 (correspondente à classe) e todas as outras são 0.
    Esta representação é essencial para problemas de classificação multiclasse em
    redes neurais, pois permite:
        Trabalhar com categorias não-numéricas
        Evitar implicar relações de ordem entre as classes
        Compatibilidade com funções de perda como cross-entropy
    """
    classes = sorted(list(set(y)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y])
    one_hot = np.zeros((len(y), len(classes)))
    one_hot[np.arange(len(y)), y_idx] = 1
    return one_hot, classes

def grid_search_mlp(X_train, y_train,X_test, y_test, n_inputs, n_outputs, hidden_grid, lr_grid, epochs=1000, seed=None):
    """
    Realiza uma busca em grade (grid search) para encontrar a melhor configuração
    de hiperparâmetros para a rede neural MLP.
    
    Para cada combinação de hiperparâmetros:
        Treina o modelo nos dados de treinamento
        Avalia o modelo nos dados de teste
        Registra a acurácia obtida
    """
    best_acc = 0
    best_params = None
    for n_hidden in hidden_grid:
        for lr in lr_grid:
            mlp = MLP(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs, learning_rate=lr, seed=seed)
            mlp.train(X_train, y_train, epochs)

            # Teste
            outputs = mlp.predict(X_test)
            y_pred_labels = np.argmax(outputs, axis=1)
            y_true_labels = np.argmax(y_test, axis=1)
            acc = np.mean(y_pred_labels == y_true_labels)
            # print(f"Acurácia no teste: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_params = (n_hidden, lr)
    
    return best_params, best_acc

def cross_validate_mlp(X, y, n_splits, n_inputs, n_outputs, hidden_grid, lr_grid, epochs=1000, seed=42):
    """
    Definimos uma seed para garantir resultados reproduzíveis
    Criamos índices para todos os exemplos e os embaralhamos aleatoriamente
    Dividimos os dados em 'n_splits' (k) partes aproximadamente iguais, chamadas de folds
    
    Para garantir que cada exemplo apareça exatamente uma vez
    no conjunto de validação, e que a divisão seja aleatória e balanceada.
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop

    """
    Para cada combinação de hiperparâmetros:
        Para cada um dos k folds:
            Usamos o fold atual como conjunto de validação
            Usamos os k-1 folds restantes como conjunto de treinamento
            Treinamos o modelo nos dados de treinamento
            Avaliamos o modelo nos dados de validação
            Registramos a acurácia obtida
        Calculamos a acurácia média entre todos os k folds
    
    Para obter uma melhor estimativa do desempenho do modelo para cada 
    configuração de hiperparâmetros, reduzindo o impacto da variabilidade
    na divisão dos dados.
    """
    best_acc = 0
    best_params = None
    for n_hidden in hidden_grid:
        for lr in lr_grid:
            accs = []
            for i in range(n_splits):
                val_idx = folds[i]
                train_idx = np.hstack([folds[j] for j in range(n_splits) if j != i])
                X_train_cv, y_train_cv = X[train_idx], y[train_idx]
                X_val_cv, y_val_cv = X[val_idx], y[val_idx]
                mlp = MLP(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs, learning_rate=lr, seed=seed)
                mlp.train(X_train_cv, y_train_cv, epochs=epochs)

                #  Teste
                outputs = mlp.predict(X_val_cv)
                y_pred_labels = np.argmax(outputs, axis=1)
                y_true_labels = np.argmax(y_val_cv, axis=1)
                acc = np.mean(y_pred_labels == y_true_labels)
                # print(f"Acurácia no teste: {acc:.4f}")

                accs.append(acc)
            mean_acc = np.mean(accs)
            # print(f"Acurácia media: {mean_acc:.4f}")

            # Identificamos a combinação de hiperparâmetros que obteve a maior acurácia média
            # Armazenamos essa configuração ótima
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = (n_hidden, lr)
    
    # Retorno dos melhores hiperparâmetros encontrados
    return best_params, best_acc

if __name__ == "__main__":
    # Carrega X e y
    X = pd.read_csv("caracteres_completo/X.txt", header=None, delimiter=",", na_values=[' ', '']).fillna(0).astype(float).values
    with open("caracteres_completo/Y_letra.txt", encoding="utf-8") as f:
        y_raw = [line.strip() for line in f.readlines()]

    # Normalização robusta
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    ranges = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
    X = (X - min_vals) / ranges

    # Separa treino e teste
    X_train, X_test = X[:-130], X[-130:]
    y_train_raw, y_test_raw = y_raw[:-130], y_raw[-130:]

    # One-hot encode dos rótulos
    y_train, classes = one_hot_encode(y_train_raw)
    y_test, _ = one_hot_encode(y_test_raw)

    # Parâmetros do MLP
    n_inputs = X_train.shape[1]
    n_outputs = y_train.shape[1]

    # Defina os grids de hiperparâmetros
    hidden_grid = [64, 128, 256]
    lr_grid = [0.01, 0.001, 0.0005]

    best_params, best_acc = grid_search_mlp(X_train, y_train, X_test, y_test, n_inputs, n_outputs, hidden_grid, lr_grid, epochs=1000, seed=42)
    print(f"\nMelhor configuração (Grid Search): n_hidden={best_params[0]}, learning_rate={best_params[1]}, acurácia={best_acc:.4f}")

    # Exemplo de uso da validação cruzada:
    # best_params, best_acc = cross_validate_mlp(X_train, y_train, n_splits=5, n_inputs=n_inputs, n_outputs=n_outputs, hidden_grid=hidden_grid, lr_grid=lr_grid, epochs=300)
    # print(f"Melhor configuração (CV): n_hidden={best_params[0]}, learning_rate={best_params[1]}, acurácia média={best_acc:.4f}")
