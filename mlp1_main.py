import numpy as np
import pandas as pd
import os
import time

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
        self.seed = seed
        
    def save_weights(self, path):
        """
        Salva os pesos em arquivo
        """
        np.savetxt(path, np.concatenate([
            self.weights_input_hidden.flatten(), 
            self.bias_hidden.flatten(), 
            self.weights_hidden_output.flatten(), 
            self.bias_output.flatten()
        ]), delimiter=',')
        
    def print_weights(self):
        """
        Imprime cada valor dos pesos e bias em linhas separadas
        """
        all_weights = np.concatenate([
            self.weights_input_hidden.flatten(),
            self.bias_hidden.flatten(),
            self.weights_hidden_output.flatten(),
            self.bias_output.flatten()
        ])
        for val in all_weights:
            print(val)
            
    def save_hyperparams(self, path):
        """
        Salva hiperparâmetros em arquivo
        """
        with open(path, 'w') as f:
            f.write(f"n_inputs: {self.weights_input_hidden.shape[0]}\n")
            f.write(f"n_hidden: {self.weights_input_hidden.shape[1]}\n")
            f.write(f"n_outputs: {self.weights_hidden_output.shape[1]}\n")
            f.write(f"learning_rate: {self.learning_rate}\n")
            if self.seed is not None:
                f.write(f"seed: {self.seed}\n")

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
        
    def save_test_outputs(self, X_test, y_test, classes, path):
        """
        Salva as saídas produzidas pela rede neural para cada um dos dados de teste
        
        Parâmetros:
        X_test -- Dados de teste
        y_test -- Rótulos one-hot dos dados de teste
        classes -- Lista de classes (rótulos originais)
        path -- Caminho para salvar o arquivo de saída
        """
        outputs = self.predict(X_test)
        y_pred_labels = np.argmax(outputs, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)
        
        with open(path, 'w') as f:
            f.write("Índice,Classe Verdadeira,Classe Prevista,Probabilidades\n")
            for i, (true_idx, pred_idx) in enumerate(zip(y_true_labels, y_pred_labels)):
                true_class = classes[true_idx]
                pred_class = classes[pred_idx]
                probs = outputs[i]
                probs_str = ",".join([f"{p:.6f}" for p in probs])
                f.write(f"{i},{true_class},{pred_class},{probs_str}\n")
                
        # Calcula e retorna a acurácia
        acc = np.mean(y_pred_labels == y_true_labels)
        return acc
    
    def save_test_outputs(self, X_test, y_test, classes, path):
        """
        Salva as saídas produzidas pela rede neural para cada um dos dados de teste
        
        Parâmetros:
        X_test -- Dados de teste
        y_test -- Rótulos one-hot dos dados de teste
        classes -- Lista de classes (rótulos originais)
        path -- Caminho para salvar o arquivo de saída
        """
        outputs = self.predict(X_test)
        y_pred_labels = np.argmax(outputs, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)
        
        with open(path, 'w') as f:
            f.write("Índice,Classe Verdadeira,Classe Prevista,Probabilidades\n")
            for i, (true_idx, pred_idx) in enumerate(zip(y_true_labels, y_pred_labels)):
                true_class = classes[true_idx]
                pred_class = classes[pred_idx]
                probs = outputs[i]
                probs_str = ",".join([f"{p:.6f}" for p in probs])
                f.write(f"{i},{true_class},{pred_class},{probs_str}\n")
                
        # Calcula e retorna a acurácia
        acc = np.mean(y_pred_labels == y_true_labels)
        return acc
    
    def evaluate_with_confusion_matrix(self, X_test, y_test, classes, save_path=None):
        """
        Avalia o modelo usando matriz de confusão e retorna métricas detalhadas.
        
        Parâmetros:
        X_test -- Dados de teste
        y_test -- Rótulos one-hot dos dados de teste
        classes -- Lista de classes (rótulos originais)
        save_path -- Caminho para salvar a matriz de confusão (opcional)
        
        Retorna:
        confusion_matrix -- Matriz de confusão
        accuracy -- Acurácia global do modelo
        """
        # Obtém as previsões do modelo
        outputs = self.predict(X_test)
        y_pred_labels = np.argmax(outputs, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)
        
        # Calcula a matriz de confusão
        n_classes = len(classes)
        conf_matrix = calculate_confusion_matrix(y_true_labels, y_pred_labels, n_classes)
        
        # Imprime a matriz de confusão
        print_confusion_matrix(conf_matrix, classes)
        
        # Calcula a acurácia
        accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
        
        # Salva a matriz de confusão se um caminho for fornecido
        if save_path:
            save_confusion_matrix(conf_matrix, classes, save_path)
            
        return conf_matrix, accuracy
    
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

def grid_search_mlp(X_train, y_train, X_test, y_test, n_inputs, n_outputs, hidden_grid, lr_grid, epochs=1000, seed=42, parada_antecipada=True, output_dir=None, classes=None): 
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
    best_model = None
    
    for n_hidden in hidden_grid:
        for lr in lr_grid:
            mlp = MLP(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs, learning_rate=lr, seed=seed)
            
            if parada_antecipada:
                diretorio = "saidas/gsp/"
                saida = diretorio + f"pesos_iniciais_{epochs}.txt"
                mlp.save_weights(saida)
                mlp.train(X_train, y_train, epochs)
            else:
                diretorio = "saidas/gs/"
                saida = diretorio + f"pesos_iniciais_{epochs}.txt"
                mlp.save_weights(saida)
                mlp.train_sem_parada(X_train, y_train, epochs)

            # Teste
            outputs = mlp.predict(X_test)
            y_pred_labels = np.argmax(outputs, axis=1)
            y_true_labels = np.argmax(y_test, axis=1)
            acc = np.mean(y_pred_labels == y_true_labels)
            # print(f"Acurácia no teste: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_params = (n_inputs, n_hidden, n_outputs, lr, seed)
                best_model = mlp
                saida = diretorio + f"pesos_finais_{epochs}.txt"
                mlp.save_weights(saida)
    
    # Avalia o melhor modelo com matriz de confusão
    if best_model is not None and classes is not None:
        # Cria diretório se não existir
        os.makedirs(diretorio, exist_ok=True)
        conf_matrix_path = diretorio + f"confusion_matrix_{epochs}.txt"
        best_model.evaluate_with_confusion_matrix(X_test, y_test, classes, save_path=conf_matrix_path)
    
    return best_params, best_acc, best_model

def cross_validate_mlp(X, y, n_splits, n_inputs, n_outputs, hidden_grid, lr_grid, epochs=1000, seed=42, classes=None):
    """
    Definimos uma seed para garantir resultados reproduzíveis
    Criamos índices para todos os exemplos e os embaralhamos aleatoriamente
    Dividimos os dados em 'n_splits' (k) partes aproximadamente iguais, chamadas de folds
    
    Para garantir que cada exemplo apareça exatamente uma vez
    no conjunto de validação, e que a divisão seja aleatória e balanceada.
    """
    # Cria diretório para salvar os pesos

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
    best_model = None
    best_val_data = None  # Para armazenar os dados de validação do melhor modelo
    
    for n_hidden in hidden_grid:
        for lr in lr_grid:
            accs = []
            models = []  # Armazena os modelos treinados em cada fold
            val_data = []  # Armazena os dados de validação de cada fold
            
            for i in range(n_splits):
                val_idx = folds[i]
                train_idx = np.hstack([folds[j] for j in range(n_splits) if j != i])
                X_train_cv, y_train_cv = X[train_idx], y[train_idx]
                X_val_cv, y_val_cv = X[val_idx], y[val_idx]
                mlp = MLP(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs, learning_rate=lr, seed=seed)
                
                # Cria diretório para salvar os pesos
                diretorio = "saidas/cv/"
                os.makedirs(diretorio, exist_ok=True)
                
                # Salva os pesos iniciais
                pesos_iniciais_path = diretorio + f"pesos_iniciais_{epochs}.txt"
                mlp.save_weights(pesos_iniciais_path)
                
                # Treina o modelo
                mlp.train(X_train_cv, y_train_cv, epochs=epochs)
                
                # Salva os pesos finais
                pesos_finais_path = diretorio + f"pesos_finais_{epochs}.txt"
                mlp.save_weights(pesos_finais_path)

                #  Teste
                outputs = mlp.predict(X_val_cv)
                y_pred_labels = np.argmax(outputs, axis=1)
                y_true_labels = np.argmax(y_val_cv, axis=1)
                acc = np.mean(y_pred_labels == y_true_labels)
                # print(f"Acurácia no teste: {acc:.4f}")

                accs.append(acc)
                models.append(mlp)
                val_data.append((X_val_cv, y_val_cv))
            
            mean_acc = np.mean(accs)
            # print(f"Acurácia media: {mean_acc:.4f}")

            # Identificamos a combinação de hiperparâmetros que obteve a maior acurácia média
            # Armazenamos essa configuração ótima
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = (n_inputs, n_hidden, n_outputs, lr, seed)
                best_model_idx = np.argmax(accs)  # Índice do melhor modelo entre os folds
                best_model = models[best_model_idx]
                best_val_data = val_data[best_model_idx]
      # Avalia o melhor modelo com matriz de confusão
    if best_model is not None and classes is not None and best_val_data is not None:
        # Cria diretório para salvar a matriz de confusão
        diretorio = "saidas/cv/"
        os.makedirs(diretorio, exist_ok=True)
        
        # Calcula e salva a matriz de confusão
        conf_matrix_path = diretorio + f"confusion_matrix_{epochs}.txt"
        X_val, y_val = best_val_data
        best_model.evaluate_with_confusion_matrix(X_val, y_val, classes, save_path=conf_matrix_path)
   
    # Retorno dos melhores hiperparâmetros encontrados
    return best_params, best_acc, best_model

def calculate_confusion_matrix(y_true, y_pred, n_classes):
    """
    Calcula a matriz de confusão para um problema de classificação multiclasse.
    
    Parâmetros:
    y_true -- Rótulos verdadeiros (índices das classes)
    y_pred -- Rótulos preditos (índices das classes)
    n_classes -- Número de classes
    
    Retorna:
    conf_matrix -- Matriz de confusão de tamanho (n_classes, n_classes)
                  onde conf_matrix[i, j] é o número de instâncias da classe i
                  que foram classificadas como classe j
    """
    # Inicializa a matriz de confusão com zeros
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Preenche a matriz de confusão
    for true_class, pred_class in zip(y_true, y_pred):
        conf_matrix[true_class, pred_class] += 1
    
    return conf_matrix

def print_confusion_matrix(conf_matrix, classes):
    """
    Imprime a matriz de confusão de forma legível.
    
    Parâmetros:
    conf_matrix -- Matriz de confusão
    classes -- Lista de nomes das classes
    """
    n_classes = len(classes)
    
    # Imprime cabeçalho das colunas
    print("\nMatriz de Confusão:")
    print("Verdadeiro (linhas) vs Predito (colunas)\n")
    
    # Imprime cabeçalho das colunas
    header = "     "
    for j in range(n_classes):
        header += f"{classes[j]:>5} "
    print(header)
    
    # Imprime as linhas da matriz
    for i in range(n_classes):
        row_str = f"{classes[i]:>5} "
        for j in range(n_classes):
            row_str += f"{conf_matrix[i, j]:>5} "
        print(row_str)
    
    # Calcula e imprime métricas de avaliação para cada classe
    print("\nMétricas por classe:")
    header = "Classe     Precisão  Recall    F1-Score"
    print(header)
    
    # Calcula métricas para cada classe
    overall_precision = 0
    overall_recall = 0
    overall_f1 = 0
    
    for i in range(n_classes):
        # True positives: elementos na diagonal
        tp = conf_matrix[i, i]
        # False positives: soma da coluna - true positives
        fp = np.sum(conf_matrix[:, i]) - tp
        # False negatives: soma da linha - true positives
        fn = np.sum(conf_matrix[i, :]) - tp
        
        # Cálculo de precisão, recall e F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Acumula para métricas globais
        overall_precision += precision
        overall_recall += recall
        overall_f1 += f1
        
        print(f"{classes[i]:>10} {precision:>9.4f} {recall:>9.4f} {f1:>9.4f}")
    
    # Imprime métricas globais (média)
    print("\nMétricas globais (média):")
    print(f"Precisão: {overall_precision/n_classes:.4f}")
    print(f"Recall: {overall_recall/n_classes:.4f}")
    print(f"F1-Score: {overall_f1/n_classes:.4f}")
    
    # Calcula acurácia global
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    print(f"Acurácia: {accuracy:.4f}")

def save_confusion_matrix(conf_matrix, classes, path):
    """
    Salva a matriz de confusão e métricas em um arquivo de texto.
    
    Parâmetros:
    conf_matrix -- Matriz de confusão
    classes -- Lista de nomes das classes
    path -- Caminho para salvar o arquivo
    """
    n_classes = len(classes)
    
    with open(path, 'w') as f:
        # Escreve cabeçalho
        f.write("Matriz de Confusão:\n")
        f.write("Verdadeiro (linhas) vs Predito (colunas)\n\n")
        
        # Escreve cabeçalho das colunas
        header = "     "
        for j in range(n_classes):
            header += f"{classes[j]:>5} "
        f.write(header + "\n")
        
        # Escreve as linhas da matriz
        for i in range(n_classes):
            row_str = f"{classes[i]:>5} "
            for j in range(n_classes):
                row_str += f"{conf_matrix[i, j]:>5} "
            f.write(row_str + "\n")
        
        # Calcula e escreve métricas de avaliação para cada classe
        f.write("\nMétricas por classe:\n")
        header = "Classe     Precisão  Recall    F1-Score"
        f.write(header + "\n")
        
        # Calcula métricas para cada classe
        overall_precision = 0
        overall_recall = 0
        overall_f1 = 0
        
        for i in range(n_classes):
            # True positives: elementos na diagonal
            tp = conf_matrix[i, i]
            # False positives: soma da coluna - true positives
            fp = np.sum(conf_matrix[:, i]) - tp
            # False negatives: soma da linha - true positives
            fn = np.sum(conf_matrix[i, :]) - tp
            
            # Cálculo de precisão, recall e F1-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Acumula para métricas globais
            overall_precision += precision
            overall_recall += recall
            overall_f1 += f1
            
            f.write(f"{classes[i]:>10} {precision:>9.4f} {recall:>9.4f} {f1:>9.4f}\n")
        
        # Escreve métricas globais (média)
        f.write("\nMétricas globais (média):\n")
        f.write(f"Precisão: {overall_precision/n_classes:.4f}\n")
        f.write(f"Recall: {overall_recall/n_classes:.4f}\n")
        f.write(f"F1-Score: {overall_f1/n_classes:.4f}\n")
        
        # Calcula acurácia global
        accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
        f.write(f"Acurácia: {accuracy:.4f}\n")
    
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
    hidden_grid = [32, 42, 50, 72, 98, 111, 121, 128]
    lr_grid = [0.1, 0.01, 0.001]
    epochs = [10, 100, 500, 1000, 5000, 10000]
    # epochs = [10, 100]

    output_dir = "saidas"
    os.makedirs(output_dir, exist_ok=True)
    
    # Cria diretórios para as saídas
    os.makedirs("saidas/gs", exist_ok=True)
    os.makedirs("saidas/gsp", exist_ok=True)
    os.makedirs("saidas/cv", exist_ok=True)    # Treinamento sem parada antecipada:
    for epoch in epochs:
        print(f"\nTreinando com {epoch} épocas (sem parada antecipada)...")
        start_time = time.time()
        best_params, best_acc, best_model = grid_search_mlp(
            X_train, y_train, X_test, y_test, 
            n_inputs, n_outputs, 
            hidden_grid, lr_grid, 
            epochs=epoch, 
            parada_antecipada=False, 
            output_dir=output_dir,
            classes=classes  # Passa as classes para calcular a matriz de confusão
        )
        elapsed_time = time.time() - start_time
        print(f"\nMelhor configuração (Grid Search sem parada antecipada): n_inputs={best_params[0]}, n_hidden={best_params[1]}, n_outputs={best_params[2]}, lr={best_params[3]}, seed={best_params[4]}, acurácia={best_acc:.4f}")
        print(f"Tempo de execução (Grid Search sem parada antecipada) para {epoch} épocas: {elapsed_time:.2f} segundos")
        with open(f"saidas/gs/{epoch}_epochs.txt", "w") as f:
            f.write(f"Melhor configuração (Grid Search sem parada antecipada): n_inputs={best_params[0]}, n_hidden={best_params[1]}, n_outputs={best_params[2]}, lr={best_params[3]}, seed={best_params[4]}, acurácia={best_acc:.4f}\n")
            f.write(f"Tempo de execução (Grid Search sem parada antecipada): {elapsed_time:.2f} segundos\n")

    # Treinamento com parada antecipada:
    for epoch in epochs:
        print(f"\nTreinando com {epoch} épocas...")
        start_time = time.time()
        best_params, best_acc, best_model = grid_search_mlp(
            X_train, y_train, X_test, y_test, 
            n_inputs, n_outputs, 
            hidden_grid, lr_grid, 
            epochs=epoch, 
            parada_antecipada=True, 
            output_dir=output_dir,
            classes=classes  # Passa as classes para calcular a matriz de confusão
        )
        elapsed_time = time.time() - start_time
        print(f"\nMelhor configuração (Grid Search com parada antecipada): n_inputs={best_params[0]}, n_hidden={best_params[1]}, n_outputs={best_params[2]}, lr={best_params[3]}, seed={best_params[4]}, acurácia={best_acc:.4f}")
        print(f"Tempo de execução (Grid Search com parada antecipada) para {epoch} épocas: {elapsed_time:.2f} segundos")
        with open(f"saidas/gsp/{epoch}_epochs.txt", "w") as f:
            
            f.write(f"Melhor configuração (Grid Search): n_inputs={best_params[0]}, n_hidden={best_params[1]}, n_outputs={best_params[2]}, lr={best_params[3]}, seed={best_params[4]}, acurácia={best_acc:.4f}\n")
            f.write(f"Tempo de execução (Grid Search com parada antecipada): {elapsed_time:.2f} segundos\n")

    # Validação cruzada:
    for epoch in epochs:
        print(f"\nTreinando com {epoch} épocas (validação cruzada)...")
        start_time = time.time()
        best_params, best_acc, best_model = cross_validate_mlp(
            X_train, y_train, 
            n_splits=5, 
            n_inputs=n_inputs, 
            n_outputs=n_outputs, 
            hidden_grid=hidden_grid, 
            lr_grid=lr_grid, 
            epochs=epoch,
            classes=classes)
        elapsed_time = time.time() - start_time
        print(f"Melhor configuração (CV): n_inputs={best_params[0]}, n_hidden={best_params[1]}, n_outputs={best_params[2]}, lr={best_params[3]}, seed={best_params[4]}, acurácia média={best_acc:.4f}")
        print(f"Tempo de execução (CV) para {epoch} épocas: {elapsed_time:.2f} segundos")
        with open(f"saidas/cv/{epoch}_epochs.txt", "w") as f:
            f.write(f"Melhor configuração (CV): n_inputs={best_params[0]}, n_hidden={best_params[1]}, n_outputs={best_params[2]}, lr={best_params[3]}, seed={best_params[4]}, acurácia média={best_acc:.4f}\n")
            f.write(f"Tempo de execução (CV): {elapsed_time:.2f} segundos\n")