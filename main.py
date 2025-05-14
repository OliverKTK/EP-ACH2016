# main.py
# Implementação manual de um Multilayer Perceptron (MLP) com uma camada escondida
# Não utiliza bibliotecas de redes neurais prontas. Apenas numpy para operações matemáticas.
# Cada passo é explicado em detalhes nos comentários.

import numpy as np  # Usado apenas para operações matemáticas e manipulação de arrays
import pandas as pd  # Adicionado para leitura robusta de CSV
import os

# Função de ativação sigmoide e sua derivada
# A sigmoide "espreme" o valor para o intervalo (0,1), útil para problemas de classificação

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Classe do MLP
class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.1, seed=None):
        """
        n_inputs: número de entradas
        n_hidden: número de neurônios na camada escondida
        n_outputs: número de saídas
        learning_rate: taxa de aprendizado
        seed: semente para reprodutibilidade
        """
        if seed is not None:
            np.random.seed(seed)
        # Inicializa pesos com valores pequenos aleatórios
        self.weights_input_hidden = np.random.uniform(-1, 1, (n_inputs, n_hidden))
        self.bias_hidden = np.random.uniform(-1, 1, (1, n_hidden))
        self.weights_hidden_output = np.random.uniform(-1, 1, (n_hidden, n_outputs))
        self.bias_output = np.random.uniform(-1, 1, (1, n_outputs))
        self.learning_rate = learning_rate

    def forward(self, X):
        """
        Propagação direta: calcula as ativações da camada escondida e da saída
        X: matriz de entradas (amostras x atributos)
        """
        self.input = X
        self.hidden = sigmoid(np.dot(self.input, self.weights_input_hidden) + self.bias_hidden)
        self.output = sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    def backward(self, y_true):
        """
        Backpropagation: ajusta os pesos com base no erro
        y_true: valores esperados (rótulos)
        """
        # Calcula erro da saída
        output_error = y_true - self.output
        output_delta = output_error * sigmoid_deriv(self.output)

        # Calcula erro da camada escondida
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_deriv(self.hidden)

        # Atualiza pesos e bias (Gradiente Descendente)
        self.weights_hidden_output += self.hidden.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += self.input.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

        # Retorna erro médio para monitoramento
        return np.mean(np.abs(output_error))

    def train(self, X, y, epochs=1000, save_error_path=None):
        """
        Treina a rede por várias épocas
        X: entradas
        y: saídas esperadas
        epochs: número de iterações
        save_error_path: caminho para salvar o erro por época
        """
        errors = []
        for epoch in range(epochs):
            self.forward(X)
            error = self.backward(y)
            errors.append(error)
            if epoch % 100 == 0:
                print(f"Época {epoch}, erro médio: {error}")
        if save_error_path:
            with open(save_error_path, 'w') as f:
                for e in errors:
                    f.write(f"{e}\n")
        return errors

    def predict(self, X):
        """
        Realiza a previsão para novos dados
        """
        return self.forward(X)

    def save_weights(self, path_inicial, path_final):
        """
        Salva os pesos iniciais e finais em arquivos
        """
        np.savetxt(path_inicial, np.concatenate([self.weights_input_hidden.flatten(), self.bias_hidden.flatten(), self.weights_hidden_output.flatten(), self.bias_output.flatten()]), delimiter=',')
        np.savetxt(path_final, np.concatenate([self.weights_input_hidden.flatten(), self.bias_hidden.flatten(), self.weights_hidden_output.flatten(), self.bias_output.flatten()]), delimiter=',')

    def save_hyperparams(self, path):
        """
        Salva hiperparâmetros em arquivo
        """
        with open(path, 'w') as f:
            f.write(f"n_inputs: {self.weights_input_hidden.shape[0]}\n")
            f.write(f"n_hidden: {self.weights_input_hidden.shape[1]}\n")
            f.write(f"n_outputs: {self.weights_hidden_output.shape[1]}\n")
            f.write(f"learning_rate: {self.learning_rate}\n")

# Função para carregar dados de um arquivo CSV
# Assume que a última coluna é o rótulo
def load_csv_data(filepath):
    # Usa pandas para lidar com possíveis BOM e formatos variados
    df = pd.read_csv(filepath, header=None)
    data = df.values
    X = data[:, :-1]
    y = data[:, -1:]
    return X, y

# Exemplo de uso com portas lógicas (AND, OR, XOR)
if __name__ == "__main__":
    # Caminho do arquivo de dados (exemplo: problemAND.csv)
    data_path = "portas_logicas/problemOR.csv"
    X, y = load_csv_data(data_path)

    # Cria o MLP: 2 entradas, 4 neurônios escondidos, 1 saída
    mlp = MLP(n_inputs=2, n_hidden=4, n_outputs=1, learning_rate=0.1, seed=42)

    # Diretório de saída
    output_dir = "saidas"
    os.makedirs(output_dir, exist_ok=True)

    # Salva pesos iniciais
    mlp.save_weights(os.path.join(output_dir, "pesos_iniciais.txt"), os.path.join(output_dir, "pesos_iniciais.txt"))
    # Salva hiperparâmetros
    mlp.save_hyperparams(os.path.join(output_dir, "hiperparametros.txt"))

    # Treina a rede
    mlp.train(X, y, epochs=1000, save_error_path=os.path.join(output_dir, "erro_por_iteracao.txt"))

    # Salva pesos finais
    mlp.save_weights(os.path.join(output_dir, "pesos_finais.txt"), os.path.join(output_dir, "pesos_finais.txt"))

    # Testa a rede e salva as saídas
    outputs = mlp.predict(X)
    np.savetxt(os.path.join(output_dir, "saidas_teste.txt"), outputs, delimiter=',')

    print("Treinamento e teste concluídos. Resultados salvos em arquivos na pasta 'saidas'.")
