import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings

warnings.filterwarnings('ignore')

# Función para crear la red neuronal con backpropagation
class RedNeuronalBackpropagation:
    def __init__(self, n_entrada, n_oculta1, n_oculta2, n_salida, alfa=0.25, error_umbral=0.001, max_iter=10000):
        # Parámetros de la red
        self.n_entrada = n_entrada
        self.n_oculta1 = n_oculta1
        self.n_oculta2 = n_oculta2
        self.n_salida = n_salida
        self.alfa = alfa
        self.error_umbral = error_umbral
        self.max_iter = max_iter

        # Inicialización de pesos aleatorios (< 0.5)
        self.pesos_entrada_oculta1 = np.random.uniform(0, 0.5, (n_entrada, n_oculta1))
        self.pesos_oculta1_oculta2 = np.random.uniform(0, 0.5, (n_oculta1, n_oculta2))
        self.pesos_oculta2_salida = np.random.uniform(0, 0.5, (n_oculta2, n_salida))

        # Inicialización de bias
        self.bias_oculta1 = np.random.uniform(0, 0.5, (1, n_oculta1))
        self.bias_oculta2 = np.random.uniform(0, 0.5, (1, n_oculta2))
        self.bias_salida = np.random.uniform(0, 0.5, (1, n_salida))

        # Historial de errores
        self.historial_errores = []

    # Función de activación (sigmoide)
    def sigmoide(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivada de la función sigmoide
    def derivada_sigmoide(self, x):
        return x * (1 - x)

    # Función de activación softmax
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    # Propagación hacia adelante
    def forward(self, X):
        # Capa oculta 1
        self.z_oculta1 = np.dot(X, self.pesos_entrada_oculta1) + self.bias_oculta1
        self.activacion_oculta1 = self.sigmoide(self.z_oculta1)

        # Capa oculta 2
        self.z_oculta2 = np.dot(self.activacion_oculta1, self.pesos_oculta1_oculta2) + self.bias_oculta2
        self.activacion_oculta2 = self.sigmoide(self.z_oculta2)

        # Capa de salida
        self.z_salida = np.dot(self.activacion_oculta2, self.pesos_oculta2_salida) + self.bias_salida
        self.activacion_salida = self.softmax(self.z_salida)

        return self.activacion_salida

    # Entrenamiento con backpropagation
    def entrenar(self, X, y):
        num_muestras = X.shape[0]
        iteraciones = 0
        error_actual = float('inf')

        while error_actual > self.error_umbral and iteraciones < self.max_iter:
            # Forward pass
            output = self.forward(X)

            # Calcular error
            error = y - output
            error_cuadratico_medio = np.mean(np.square(error))
            self.historial_errores.append(error_cuadratico_medio)
            error_actual = error_cuadratico_medio

            # Backpropagation
            # Delta de salida
            delta_salida = error * self.derivada_sigmoide(output)

            # Delta de capa oculta 2
            error_oculta2 = delta_salida.dot(self.pesos_oculta2_salida.T)
            delta_oculta2 = error_oculta2 * self.derivada_sigmoide(self.activacion_oculta2)

            # Delta de capa oculta 1
            error_oculta1 = delta_oculta2.dot(self.pesos_oculta1_oculta2.T)
            delta_oculta1 = error_oculta1 * self.derivada_sigmoide(self.activacion_oculta1)

            # Actualizar pesos y bias
            # Pesos entre capa oculta 2 y salida
            self.pesos_oculta2_salida += self.activacion_oculta2.T.dot(delta_salida) * self.alfa
            self.bias_salida += np.sum(delta_salida, axis=0, keepdims=True) * self.alfa

            # Pesos entre capa oculta 1 y capa oculta 2
            self.pesos_oculta1_oculta2 += self.activacion_oculta1.T.dot(delta_oculta2) * self.alfa
            self.bias_oculta2 += np.sum(delta_oculta2, axis=0, keepdims=True) * self.alfa

            # Pesos entre entrada y capa oculta 1
            self.pesos_entrada_oculta1 += X.T.dot(delta_oculta1) * self.alfa
            self.bias_oculta1 += np.sum(delta_oculta1, axis=0, keepdims=True) * self.alfa

            iteraciones += 1

            if iteraciones % 1000 == 0:
                print(f"Iteración {iteraciones}, Error: {error_actual}")

        return iteraciones, error_actual

    # Predecir
    def predecir(self, X):
        return self.forward(X)

    # Guardar modelo
    def guardar_modelo(self, nombre_archivo='modelo_red_neuronal.pkl'):
        modelo = {
            'pesos_entrada_oculta1': self.pesos_entrada_oculta1,
            'pesos_oculta1_oculta2': self.pesos_oculta1_oculta2,
            'pesos_oculta2_salida': self.pesos_oculta2_salida,
            'bias_oculta1': self.bias_oculta1,
            'bias_oculta2': self.bias_oculta2,
            'bias_salida': self.bias_salida,
            'historial_errores': self.historial_errores
        }
        with open(nombre_archivo, 'wb') as f:
            pickle.dump(modelo, f)
        print(f"Modelo guardado en {nombre_archivo}")

    # Cargar modelo
    def cargar_modelo(self, nombre_archivo='modelo_red_neuronal.pkl'):
        with open(nombre_archivo, 'rb') as f:
            modelo = pickle.load(f)

        self.pesos_entrada_oculta1 = modelo['pesos_entrada_oculta1']
        self.pesos_oculta1_oculta2 = modelo['pesos_oculta1_oculta2']
        self.pesos_oculta2_salida = modelo['pesos_oculta2_salida']
        self.bias_oculta1 = modelo['bias_oculta1']
        self.bias_oculta2 = modelo['bias_oculta2']
        self.bias_salida = modelo['bias_salida']
        self.historial_errores = modelo['historial_errores']
        print(f"Modelo cargado desde {nombre_archivo}")

    # Mostrar pesos
    def mostrar_pesos(self):
        print("Pesos entre capa de entrada y primera capa oculta:")
        print(self.pesos_entrada_oculta1)
        print("\nPesos entre primera capa oculta y segunda capa oculta:")
        print(self.pesos_oculta1_oculta2)
        print("\nPesos entre segunda capa oculta y capa de salida:")
        print(self.pesos_oculta2_salida)
        print("\nBias de primera capa oculta:")
        print(self.bias_oculta1)
        print("\nBias de segunda capa oculta:")
        print(self.bias_oculta2)
        print("\nBias de capa de salida:")
        print(self.bias_salida)

# Función para transformar la salida a valores discretos
def discretizar_salida(y_pred):
    return np.argmax(y_pred, axis=1)

# Función para realizar validación Leave-One-Out con la red neuronal
def leave_one_out_validation(X, y, nombres, n_entrada, n_oculta1, n_oculta2, n_salida, alfa=0.25, error_umbral=0.001):
    loo = LeaveOneOut()
    predicciones = np.zeros(y.shape[0])

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Crear y entrenar modelo
        modelo = RedNeuronalBackpropagation(n_entrada, n_oculta1, n_oculta2, n_salida, alfa, error_umbral)
        modelo.entrenar(X_train, y_train)

        # Predecir
        y_pred = modelo.predecir(X_test)
        predicciones[test_index] = discretizar_salida(y_pred)

        print(f"Validación {test_index[0] + 1}/{y.shape[0]}: "
              f"\tNombre: {nombres[test_index[0]]}, \tEtiqueta real: {np.argmax(y_test)}, \tPredicción: {predicciones[test_index][0]}")

    # Calcular métricas
    accuracy = accuracy_score(np.argmax(y, axis=1), predicciones)
    precision = precision_score(np.argmax(y, axis=1), predicciones, average='macro', zero_division=0)
    recall = recall_score(np.argmax(y, axis=1), predicciones, average='macro', zero_division=0)
    f1 = f1_score(np.argmax(y, axis=1), predicciones, average='macro', zero_division=0)

    print("\nResultados de validación Leave-One-Out con Red Neuronal:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return accuracy, precision, recall, f1

# Función para realizar validación Leave-One-Out con Random Forest
def leave_one_out_validation_rf(X, y, nombres):
    loo = LeaveOneOut()
    predicciones = np.zeros(y.shape[0])

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Crear y entrenar modelo
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        # Predecir
        y_pred = modelo.predict(X_test)
        predicciones[test_index] = y_pred

        print(f"Validación {test_index[0] + 1}/{y.shape[0]}: "
              f"\tNombre: {nombres[test_index[0]]}, \tEtiqueta real: {y_test[0]}, \tPredicción: {y_pred[0]}")

    # Calcular métricas
    accuracy = accuracy_score(y, predicciones)
    precision = precision_score(y, predicciones, average='macro', zero_division=0)
    recall = recall_score(y, predicciones, average='macro', zero_division=0)
    f1 = f1_score(y, predicciones, average='macro', zero_division=0)

    print("\nResultados de validación Leave-One-Out con Random Forest:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return accuracy, precision, recall, f1

# Función principal
def main():
    # Cargar datos
    print("Cargando datos...")
    try:
        df = pd.read_csv('cinco.csv', encoding='utf-8')
    except UnicodeDecodeError:
        print("Error al leer el archivo con codificación UTF-8. Intentando con otras codificaciones...")
        try:
            df = pd.read_csv('cinco.csv', encoding='latin1')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv('cinco.csv', encoding='ISO-8859-1')
            except Exception as e:
                print(f"No se pudo leer el archivo: {e}")
                return

    # Mostrar información del dataframe
    print("\nColumnas del dataset:")
    for i, col in enumerate(df.columns):
        print(f"{i + 1}: {col}")
    print("\nPrimeros registros:")
    print(df.head())

    # Pedir al usuario que especifique las columnas de características y la columna de etiquetas
    input_caract = input(
        "Por favor, ingrese las columnas de características (números o nombres separados por comas): ").strip()
    input_etiqueta = input("Por favor, ingrese la columna de etiquetas (número o nombre): ").strip()

    # Procesar entradas para características
    cols_caracteristicas = []
    for item in input_caract.split(','):
        item = item.strip()
        if not item:  # Salta elementos vacíos
            continue

        try:
            # Si es un número, convertir a índice de columna (restando 1 porque el usuario ve índices desde 1)
            idx = int(item) - 1
            if 0 <= idx < len(df.columns):
                cols_caracteristicas.append(df.columns[idx])
            else:
                print(f"Advertencia: Índice {item} fuera de rango, ignorado.")
        except ValueError:
            # Si no es un número, usar el nombre directamente
            if item in df.columns:
                cols_caracteristicas.append(item)
            else:
                print(f"Advertencia: Columna '{item}' no encontrada, ignorada.")

    # Procesar columna de etiqueta
    col_etiqueta = None
    input_etiqueta = input_etiqueta.strip()
    try:
        idx = int(input_etiqueta) - 1
        if 0 <= idx < len(df.columns):
            col_etiqueta = df.columns[idx]
        else:
            print(f"Error: Índice de etiqueta {input_etiqueta} fuera de rango.")
    except ValueError:
        if input_etiqueta in df.columns:
            col_etiqueta = input_etiqueta
        else:
            print(f"Error: Columna de etiqueta '{input_etiqueta}' no encontrada.")

    if not cols_caracteristicas or col_etiqueta is None:
        print("Error: No se pudo procesar correctamente las columnas seleccionadas.")
        return

    print("\nColumnas de características seleccionadas:")
    for col in cols_caracteristicas:
        print(f"- {col}")
    print(f"Columna de etiqueta seleccionada: {col_etiqueta}")

    # Separar características, etiquetas y nombres
    # Verificar si la columna 'Nombre' existe, sino usar un índice
    nombres = df['Nombre'].values if 'Nombre' in df.columns else np.arange(len(df))
    X = df[cols_caracteristicas].values
    y = df[col_etiqueta].values

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = np.eye(len(label_encoder.classes_))[y_encoded]

    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Definir parámetros de la red
    n_entrada = X_scaled.shape[1]  # Número de características
    n_oculta1 = 8  # Primera capa oculta
    n_oculta2 = 6  # Segunda capa oculta
    n_salida = len(label_encoder.classes_)  # Salida (número de clases)
    alfa = 0.25  # Tasa de aprendizaje
    error_umbral = 0.001  # Error objetivo

    # Crear la red neuronal
    red_neuronal = RedNeuronalBackpropagation(n_entrada, n_oculta1, n_oculta2, n_salida, alfa, error_umbral)

    # Entrenar la red
    print("\nEntrenando la red neuronal...")
    iteraciones, error_final = red_neuronal.entrenar(X_scaled, y_onehot)

    print(f"\nEntrenamiento completado en {iteraciones} iteraciones")
    print(f"Error final: {error_final:.6f}")

    # Mostrar los pesos finales
    print("\nPesos finales del modelo:")
    red_neuronal.mostrar_pesos()

    # Guardar el modelo
    red_neuronal.guardar_modelo('modelo_backpropagation.pkl')

    # Predecir y evaluar
    print("\nRealizando predicciones...")
    y_pred_prob = red_neuronal.predecir(X_scaled)
    y_pred = discretizar_salida(y_pred_prob)

    # Mostrar resultados por alumno
    print("\nResultados por Especie:")
    for i in range(len(nombres)):
        print(
            f"Especie: {nombres[i]}, Etiqueta real: {y_encoded[i]}, Etiqueta predicha: {y_pred[i]}, Probabilidad: {y_pred_prob[i][y_pred[i]]:.4f}")

    # Calcular matriz de confusión
    cm = confusion_matrix(y_encoded, y_pred)
    print("\nMatriz de confusión:")
    print(cm)

    # Visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.savefig('matriz_confusion.png')

    # Calcular métricas de clasificación
    accuracy = accuracy_score(y_encoded, y_pred)
    precision = precision_score(y_encoded, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_encoded, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_encoded, y_pred, average='macro', zero_division=0)

    print("\nMétricas de clasificación con Red Neuronal:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensibilidad): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Gráfico de la curva de aprendizaje
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(red_neuronal.historial_errores)), red_neuronal.historial_errores)
    plt.title('Curva de Aprendizaje')
    plt.xlabel('Iteraciones')
    plt.ylabel('Error Cuadrático Medio')
    plt.grid(True)
    plt.savefig('curva_aprendizaje.png')
    plt.show()

    # Validación cruzada Leave-One-Out con Red Neuronal
    print("\nRealizando validación cruzada Leave-One-Out con Red Neuronal...")
    leave_one_out_validation(X_scaled, y_onehot, nombres, n_entrada, n_oculta1, n_oculta2, n_salida, alfa, error_umbral)

    # Validación cruzada Leave-One-Out con Random Forest
    print("\nRealizando validación cruzada Leave-One-Out con Random Forest...")
    leave_one_out_validation_rf(X_scaled, y_encoded, nombres)

if __name__ == "__main__":
    main()
