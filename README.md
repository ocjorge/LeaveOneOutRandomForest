# Red Neuronal con Backpropagation y Validación

## Descripción
Este proyecto implementa una red neuronal artificial con backpropagation y validación Leave-One-Out. Además, se incluye una comparación de resultados con un clasificador Random Forest. El código permite cargar un dataset en formato CSV, seleccionar las columnas de características y etiquetas, y entrenar modelos para evaluar su desempeño.

## Requisitos
Antes de ejecutar el código, asegúrese de tener instaladas las siguientes bibliotecas de Python:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Archivos principales
- `main.py`: Contiene la implementación de la red neuronal, la validación Leave-One-Out y la evaluación con Random Forest.
- `cinco.csv`: Archivo de datos utilizado para entrenar y evaluar los modelos.

## Estructura del Código
### Clases y Funciones Principales
- **`RedNeuronalBackpropagation`**: Implementa una red neuronal de tres capas con backpropagation.
  - `entrenar(X, y)`: Entrena la red neuronal con los datos proporcionados.
  - `predecir(X)`: Realiza predicciones sobre nuevos datos.
  - `guardar_modelo(nombre_archivo)`: Guarda los pesos y bias del modelo entrenado.
  - `cargar_modelo(nombre_archivo)`: Carga un modelo previamente guardado.
- **`leave_one_out_validation(X, y, nombres, n_entrada, n_oculta1, n_oculta2, n_salida)`**: Realiza la validación Leave-One-Out con la red neuronal.
- **`leave_one_out_validation_rf(X, y, nombres)`**: Realiza la validación Leave-One-Out utilizando Random Forest.
- **`discretizar_salida(y_pred)`**: Transforma la salida de la red neuronal en valores discretos.

## Uso
1. Ejecute el script `main.py`:

   ```bash
   python main.py
   ```

2. Se solicitará ingresar las columnas de características y la columna de etiquetas.
3. El código entrenará la red neuronal y mostrará los resultados de validación.

## Métricas de Evaluación
Se utilizan las siguientes métricas para evaluar el desempeño de los modelos:
- **Precisión (Accuracy)**
- **Precisión por clase (Precision)**
- **Sensibilidad (Recall)**
- **Puntuación F1 (F1-Score)**

Los resultados se imprimen en la consola al finalizar la ejecución.

## Autores
Este proyecto fue desarrollado para aplicar técnicas de aprendizaje automático y redes neuronales en la clasificación de datos.

