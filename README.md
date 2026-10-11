# Redes Neuronales II

**Actividad:** R1-A2-S4 · Redes Neuronales II
**Materia:** 604013/DEEP LEARNING -CONCEPTOS/FU/CAD2202023205/EIAIIPA2026
**Universidad de Cundinamarca** · Especialización en Inteligencia Artificial
**Docente:** Nidia Stella Garcia Roa
**Estudiante:** Yohann Javier Nikolay Salazar Cardenas · 2026
**Repositorio:** https://github.com/Yohann0011/Redes-Neuronales-II

Repositorio con las seis prácticas de *deep learning* desarrolladas en Google Colab: desde un
perceptrón implementado desde cero con NumPy hasta la clasificación multiclase de dígitos MNIST con
TensorFlow + Keras y PyTorch, **sin utilizar redes convolucionales**.

---

## Contenido

| Notebook | Tema | Tecnología | Problema |
|---|---|---|---|
| `01_Perceptron_NumPy.ipynb` | Perceptrón con activación escalón y regla de aprendizaje | NumPy | Manzana vs banano (binaria) |
| `02_Una_Capa_y_Multicapa_NumPy_Backpropagation.ipynb` | Red de una capa y red multicapa: Sigmoide, ReLU y **backpropagation manual** | NumPy | Manzana vs banano (binaria) |
| `03_Frutas_Binario_Keras.ipynb` | Clasificación binaria con framework | TensorFlow + Keras | Manzana vs banano (binaria) |
| `04_Frutas_Multiclase_Keras.ipynb` | Clasificación **multiclase** con Softmax | TensorFlow + Keras | Manzana, banano y naranja (3 clases) |
| `05_MNIST_Keras.ipynb` | Reconocimiento de dígitos, red densa 784 → 128 → 10 | TensorFlow + Keras | MNIST (10 clases) |
| `06_MNIST_PyTorch.ipynb` | Réplica de la arquitectura con ciclo de entrenamiento explícito | PyTorch | MNIST (10 clases) |

Documentación adicional:

- `Documento_Tecnico_Redes_Neuronales_II.pdf` — documento técnico entregable (formato técnico con código resaltado).
- `Documento_Tecnico_Redes_Neuronales_II_APA7.pdf` — **versión en formato APA 7.ª edición**: portada, tabla de contenido, cuerpo con citaciones, tablas y figuras APA, referencias con sangría francesa y apéndices A-F con el código fuente.
  fuente, resultados, análisis y evidencias de ejecución.
- `figuras/` — las 20 figuras generadas por los notebooks.
- `evidencias/` — salidas literales de consola de cada notebook y resumen de la ejecución.

---

## Resultados obtenidos

Valores reales de la ejecución documentada (semilla 42 en todas las prácticas).

| Modelo | Tecnología | Arquitectura | Épocas | Muestras de prueba | Loss | Accuracy |
|---|---|---|---|---|---|---|
| Perceptrón (escalón) | NumPy | 3 → 1 | 2 (converge) | 4 | — | 1.0000 |
| Red de una capa (Sigmoide) | NumPy | 3 → 1 | 2000 | 4 | 0.002713 | 1.0000 |
| Red multicapa (ReLU) | NumPy | 3 → 4 → 1 | 3000 | 4 | 0.011104 | 1.0000 |
| Clasificación binaria | TensorFlow + Keras | 3 → 4 → 1 | 200 | 4 | 0.017033 | 1.0000 |
| Clasificación multiclase | TensorFlow + Keras | 3 → 6 → 3 | 300 | 6 | 0.003006 | 1.0000 |
| **MNIST** | **TensorFlow + Keras** | **784 → 128 → 10** | **5** | **10 000** | **0.089279** | **0.9735** |
| **MNIST** | **PyTorch** | **784 → 128 → 10** | **3** | **10 000** | **0.1061** (entr.) | **0.9690** |

Otros hallazgos relevantes:

- **Desvanecimiento del gradiente:** con la misma semilla y arquitectura, la norma del gradiente en la
  primera capa es **1.158 con ReLU** frente a **0.438 con Sigmoide** (razón 2.65×).
- **Confianza del modelo MNIST:** 0.9764 de probabilidad máxima promedio en los aciertos frente a
  0.6941 en los errores; el modelo «sabe» cuándo duda.
- **Confusiones más frecuentes en MNIST:** 5→3, 2→8, 9→4, 8→3 y 4→9, pares morfológicamente similares.
- **Capacidad vs desempeño:** en 2 épocas, 32 neuronas alcanzan 0.9124; 128 llegan a 0.9311 y 256 a
  0.9328. Duplicar parámetros no duplica el desempeño.

---

## Entorno de ejecución

| Componente | Versión |
|---|---|
| Python | 3.11 |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 (backend TensorFlow) |
| PyTorch | 2.14.1 |
| torchvision | 0.29.1 |

Los notebooks están pensados para **Google Colaboratory**. Se recomienda:
`Entorno de ejecución → Cambiar tipo de entorno de ejecución → Python 3` (o GPU para MNIST) y
`Entorno de ejecución → Ejecutar todo`.

### Nota sobre la Práctica 5 (MNIST con Keras)

El entrenamiento se realiza con un **ciclo explícito por lotes** (`tf.GradientTape` +
`optimizer.apply_gradients()`) en lugar de `modelo.fit()`. El motivo es técnico: `fit()` materializa una
copia `float32` completa del conjunto de entrenamiento (188 MB) que, sumada a la huella base de
TensorFlow (~620 MB), puede provocar que el *kernel* sea terminado por falta de memoria en entornos con
recursos limitados. El ciclo por lotes mantiene el consumo plano (~725 MB) y es matemáticamente idéntico
al que `fit()` ejecuta internamente. La versión de alto nivel equivalente está documentada en el propio
notebook.

---

## Conceptos cubiertos

- Aprendizaje supervisado, *forward propagation*, función de pérdida y *epochs*/*batches*.
- Funciones de activación: **escalón**, **Sigmoide**, **ReLU** y **Softmax**, con sus derivadas.
- **Backpropagation** implementada manualmente (una capa y multicapa) y su equivalente automático en
  `fit()` (Keras) y `error.backward()` (PyTorch).
- Descenso de gradiente, *learning rate*, optimizador Adam e inicialización de He.
- Clasificación **binaria** vs **multiclase**: neuronas de salida, activación y pérdida.
- Métricas: *accuracy*, pérdida, matriz de confusión, precisión y *recall* por clase, análisis de errores.
- Dataset **MNIST** (60 000 / 10 000 imágenes de 28 × 28) con arquitecturas **densas únicamente**.

---

## Reproducir

```bash
# 1. Abrir cualquier notebook en Google Colab (Archivo → Subir notebook)
# 2. Entorno de ejecución → Ejecutar todo
# Las semillas están fijadas en 42 para garantizar resultados reproducibles.
```

En local:

```bash
pip install numpy scikit-learn matplotlib tensorflow-cpu torch torchvision
jupyter notebook
```

---

## Restricciones de la actividad cumplidas

- Desarrollo individual.
- Notebooks desarrollados en Google Colab y ejecutados de extremo a extremo.
- Código fuente funcional incluido y documentado.
- **Sin redes convolucionales** (verificado programáticamente en los notebooks 05 y 06).
- Uso obligatorio de **TensorFlow + Keras** y **PyTorch**.
- Resultados y evidencias de ejecución incluidos.

## Referencias

- Trask, A. (2019). *Grokking Deep Learning*. Manning Publications.
- Gulli, A., Kapoor, A., Pal, S., & Ciraci, S. (2023). *Deep Learning with TensorFlow and Keras* (3rd ed.). Packt Publishing.
- Stevens, E., Antiga, L., & Viehmann, T. (2020). *Deep Learning with PyTorch*. Manning Publications.
- LeCun, Y., Cortes, C., & Burges, C. J. C. (2010). *MNIST handwritten digit database*.
- Google (2026). *Google Colaboratory*. https://colab.research.google.com/
