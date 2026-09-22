# Paralelización de Multiplicación Matricial 

Instituto Politécnico Nacional, Unidad Profesional Interdisciplinaria de Ingeniería Campus Tlaxcala (UPIIT).

Este proyecto se centra en el diseño e implementación de un algoritmo de multiplicación de matrices masivas en un entorno de cómputo paralelo. El objetivo principal es utilizar este motor matemático para acelerar el entrenamiento de un modelo de regresión lineal optimizado mediante el método del descenso del gradiente.

## Equipo de Desarrollo
* Alejandro Esponda Meza
* Arturo Salazar Soto
* Carlos Antonio López Prieto

## Definición del Problema
En el entrenamiento de modelos de aprendizaje computacional con bases de datos masivas, el cálculo matemático subyacente representa un cuello de botella crítico. Específicamente, en la regresión lineal con descenso del gradiente, el modelo requiere multiplicar matrices de gran escala de forma iterativa para ajustar sus parámetros. 

Al ejecutar estas operaciones de manera secuencial, el procesador se satura y los tiempos de ejecución crecen de manera insostenible, impidiendo el escalamiento adecuado del procesamiento de datos en arquitecturas modernas.

## Propuesta del Proyecto
El proyecto propone mitigar este problema mediante la transición de un paradigma de ejecución secuencial a uno paralelo. Se desarrollará un algoritmo que distribuya la carga de las operaciones matriciales entre múltiples unidades de procesamiento de forma simultánea. Posteriormente, este componente será acoplado como el núcleo matemático del algoritmo de regresión lineal para validar la reducción de los tiempos de cálculo.

## Metodología

1. Generación de matrices: Construcción de cadenas de matrices de diferentes dimensiones, asegurando el criterio matemático fundamental (el número de columnas de la primera matriz debe ser igual al número de filas de la segunda).
2. Implementación secuencial: Desarrollo de la multiplicación tradicional mediante bucles para establecer un punto de referencia (baseline) y validar la exactitud de los resultados.
3. Implementación en paralelo: Diseño de la arquitectura de distribución de tareas (gestión de hilos y memoria) y codificación del algoritmo concurrente.
4. Validación de resultados: Pruebas comparativas para garantizar que la implementación paralela genere exactamente los mismos resultados que el método secuencial.
5. Evaluación experimental: Medición del rendimiento utilizando métricas de eficiencia y aceleración (speedup) frente a diferentes tamaños de matrices.
6. Implementación de la regresión lineal: Integración de la multiplicación matricial paralelizada dentro del algoritmo final de aprendizaje automático.

## Resultados Esperados
Se espera comprobar empíricamente que la paralelización de operaciones algebraicas reduce drásticamente el tiempo total de ejecución. El proyecto busca lograr una aceleración notable frente a la versión secuencial, validando la viabilidad del cómputo paralelo como una herramienta indispensable para optimizar algoritmos de aprendizaje computacional.