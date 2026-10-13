# Proyecto MiBici - Equipo 04
## Conclusiones de los modelos P1, P2 y P3

### P1 - Predicción de demanda

Se comparó un modelo baseline de persistencia contra Random Forest.

Resultados en prueba:
- Baseline: MAE = 0.9846; RMSE = 1.9374.
- Random Forest: MAE = 0.8086; RMSE = 1.4797.

Random Forest redujo el MAE aproximadamente 17.9 % y el RMSE
23.6 % respecto al baseline. Por tanto, obtuvo el mejor desempeño
predictivo entre los modelos evaluados.

### P2 - Clasificación de riesgo

Se evaluaron tres alternativas para identificar riesgos de vaciado
y saturación de estaciones.

El árbol de decisión balanceado obtuvo en prueba:
- Recall de vaciado: 0.7819.
- Recall de saturación: 0.6186.
- Balanced accuracy: 0.7155.
- F1 macro: 0.2925.

El baseline de clase mayoritaria obtuvo accuracy de 0.9979,
pero balanced accuracy de 0.3333, lo que evidencia las
limitaciones de utilizar accuracy con clases desbalanceadas.

El árbol con umbral 0.90 obtuvo accuracy de 0.9872, pero
recall de vaciado de 0.2215 y de saturación de 0.1982.

El árbol balanceado es la alternativa más adecuada de las
evaluadas cuando se prioriza detectar eventos de riesgo,
aunque requiere mejoras adicionales.

### P3 - Agrupamiento de estaciones

Se aplicó K-Means para identificar perfiles de actividad
de las estaciones de MiBici.

Se evaluaron valores de k entre 2 y 6. El mejor Silhouette
Score fue 0.4265, correspondiente a k=2.

Los grupos obtenidos fueron:
- Grupo 0: 50 estaciones de alta actividad.
- Grupo 1: 376 estaciones de menor actividad.

La agrupación permite identificar diferentes perfiles de
uso y puede apoyar decisiones operativas sobre las estaciones.

### Seguimiento de experimentos

Los resultados de P1, P2 y P3 están integrados en el
experimento MLflow "mibici-equipo04", con 11 ejecuciones
finalizadas y sus métricas registradas.

El mapa de agrupamiento de P3 está disponible como
artefacto PNG en la ejecución "P3_KMeans_final_mapa".
