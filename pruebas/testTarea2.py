import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Student_Performance.csv')
df = pd.get_dummies(df, drop_first=True) # Convierte las variables categóricas del DataFrame en variables dummy (o variables indicadoras) ('Extracurricular Activities')

print('-----------------------------------------------------')
# Definir la variable objetivo y las características
X = df.drop('Performance Index', axis=1) # Caracteristicas (Todas las columnas excepto performance index)
y = df['Performance Index'] # Variable objetivo (Performance index)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,random_state=450) # Dividir los datos en conjunto de entrenamiento y prueba
model = LinearRegression() # Crear instancia del modelo de regresion lineal
model.fit(X_train, y_train) # Entrenar instancia del modelo de regresion lineal
y_pred = model.predict(X_test) # Predecir los valores para el conjunto de prueba

# Calcular métricas de rendimiento
mse = mean_squared_error(y_test, y_pred) #Calcular el Error Cuadrático Medio
r2 = r2_score(y_test, y_pred) #Calcular el Coeficiente de Determinación
print(f"Error cuadratico medio: {mse}")
print(f"Coeficiente de determinacion: {r2}")
coeficientes = pd.DataFrame(model.coef_, X.columns, columns=['Coeficientes']) # Mostrar los coeficientes del modelo
print(coeficientes)
print('-----------------------------------------------------')

# Gráfico de valores reales vs. predichos
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Real')
plt.ylabel('Predicción')
plt.title('Valores Reales vs. Predichos')
plt.show()