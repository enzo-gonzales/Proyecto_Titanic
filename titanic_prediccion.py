import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split




# -------- Cargar y explorar el dataset --------
# Cargar el dataset del Titanic
df_titanic = pd.read_csv('titanic.csv')


# Verificar si hay valores duplicados
duplicados = df_titanic.duplicated().sum()
print(f"Cantidad de filas duplicadas: {duplicados}")


# Eliminar filas duplicadas si existen
if duplicados > 0:
    df_titanic = df_titanic.drop_duplicates()
    print("Duplicados eliminados.")


# Verificar si hay valores nulos en cada columna
print("Valores nulos por columna antes de preprocesar:\n", df_titanic.isnull().sum())




# -------- Preprocesamiento de datos --------
# Eliminar columnas irrelevantes para el análisis o predicción
df_titanic = df_titanic.drop(columns=["Cabin", "Fare", "Ticket", "Name"])



# Separar las variables predictoras (X) de la variable objetivo (y)
X = df_titanic.drop("Survived", axis=1)
y = df_titanic["Survived"]




# -------- Conversión de columnas categóricas a numéricas --------
# Identificar las columnas categóricas (variables no numéricas)
columnas_categoricas = X.select_dtypes(include=["object"]).columns


# Utilizar OrdinalEncoder para convertir las variables categóricas en numéricas
ordinal_encoder = OrdinalEncoder()
X[columnas_categoricas] = ordinal_encoder.fit_transform(X[columnas_categoricas])




# -------- Rellenar valores nulos --------
# Rellenar los valores nulos utilizando SimpleImputer
imputer = SimpleImputer()
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


# Verificar que no queden valores nulos después de la imputación
print("Valores nulos después de imputar:\n", X_imputed.isnull().sum())




# -------- División del dataset en entrenamiento y prueba --------
# Dividir el dataset en conjuntos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)




# -------- Entrenamiento del modelo Random Forest --------

# Crear y entrenar el modelo Random Forest con los datos de entrenamiento
modelo_random_forest = RandomForestClassifier()
modelo_random_forest.fit(X_train, y_train)




# -------- Predicciones --------
# Realizar predicciones con los datos de prueba
predicciones_test = modelo_random_forest.predict(X_test)


# Imprimir las predicciones para el conjunto de prueba
print("Predicciones en el conjunto de prueba:")
print(predicciones_test)




# -------- Evaluar el modelo --------
# Calcular y mostrar el rendimiento del modelo en los datos de prueba
rendimiento_test = modelo_random_forest.score(X_test, y_test)
print(f"Rendimiento del modelo (accuracy): {rendimiento_test:.2f}")




# -------- Predicción para un nuevo pasajero --------
# Crear una función para predecir si un pasajero sobreviviría según sus datos
def predecir_sobrevivencia(datos_pasajero):
    # Convertir los datos del pasajero a DataFrame
    pasajero_df = pd.DataFrame([datos_pasajero], columns=X.columns)
    
    # Transformar las columnas categóricas del nuevo pasajero
    pasajero_df[columnas_categoricas] = ordinal_encoder.transform(pasajero_df[columnas_categoricas])
    
    # Rellenar los valores nulos en caso de que existan
    pasajero_df_imputed = pd.DataFrame(imputer.transform(pasajero_df), columns=X.columns)
    
    # Realizar la predicción
    prediccion = modelo_random_forest.predict(pasajero_df_imputed)
    
    # Imprimir el resultado de la predicción
    if prediccion == 1:
        print("El pasajero sobreviviría.")
    else:
        print("El pasajero no sobreviviría.")




# Ejemplo de uso de la función para predecir si un nuevo pasajero sobreviviría
nuevo_pasajero = {
    'Pclass': 3,     # Clase del pasajero (1, 2, 3)
    'Sex': 'female', # Sexo del pasajero ('male', 'female')
    'Age': 29,       # Edad del pasajero
    'SibSp': 1,      # Número de hermanos/esposos a bordo
    'Parch': 0,      # Número de padres/hijos a bordo
    'Embarked': 'S'  # Puerto de embarque ('C', 'Q', 'S')
}



predecir_sobrevivencia(nuevo_pasajero)