import requests
import pandas as pd

# URL base de la API
url = "https://data.seattle.gov/resource/teqw-tu6e.json"

# Inicializar variables para la paginación
limit = 1000  # Número de registros por solicitud
offset = 0    # Desplazamiento inicial
all_data = [] # Lista para almacenar todos los datos

while True:
    # Definir parámetros de la solicitud
    params = {
        "$limit": limit,
        "$offset": offset
    }
    
    # Realizar la solicitud GET a la API
    response = requests.get(url, params=params)
    
    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        data = response.json()
        if not data:
            # Si no se reciben más datos, salir del bucle
            break
        all_data.extend(data)
        offset += limit  # Incrementar el desplazamiento
    else:
        print(f"Error en la solicitud: Código de estado {response.status_code}")
        break

# Convertir la lista de datos a un DataFrame de pandas
df = pd.DataFrame(all_data)

# Opcional: Guardar el DataFrame en un archivo CSV
df.to_csv("Data/datos_seattle.csv", index=False)
print(f"Datos guardados en 'datos_seattle.csv' con {len(df)} registros.")
