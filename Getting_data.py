import requests
import pandas as pd

# URL de la API (en este caso, datos de Seattle)
url = "https://data.seattle.gov/resource/teqw-tu6e.json"

try:
    # Hacer la solicitud GET a la API
    response = requests.get(url)

    # Verificar si la solicitud fue exitosa (código 200)
    if response.status_code == 200:
        # Convertir la respuesta a formato JSON
        data = response.json()

        # Convertir los datos JSON a un DataFrame
        df = pd.DataFrame(data)

        # Imprimir el DataFrame
        print("DataFrame creado:")

        # Opcional: Guardar el DataFrame en un archivo CSV
        df.to_csv("Data/datos_seattle.csv", index=False)
        print("Datos guardados en 'datos_seattle.csv'")

    else:
        print(f"Error en la solicitud: Código de estado {response.status_code}")


except requests.exceptions.RequestException as e:
    print(f"Ocurrió un error: {e}")

df.head()