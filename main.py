# Importamos las librerías
import pandas as pd 
import numpy as np
import sklearn
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer

# Indicamos título y descripción de la API
app = FastAPI(title='PROYECTO INDIVIDUAL Nº1 -Machine Learning Operations (MLOps)',
            description='API de datos y recomendaciones de películas')

# Datasets
df = pd.read_csv('movies_final.csv')
df1 = pd.read_csv('movies_ml.csv')


# Función para reconocer el servidor local

@app.get('/')
async def index():
    return {'Hola! Bienvenido a la API de recomedación. Por favor dirigite a /docs'}

@app.get('/about/')
async def about():
    return {'PROYECTO INDIVIDUAL Nº1 -Machine Learning Operations (MLOps)'}


# Función de películas por mes

@app.get('/peliculas_mes/({mes})')
def peliculas_mes(mes:str):
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes 
    (nombre del mes, en str, ejemplo 'enero') historicamente
    ''' 
    mes = mes.lower()
    meses = {
    'enero': 1,
    'febrero': 2,
    'marzo': 3,
    'abril': 4,
    'mayo': 5,
    'junio': 6,
    'julio': 7,
    'agosto': 8,
    'septiembre': 9,
    'octubre': 10,
    'noviembre': 11,
    'diciembre': 12}

    mes_numero = meses[mes]

    # Convertir la columna "fecha" a un objeto de tipo fecha
    df['release_date'] = pd.to_datetime(df['release_date'])

    # Tratamos la excepciòn
    try:
        month_filtered = df[df['release_date'].dt.month == mes_numero]
    except (ValueError, KeyError, TypeError):
        return None

    # Filtramos valores duplicados del dataframe y calculamos
    month_unique = month_filtered.drop_duplicates(subset='id')
    respuesta = month_unique.shape[0]

    return {'mes':mes, 'cantidad':respuesta}


# Función de películas por día

@app.get('/peliculas_dia/({dia})')
def peliculas_dia(dia:str):
    '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se 
    estrenaron ese dia (de la semana, en str, ejemplo 'lunes') historicamente
    '''
    # Creamos diccionario para normalizar
    days = {
    'lunes': 'Monday',
    'martes': 'Tuesday',
    'miercoles': 'Wednesday',
    'jueves': 'Thursday',
    'viernes': 'Friday',
    'sabado': 'Saturday',
    'domingo': 'Sunday'}

    day = days[dia.lower()]

    # Filtramos los duplicados del dataframe y calculamos
    lista_peliculas_day = df[df['release_date'].dt.day_name() == day].drop_duplicates(subset='id')
    respuesta = lista_peliculas_day.shape[0]

    return {'dia': dia, 'cantidad': respuesta}

# Función de métricas por franquicia

@app.get('/franquicia/({franquicia})')
def franquicia(franquicia:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio
    ''' 
    # Filtramos el dataframe
    lista_peliculas_franquicia = df[(df['collection'] == franquicia)].drop_duplicates(subset='id')

    # Calculamos
    cantidad_peliculas_franq = (lista_peliculas_franquicia).shape[0]
    revenue_franq = lista_peliculas_franquicia['revenue'].sum()
    promedio_franq = revenue_franq/cantidad_peliculas_franq

    return {'franquicia':franquicia, 'cantidad':cantidad_peliculas_franq, 'ganancia_total':revenue_franq, 'ganancia_promedio':promedio_franq}


# Función películas por país

@app.get('/peliculas_pais/({pais})')
def peliculas_pais(pais:str):
    '''
    Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo
    '''
    # Filtramos el dataframe y contamos filas
    movies_filtered = df[(df['country'] == pais)]
    movies_unique = movies_filtered.drop_duplicates(subset='id')    
    respuesta = movies_unique.shape[0]
    
    return {'pais':pais, 'cantidad':respuesta}

# Función métricas por productora

@app.get('/productoras/({productora})')
def productoras(productora:str):
    '''Ingresas la productora, retornando la ganancia total y la cantidad de peliculas que produjeron
    ''' 
    # Filtramos el dataframe
    lista_peliculas_productoras = df[(df['company'] == productora)].drop_duplicates(subset='id')

    # Calculamos
    cantidad_peliculas_prod = (lista_peliculas_productoras).shape[0]
    revenue_prod = lista_peliculas_productoras['revenue'].sum()

    return {'productora':productora, 'ganancia_total':revenue_prod, 'cantidad':cantidad_peliculas_prod}


# Función métricas por película

@app.get('/retorno/({pelicula})')
def retorno(pelicula):
    '''Ingresas la pelicula, retornando la inversion, la ganancia, el retorno y el año en el que se lanzó
    ''' 
    # Filtramos el dataframe y Calculamos
    info_pelicula = df[(df['title'] == pelicula)].drop_duplicates(subset='title')
    pelicula_nombre = info_pelicula['title'].iloc[0]
    inversion_pelicula = str(info_pelicula['budget'].iloc[0])
    ganancia_pelicula = str(info_pelicula['revenue'].iloc[0])
    retorno_pelicula = str(info_pelicula['return'].iloc[0])
    year_pelicula = str(info_pelicula['release_year'].iloc[0])

    return {'pelicula':pelicula_nombre, 'inversion':inversion_pelicula, 'ganacia':ganancia_pelicula,'retorno':retorno_pelicula, 'anio':year_pelicula}


# Función de recomendación

# Aseguramos que los datos de la columna 'overview' sean strings
df1['overview'] = df1['overview'].fillna('').astype('str')

# Aseguramos que los datos de la columna 'genres' sean strings
df1['genres'] = df1['genres'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else '')

# Reemplazar los valores NaN con cadenas vacías en la columna 'production_companies'
df1['production_companies'] = df1['production_companies'].fillna('')

# Convertir la columna 'production_companies' a string si es necesario
df1['production_companies'] = df1['production_companies'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)

# Crear una nueva columna combinando las características de interés
df1['combined_features'] = df1['overview'] + ' ' + df1['genres'] + ' ' + df1['production_companies']

# Convertimos todos los textos a minusculas para evitar duplicados
df1['combined_features'] = df1['combined_features'].str.lower()

# Inicializamos el HashingVectorizer
hash_vectorizer = HashingVectorizer(stop_words='english', n_features=2000)

# Transformamos los datos
hash_matrix = hash_vectorizer.fit_transform(df1['combined_features'])

# Calculamos la similitud del coseno
cosine_sim = cosine_similarity(hash_matrix)

# Creamos un índice con los títulos de las películas
indices = pd.Series(df1.index, index=df1['title']).drop_duplicates()


@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    '''Ingresas un nombre de pelicula y te recomienda 5 similares
    '''
    if titulo not in df1['title'].values:
        return {'message': 'La película no se encuentra en el conjunto de datos de muestra.'}
    else:
        # Obtenemos el índice de la película que coincide con el título
        idx = indices[titulo]

        # Obtenemos las puntuaciones de similitud de todas las películas con la película dada
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Ordenamos las películas en función de las puntuaciones de similitud
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Obtenemos las puntuaciones de las 5 películas más similares
        sim_scores = sim_scores[1:6]

        # Obtenemos los índices de las películas
        movie_indices = [i[0] for i in sim_scores]

        # Devolvemos las 5 películas más similares
        return {'lista recomendada': df1['title'].iloc[movie_indices].tolist()}
