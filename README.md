![Logo](https://blog.soyhenry.com/content/images/2021/02/HEADER-BLOG-NEGRO-01.jpg)

# DATA SCIENCE - PROYECTO INDIVIDUAL Nº1
# Machine Learning Operations (MLOps)

Este proyecto abarca una serie de pasos para desarrollar un proceso de **`Data Engineering`** sobre un dataset de películas, para posteriormente disponibilizar una serie de endpoints y un modelo de recomendación de películas utilizando **`Machine Learning`**, a través de una **`API`**.

![Logo](https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png)

## Contexto

Se plantea desde los depártamentos de Machine Learning y Analytics la necesidad de contar con los datos en una API para poder ser consumidos.
Por otro lado existe la necesidad de poder realizar las consultas al modelo de recomendación para lo cual resulta necesario hacer un deploy de la API.

## Dataset

El [dataset](https://github.com/fedeandresg/1-proyecto-individual-MLOps/blob/main/movies_dataset.csv) en cuestión posee información acerca de películas y distintos atributos de las mismas. El mismo cuenta con 45466 filas (representando cada fila una película) y 24 columnas (atributos de cada título).

## Data Engineering

Para el trabajo de **`Data Engineering`** se procedió a efectuar una serie de transformaciones solicitadas sobre los datos, dentro de las cuales se encuentran:

- Algunos campos, como belongs_to_collection, production_companies y otros están anidados, deberás desanidarlos para poder y unirlos al dataset de nuevo hacer alguna de las consultas de la API.

- Los valores nulos de los campos revenue, budget deben ser rellenados por el número 0.

- Los valores nulos del campo release date deben eliminarse.

- De haber fechas, deberán tener el formato AAAA-mm-dd, además deberás crear la columna release_year donde extraerás el año de la fecha de estreno.

- Crear la columna con el retorno de inversión, llamada return con los campos revenue y budget, dividiendo estas dos últimas revenue / budget, cuando no hay datos disponibles para calcularlo, deberá tomar el valor 0.

- Eliminar las columnas que no serán utilizadas, video,imdb_id,adult,original_title,vote_count,poster_path y homepage.

Se pueden visualizar las transformaciones y los análisis realizados en el siguiente
[archivo](https://github.com/fedeandresg/1-proyecto-individual-MLOps/blob/main/movies_analysis_ETL.ipynb)

## API

Se solicitó efectuar la disponibilización de los siguientes endpoints a través del Framework **`FastAPI`**:

- def peliculas_mes(mes): '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes (nombre del mes, en str, ejemplo 'enero') historicamente''' return {'mes':mes, 'cantidad':respuesta}

- def peliculas_dia(dia): '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrenaron ese dia (de la semana, en str, ejemplo 'lunes') historicamente''' return {'dia':dia, 'cantidad':respuesta}

- def franquicia(franquicia): '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio''' return {'franquicia':franquicia, 'cantidad':respuesta, 'ganancia_total':respuesta, 'ganancia_promedio':respuesta}

- def peliculas_pais(pais): '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo''' return {'pais':pais, 'cantidad':respuesta}

- def productoras(productora): '''Ingresas la productora, retornando la ganancia total y la cantidad de peliculas que produjeron''' return {'productora':productora, 'ganancia_total':respuesta, 'cantidad':respuesta}

- def retorno(pelicula): '''Ingresas la pelicula, retornando la inversion, la ganancia, el retorno y el año en el que se lanzo''' return {'pelicula':pelicula, 'inversion':respuesta, 'ganacia':respuesta,'retorno':respuesta, 'anio':respuesta}

El código para correr la API dentro de FastAPI se puede visualizar [aquí](https://github.com/fedeandresg/1-proyecto-individual-MLOps/blob/main/main.py) 

## Análisis exploratorio de datos

A los efectos de poder entender los datos presentados, se realizaron una serie de análisis y estudios sobre las variables del dataset a los efectos de poder encontrar relaciones entre los datos y comprender la relevancia de los mismos.
Dentro de los análisis efectuados se encuentran distribuciones de frecuencias de las variables numéricas, identificación de variables categóricas y sus valores, correlación entre variables, análisis temporales y por categoría.

Se destaca que se efectuaron algunas transformaciones diferentes a las realizadas para la sección de Engineering.

Se pueden visualizar las transformaciones y los análisis realizados en el siguiente
[archivo](https://github.com/fedeandresg/1-proyecto-individual-MLOps/blob/main/movies_EDA_ML.ipynb)

## Modelo de recomendación - Machine Learning

Para el modelo de recomendación de películas **`Machine Learning`** se utilizó como criterio filtrar el dataset para las películas estrenadas a partir de 1970 `release_year` (por mostrar una gran crecimiento a partir de dicho año) y posteriormente filtros por puntajes obtenidos por las mismas `vote_average` para concluir seleccionando 10.000 títulos.

Se trabajó uniendo los textos de las columnas `overview`, `genres` y `production_companies` para concatenarlos en un solo campo de modo que el algoritmo pueda trabajar sobre el mismo `combined_features`.
Una vez con el dataframe listo se procedió a construir 2 modelos para efectuar recomendaciones de películas (uno basado en la similitud del coseno y otro basado en el algoritmo de K-Vecinos).

Se pueden visualizar los códigos realizados en el siguiente
[archivo](https://github.com/fedeandresg/1-proyecto-individual-MLOps/blob/main/movies_EDA_ML.ipynb)

## Deployment

Para el deploy de la API, se utilizó la plataforma **`Render`**.
Los datos están listos para ser consumidos y consultados a partir del siguiente link

[Link al Deployment](https://deploy-proyecto-1-henry.onrender.com/docs#/)

## Video 

Para consultar sobre los pasos del proceso y una explicación más profunda es posible acceder al siguiente [enlace](https://www.youtube.com/watch?v=7rNNCXf-Bh4)
