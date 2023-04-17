# **Proyecto NLP**

-----
### Exte proyecto va a consistir en el análisis, exploración de datos y creación de un modelo de machine learning para predecir la puntuacion de toxicidad en un comentario de reddit. Dentor de todo el proyecto se ha usado la librería de PySpark que permite el procesamiento distribuido de datos haciendo más eficientes todas las operaciones de analisis y machine learning que se realizan. Este es un proyecto de NLP por lo que se han usado las herramientas y librerías correspondientes con el lenguaje de Python.



-----

### Organización de carpetas: 

* scr/
    * data/: Contiene los archivos usados en el proyecto.
    
    * Images /: Contiene imágenes usadas en este archivo Markdown.

    * notebooks/: son archivos jupyter notebook usados en todo el proceso.

------

### Fuente: [Kaggle](https://www.kaggle.com/datasets/estebanmarcelloni/ruddit-papers-comments-scored)

------

### En este proyecto de pueden apreciar conocimientos en:

* Python
* Spark
* Big Data
* NLP
* Supervised Learning
* Regression Models
* Functional Programming

------

## **Importación de los datos**

#### Antes de nada abrimos los archivos correspondientes con pyspark.

```python
spark = SparkSession.builder.appName("ProyectoNLPSpark").getOrCreate()
data = spark.read.csv('../data/ruddit_comments_score.csv', header=True, inferSchema=True, sep = ",", multiLine=True)
data = data.withColumnRenamed("comment_id", "ID").withColumnRenamed("body", "Comentario").withColumnRenamed("score", "Puntuacion")
data.show()
```

#### Estos son los datos con los que vamos a trabajar.
![Image](src/Images/Image1.PNG)
#### Este es el esquema del dataframe.
![Image](src/Images/Image2.PNG)

## **Limpieza y transformación de datos**
#### Realizamos una limpieza sobre los datos para eliminar los registros invalidos y vamlores nulos.
```python
data = data.withColumn("Puntuacion", data["Puntuacion"].cast("float"))
data = data.dropna()
data = data.filter(data["Comentario"] != "[deleted]")
data = data.rdd.map(lambda x: (x[0],x[1][1:] if (x[1][0] == '"') else x[1],x[2])).toDF()
data = data.rdd.map(lambda x: (x[0],x[1][:-1] if (x[1][-1] == '"') else x[1],x[2])).toDF()
data = data.rdd.map(lambda x: (x[0],x[1][1:] if (x[1][0] == '>') else x[1],x[2])).toDF()
data = data.rdd.map(lambda x: (x[0],x[1][1:] if (x[1][0] == ' ') else x[1],x[2])).toDF()
data = data.rdd.map(lambda x: (x[0],x[1][:-1] if (x[1][-1] == ' ') else x[1],x[2])).toDF()
data = data.rdd.map(lambda x: (x[0],x[1][:-1] if (x[1][-1] == '.') else x[1],x[2])).toDF()
data = data.rdd.map(lambda x: (x[0],x[1].replace('""', '"'),x[2])).toDF()
data = data.rdd.map(lambda x: (x[0],x[1].replace('\n', ''),x[2])).toDF()
data = data.rdd.map(lambda x: (x[0],x[1].replace('\t', ''),x[2])).toDF()
data = data.withColumnRenamed("_1", "ID").withColumnRenamed("_2", "Comentario").withColumnRenamed("_3", "Puntuacion")
```
#### Así es como nos quedan nuestros datos luego de realizar la limpieza de los valores nulos, los comentarios eliminados y la eliminación de caracteres que no aportan información. Esto para que nuestro modelo aprenda mejor y con datos más útiles.
![Image](src/Images/Image2.PNG)
