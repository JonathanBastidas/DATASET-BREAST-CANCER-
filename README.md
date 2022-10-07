# MODELO DETECCIÓN DE CÁNCER DE MAMA – JUPYTER
La finalidad del modelo es detectar a tiempo el cáncer de mama de la manera mas temprana con la finalidad de promover los tratamientos clínicos oportunos.
Los dataset y la descripción se obtuvieron de la siguiente dirección: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Descargamos los dataset y empezamos con la exploración de cada uno para determinar las características.
El código fue realizado con IA en Anaconda en Jupyter.

# DATASET 1 breast-cancer.csv (wdbc.data)

#Importamos las librerías que usaremos en la construcción del modelo para reducir errores en la ejecución.

#Verificar los campos del dataset y procedemos a renombrar las columnas con los nombres que se obtuvieron en proceso de investigación.

#Realizamos un conteo de numero de filas y columnas en el dataset.  Existen 568 pacientes en el dataset, y 32 columnas que son las características y datos de cada paciente.

#Realizamos la búsqueda de columnas vacías para limpiar el dataset. 

#Haremos el conteo del tipo de cáncer Maligno como Benigno, de los pacientes y poder determinar esta estadística y lo graficamos para una mejor visualización.

#Al confirmar que Diagnosis es el único dato de diferente tipo, y Diagnosis siendo de carácter binario, es decir, Maligno o Benigno, podemos sustituirlo por 1 y 0 respectivamente; realizamos el cambio y lo visualizamos:

#Realizamos un pair plot con Diagnosis, donde se utiliza una variable para compararlas con los valores de las demás variables.

#Obtenemos la correlación de las columnas.

#Para visualizar mejor esta interacción entre columnas podemos utilizar un mapa de calor para la comparación.
Preparación Data Set:

#Una vez analizado y limpio el dataset procedemos a prepararlo para el modelo separando en dos en variables independiente y dependiente.

#Preparamos el dataset, un 75% de este para entrenamiento y el restante 25% para pruebas.

#Creamos diferentes funciones para almacenar diferentes modelos y clasificarlos. Estos serían los modelos que nos brindarían la capacidad de comprender si un paciente tiene o no cáncer.

#Creamos los modelos y comprobamos sus respectivas precisiones, característica importante para conocer la calidad del mismo.

#Construimos la matriz de confusión de cada uno de los modelos para conocer aún más sobre la calidad de estos.

#La precisión y matriz de confusión no son los únicos factores que determinan la calidad de un modelo por lo que buscamos algunas características extra que nos ayuden a elegir el mejor de estos.

# A partir de los datos recopilados el mejor modelo al que podríamos optar es Logistic Regression, ya que las métricas y precisión obtenidas anteriormente nos hace creer esto, con una precisión de casi 97%.

# DATASET 2 breast-cancer-wisconsin.data

# Cargamos las librerias que vamos a utilizar.
#Renombramos columnas según lo que indica el archivo de names.
#Exploración dataset.
#Clase a predecir:
Cuando dice 2 significa que el tumor es benigno (no tiene cáncer). En cambio cuando dice 4, significa maligno ( si tiene cáncer).
Le asignamos 0 y 1 a la clase. 0 = benigno 1 = maligno
Esto lo hacemos para posteriormente entrenar correctamente el modelo

#¿Cuantos casos son de tumores benignos y cuantos malignos? (tabla y countplot)
65% de los pacientes presentan tumores benignos
El 35% restante presenta tumor maligno.

#¿Las variables tienen mucha variación? ¿cuales son sus percentiles?

# Preparación de datos.
Son 54 en total
Pero los voy a dejar, podría ser que la misma persona se haya hecho estudios otra vez, pero tendría distintos valores.

#Entonces, veamos si alguno de esos duplicados, también este duplicado en las demás variables.
#Efectivamente, hay 8 registros que están completamente duplicados. Esos los vamos a eliminar.
#Eliminemos la variable ID Sample code number, ya que no la vamos a utilizar en el modelo.
#Veamos que sucede con la variable Bare Nuclei.

#Hay 16 registros en donde la variable llega como '?'
Según la documentación oficial (en ingles):
8. Missing attribute values: 16
   There are 16 instances in Groups 1 to 6 that contain a single missing 
   (i.e., unavailable) attribute value, now denoted by "?".  
Un posible enfoque sería reemplazar esos '?' por la media (promedio) de todo el set de datos para esa variable.

# Correlaciones
¿Cuales son las variables que más influyen a detectar si tiene cancer o no?
Vamos a usar la función corr_pair de funpymodeling.

#Vemos rápidamente las correlaciones.
R2 indica si hay correlación. 0= no hay correlacion 1= mucha correlacion
R tambien indica si hay correlación, y si es positiva o negativa. -1=negativa 0=no hay correlacion 1=positiva

Estas parecen ser las 5 variables que más ayudan a predecir la clase: (en este orden)
1. Uniformity of Cell Shape
1. Uniformity of Cell Size
1. Bare Nuclei
1. Bland Chromatin
1. Clump Thickness

¿Hay variables que tienen mucha correlación entre si? Esto indican que podemos prescindir de alguna<br>
Por ejemplo, sabiendo la fecha de nacimiento, podríamos saber la edad de una persona. Entonces la variable edad es prescindible
Veámoslo con el estadístico Pearson.

#Parece que hay correlacion entre Uniformity of Cell Shape y Uniformity of Cell Size (0.906814). 
Es decir, sabiendo una, es posible predecir la otra.
Si consideramos 0.90 como mínimo para decir que esta altamente correlacionada, esas dos serían las unicas (algunas dan 1 porque se comparan con si mismas).
Nota: si bien vemos que estan muy correlacionadas, en este caso dejamos ambas variables ya que tenemos pocas en nuestro set de datos.

# Graficamente: Cuanto más azul el cuadro, menos relación hay.
En contraste, en los focos rojos es donde se encuentran las correlaciones.

Graficamente: Cuanto más azul el cuadro, menos relación hay.
En contraste, en los focos rojos es donde se encuentran las correlaciones.	

#En casos que no hay cáncer, la mayoria tiene un Bare Nuclei de 1.
Sin embargo, en los que si hay cáncer, vemos que muchos tienen un Bare Nuclei de 10
Es decir, si el Bare nuclei es de 10, es probable que la persona tenga cancer.

#En los tumores benignos, la mayoria tiene un Uniformity of Cell Shape de 1.
Sin embargo los malignos, vemos que tienen un Uniformity of Cell Shape de 10
Es decir, si el Uniformity of Cell Shape es de 10, es probable que la persona tenga cancer.

# Modelo de predicción.
Vamos a usar un modelo de regresión logística. 
Este modelo nos va a predecir, por cada registro, la probabilidad de que la persona tenga cáncer
1. Primero que todo separamos nuestras variables de entrada, de nuestras variables de salida
1. Luego escalamos los datos de entrada (este es requisito del modelo)
1. Una vez tenemos nuestros datos de entrada ya escalados, separamos nuestros datos. 
Por un lado vamos a tener los datos de training (80% de los datos) y por el otro, datos de testing (20%). 
Entrenamos nuestro modelo con los datos de training, y luego eso se prueba con los datos de test.

#Entrenamos nuestro modelo con los datos de training
básicamente le decimos al modelo que con las variables de entrada 'x', la salida es 'y'

#¿Qué devuelve el modelo? Devuelve la probabilidad de que sí tenga cáncer. 
Sería la segunda columna del siguiente array.

#Veamos la matriz de confusión para training, con un punto de corte de 0.5
Un punto de corte de 0.5 quiere decir que si la probabilidad de que tenga cáncer es mayor a 50% el modelo le asigna que SI tiene cáncer. 
En cambio, si es menor a 50% el modelo declara que NO tiene cáncer

#No está mal. Pero pensemos que quiere decir esta matriz de confusión.
Recordemos que 0 = NO TIENE CANCER / 1= SI TIENE CANCER.	

# Testing:
#Veamos la misma matriz, pero con datos de Testing (estos son los datos que el modelo desconoce)

# Curva ROC.
También responde muy bien con datos que no conoce 97%.

