# ML_Ops

Proyecto de integración de modelo de segmentación YOLO de comida. 

# Explicación

El modelo recibe una imagen y devuelve otra con el tipo de comida y dónde está.

# Cómo ejecutar modelo en local

Abrir Git Bash en la carpeta raíz del proyecto (ML_Ops) 

- docker build --no-cache -t ml-api # Hacemos el build

- docker run -p 80:80 -d ml-api # Ajustar puertos si se necesita

Ahora que está en el puerto, podemos ejecutar el comando de predicción con una imagen (se proporciona test.jpg en el proyecto)

- curl -X POST -F "file=@test.jpg" http://localhost:80/predict --output result.jpg

En la misma carpeta aparecerá el archive result.jpg con el resultado

# Cómo entrenar el modelo

Descargar el dataset de este link https://drive.google.com/file/d/1dIDZxIZ4IXdEkCT-7ggwA6hQxkCVqNZA/view?usp=drive_link 

Descomprimir el archivo y ponerlo en la raíz del Proyecto (ML_Ops)

Ejecutar el código preprocess_yolo y train_yolo

- python train/preprocess_yolo.py

- python train/train_yolo.py

Debería aparecer un nuevo modelo Food_seg_model.pt en la raíz del proyecto