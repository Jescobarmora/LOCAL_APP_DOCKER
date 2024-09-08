# LOCAL_APP_DOCKER

Este repositorio contiene la configuración de un entorno Docker para ejecutar una aplicación local desarrollada en Python con Streamlit. La aplicación está diseñada para predecir el estado de un estudiante basado en un conjunto de características específicas, utilizando un modelo de Machine Learning `LightGBM`.

## Características

- **Dockerizado**: La aplicación está completamente encapsulada en un contenedor Docker, lo que facilita su despliegue y ejecución en cualquier entorno sin necesidad de configurar dependencias localmente.
- **Streamlit**: La interfaz de usuario está desarrollada con Streamlit, proporcionando una forma interactiva y fácil de usar para hacer predicciones.
- **Modelo de Machine Learning**: Utiliza un modelo `LightGBM` previamente entrenado para realizar predicciones en tiempo real.

## Instrucciones para Ejecutar el Docker

## Descargar la Imagen desde Docker Hub

Puedes descargar la imagen preconstruida desde Docker Hub:

### Paso 1: Descargar la Imagen

Descarga la imagen desde Docker Hub utilizando el siguiente comando:

```bash
docker pull jescobarmora/streamlit-app:latest
```

### Paso 2: Ejecuta el contenedor

```bash
docker run -p 8501:8501 jescobarmora/streamlit-app:latest
```
