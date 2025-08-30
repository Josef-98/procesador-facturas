# Proyecto de Josef y sus amiguillos

Aplicacion para procesar facturas de clientes utilizando una red neuronal super guay


# comando para iniciar la api
python app.py

# comandos de prueba de la api
curl -X GET http://localhost:5000/
curl -X POST -F "file=@facturaPrueba.jpg" http://localhost:5000/procesar-factura

# Recomendacion base de datos: 
con IA pidele que te guie creando una base sqllite con una sola tabla facturas y las columnas que quieren

# Lo q dice david q se necesita para subirlo a la web:
red neuronal unificada
Código Red Neuronal Facturas Extra.ipynb
Código Red Neuronal.ipynb
best.pt


# Simple Flask API

A simple Flask API with test endpoints.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the API

1. Start the Flask application:
```bash
python app.py
```

2. The API will be available at `http://localhost:5000`

## Available Endpoints

- **GET /** - Home endpoint with welcome message
- **GET /test** - Test endpoint that returns "successful" status
- **GET /health** - Health check endpoint

## Test the API

You can test the endpoints using:

- Browser: Navigate to `http://localhost:5000/test`
- curl: `curl http://localhost:5000/test`
- Postman or any API testing tool

The test endpoint will return:
```json
{
  "status": "successful",
  "message": "Test endpoint working correctly"
}
``` 
