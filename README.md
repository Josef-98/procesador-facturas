# Proyecto de Josef y sus amiguillos

Aplicacion para procesar facturas de clientes utilizando una red neuronal super guay


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