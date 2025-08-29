from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask API!"})

@app.route('/procesar-factura', methods=['POST'])
def procesar_factura():
    from flask import request
    data = request.get_json()
    nombre = data.get('nombre') if data else None
    if not nombre:
        return jsonify({"status": "error", "message": "Falta el nombre en la solicitud"}), 400
    return jsonify({"status": "successful", "message": f"Factura procesada correctamente para {nombre}"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 