from flask import Flask, jsonify, request
from pathlib import Path
import os
import tempfile
import json
import csv
import unicodedata
from datetime import datetime

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize neural network components
def init_neural_network():
    """Initialize YOLO models and processing functions"""
    try:
        from ultralytics import YOLO
        from PIL import Image, ImageDraw, ImageFont
        import pytesseract
        import cv2
        import numpy as np
        
        # Check for model files
        models_dir = Path("./models")
        model_general_path = models_dir / "best_general.pt"
        model_extra_path = models_dir / "best_extra.pt"
        
        if model_general_path.exists() and model_extra_path.exists():
            # Load real models
            modelo_general = YOLO(str(model_general_path))
            modelo_especifico = YOLO(str(model_extra_path))
            return True, modelo_general, modelo_especifico
        else:
            # Models not found
            return False, None, None
            
    except ImportError as e:
        print(f"Warning: Missing dependencies for neural network: {e}")
        return False, None, None

# Initialize models at startup
MODELS_AVAILABLE, MODELO_GENERAL, MODELO_ESPECIFICO = init_neural_network()

def normalizar(texto):
    """Normalize text for class matching"""
    return unicodedata.normalize('NFKD', texto.lower()).encode('ascii', 'ignore').decode('utf-8').strip()

def procesar_factura_neural(imagen_path):
    """Process invoice using neural network or mock data"""
    try:
        if MODELS_AVAILABLE:
            # Real processing with YOLO models
            from ultralytics import YOLO
            from PIL import Image, ImageDraw, ImageFont
            import pytesseract
            import cv2
            import numpy as np
            
            # Process with real models
            pred_g = MODELO_GENERAL.predict(str(imagen_path), conf=0.25)
            pred_s = MODELO_ESPECIFICO.predict(str(imagen_path), conf=0.25)
            
            # Extract and process results (simplified version)
            result = extract_invoice_data(imagen_path, pred_g, pred_s)
            
        else:
            # Mock processing when models are not available
            result = create_mock_result(imagen_path)
            
        return {"status": "success", "data": result}
        
    except Exception as e:
        return {"status": "error", "message": f"Error processing invoice: {str(e)}"}

def create_mock_result(imagen_path):
    """Create mock result when models are not available"""
    return {
        "nombre del cliente": [{"confianza": 0.85, "bbox": [100, 150, 300, 180], "texto": "MARTA QUINTANA STROHECKER SAN MARCOS"}],
        "telefono": [{"confianza": 0.78, "bbox": [100, 280, 200, 300], "texto": "954.35.71.07"}],
        "email": [{"confianza": 0.82, "bbox": [400, 280, 600, 300], "texto": "martaquin@yahoo.es"}],
        "num factura": [{"confianza": 0.90, "bbox": [115, 460, 180, 480], "texto": "115.395"}],
        "fecha factura": [{"confianza": 0.88, "bbox": [240, 460, 290, 480], "texto": "20/10/23"}],
        "total factura": [{"confianza": 0.92, "bbox": [750, 1080, 820, 1100], "texto": "77,00"}],
        "processing_mode": "mock",
        "message": "Processed with mock data - YOLO models not available"
    }

def extract_invoice_data(imagen_path, pred_g, pred_s):
    """Extract invoice data from YOLO predictions - Full implementation"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import pytesseract
        import cv2
        import numpy as np
        from collections import defaultdict
        
        # Configuration from notebook
        clases_especiales = {normalizar(c) for c in ["albarán", "total bruto", "base imponible", "forma de pago"]}
        umbral_clase = {
            "albaran": 0.40, "total bruto": 0.35, "base imponible": 0.35, "forma de pago": 0.40,
            "descuento": 0.25, "fecha factura": 0.25, "codigo articulo": 0.25, "Descripción": 0.20
        }
        umbral_default = 0.30
        
        clases_esperadas_raw = {
            "Nombre del cliente", "Primer apellido del cliente", "Segundo apellido del cliente", "Dirección del cliente",
            "Numero de dirección del cliente", "Codigo postal", "Municipio", "Ciudad", "Telefono", "Email",
            "Num factura", "Fecha factura", "Referencia", "Albarán", "Descripción", "Unidades", "Precio",
            "Total linea", "Total bruto", "Descuento", "Portes", "Base imponible", "% IVA", "Importe IVA",
            "Total factura", "Forma de pago", "Codigo articulo"
        }
        clases_esperadas = {normalizar(c) for c in clases_esperadas_raw}
        
        # Combine predictions
        predicciones = []
        
        # General model predictions
        boxes_g = pred_g[0].boxes
        if len(boxes_g) > 0:
            nombres_g = [normalizar(MODELO_GENERAL.names[int(c)]) for c in boxes_g.cls]
            for i, nombre in enumerate(nombres_g):
                conf = float(boxes_g[i].conf[0])
                umbral = umbral_clase.get(nombre, umbral_default)
                if conf >= umbral:
                    predicciones.append((nombre, boxes_g[i]))
        
        # Specialized model predictions
        boxes_s = pred_s[0].boxes
        if len(boxes_s) > 0:
            nombres_s = [normalizar(MODELO_ESPECIFICO.names[int(c)]) for c in boxes_s.cls]
            for i, nombre in enumerate(nombres_s):
                if nombre not in clases_especiales:
                    continue
                conf = float(boxes_s[i].conf[0])
                umbral = umbral_clase.get(nombre, umbral_default) + 0.10
                if conf >= umbral:
                    predicciones.append((nombre, boxes_s[i]))
        
        # Filter duplicates for unique classes
        clases_multivalor = {
            "codigo articulo", "descripcion", "unidades", "precio", "descuento", "total linea"
        }
        agrupadas_temp = defaultdict(list)
        for nombre, box in predicciones:
            agrupadas_temp[nombre].append((nombre, box))
            
        predicciones_filtradas = []
        for clase, elementos in agrupadas_temp.items():
            if clase in clases_multivalor:
                predicciones_filtradas.extend(elementos)
            else:
                mejor = max(elementos, key=lambda x: float(x[1].conf[0]))
                predicciones_filtradas.append(mejor)
        
        # Extract text from detected regions
        texto_extraido = extraer_texto_por_cajas(imagen_path, predicciones_filtradas)
        
        # Build final result structure
        agrupadas = {}
        for nombre, box in predicciones_filtradas:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            texto = texto_extraido.get(nombre, [""])[0] if nombre in texto_extraido else ""
            texto = texto.strip()
            
            # Email correction
            if nombre == "email":
                texto = texto.replace("fgmail.com", "@gmail.com").replace(" ", "")
                if "@" not in texto and "gmail" in texto:
                    texto = texto.replace("gmail", "@gmail")
                if not texto.endswith(".com") and ".com" in texto:
                    texto = texto.split(".com")[0] + ".com"
            
            # Fallbacks for empty fields
            if not texto:
                if nombre in {"unidades", "importe iva", "portes"}:
                    texto = "0,00"
                elif nombre == "numero de direccion del cliente" and nombre not in texto_extraido:
                    texto = "N/D"
            
            if nombre not in agrupadas:
                agrupadas[nombre] = []
            agrupadas[nombre].append({
                "confianza": conf,
                "bbox": [x1, y1, x2, y2],
                "texto": texto
            })
        
        # Create complete result structure
        datos_factura = {}
        for campo in clases_esperadas:
            datos_factura[campo] = agrupadas.get(campo, None)
        
        # Add processing metadata
        datos_factura["processing_mode"] = "neural_network"
        datos_factura["total_predictions"] = len(predicciones_filtradas)
        datos_factura["detected_classes"] = list(agrupadas.keys())
        
        clases_detectadas = set(agrupadas.keys())
        clases_faltantes = clases_esperadas - clases_detectadas
        if clases_faltantes:
            datos_factura["missing_classes"] = list(clases_faltantes)
        
        return datos_factura
        
    except Exception as e:
        print(f"Error in extract_invoice_data: {e}")
        return create_mock_result(imagen_path)

def extraer_texto_por_cajas(imagen_path, predicciones, margen=5):
    """Extract text from detected bounding boxes using OCR"""
    try:
        import pytesseract
        from PIL import Image
        import cv2
        import numpy as np
        
        imagen_pil = Image.open(imagen_path).convert("RGB")
        imagen_cv = np.array(imagen_pil)
        imagen_cv = cv2.cvtColor(imagen_cv, cv2.COLOR_RGB2BGR)
        alto, ancho = imagen_cv.shape[:2]

        resultados = {}

        for nombre, box in predicciones:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Add margin
            x1 = max(0, x1 - margen)
            y1 = max(0, y1 - margen)
            x2 = min(ancho, x2 + margen)
            y2 = min(alto, y2 + margen)

            region = imagen_cv[y1:y2, x1:x2]
            
            # Convert to grayscale and apply adaptive threshold
            region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            region_thresh = cv2.adaptiveThreshold(
                region_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10
            )

            # Convert back to PIL for OCR
            region_pil = Image.fromarray(region_thresh)
            texto = pytesseract.image_to_string(region_pil, lang="spa").strip()

            # Clean text
            texto = "\n".join([line for line in texto.splitlines() if line.strip() != ""])
            texto = texto.replace("ﬁ", "fi").replace("ﬂ", "fl")

            if nombre not in resultados:
                resultados[nombre] = []
            resultados[nombre].append(texto)

        return resultados
        
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return {}

@app.route('/')
def home():
    return jsonify({
        "message": "Invoice Processing API",
        "models_available": MODELS_AVAILABLE,
        "endpoints": {
            "POST /procesar-factura": "Upload image file to process invoice"
        }
    })

@app.route('/procesar-factura', methods=['POST'])
def procesar_factura():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "status": "error", 
                "message": f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Save uploaded file temporarily
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Process the invoice
            result = procesar_factura_neural(filepath)
            
            # Add processing metadata
            result["filename"] = file.filename
            result["processed_at"] = datetime.now().isoformat()
            result["models_available"] = MODELS_AVAILABLE
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Internal server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    print(f"Starting Invoice Processing API...")
    print(f"Neural network models available: {MODELS_AVAILABLE}")
    app.run(debug=True, host='0.0.0.0', port=5000) 