import os
import json
from http.server import SimpleHTTPRequestHandler
import socketserver
from urllib.parse import parse_qs

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ================================
# Carga de modelo y preprocesadores
# ================================

MODEL_PATH = "modelo_rn.keras"
IMP_PATH = "imp.pkl"
SC_PATH = "sc.pkl"

print("Cargando modelo y preprocesadores...")
model = load_model(MODEL_PATH)
imp = joblib.load(IMP_PATH)
sc = joblib.load(SC_PATH)
print("Modelo y preprocesadores cargados correctamente.")

# Columnas que espera el modelo
COLS_KEEP = ["Rooms", "Distance", "Bathroom", "Car", "Landsize"]
print("Columnas esperadas:", COLS_KEEP)


class PredictHandler(SimpleHTTPRequestHandler):

    # ------------ GET -------------
    def do_GET(self):
        # Landing + formulario en "/" y "/predict"
        if self.path in ("/", "/predict"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = """
<html>
<head>
  <title>Modelo de Clasificación de Viviendas</title>
  <style>
    * {
      box-sizing: border-box;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(135deg, #0f172a, #1e293b);
      margin: 0;
      padding: 0;
      color: #0f172a;
    }
    .topbar {
      width: 100%;
      padding: 12px 32px;
      background: rgba(15, 23, 42, 0.9);
      color: #e5e7eb;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .topbar-title {
      font-weight: 600;
      letter-spacing: 0.03em;
    }
    .topbar-badge {
      font-size: 12px;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.25);
      border: 1px solid rgba(148, 163, 184, 0.6);
    }
    .wrapper {
      min-height: calc(100vh - 56px);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 32px 16px 40px;
    }
    .layout {
      max-width: 1000px;
      width: 100%;
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
      gap: 24px;
    }
    .card {
      background: #f9fafb;
      border-radius: 16px;
      box-shadow: 0 18px 45px rgba(15, 23, 42, 0.38);
      padding: 24px 28px 24px;
    }
    .card-right {
      background: #ffffff;
      border-radius: 16px;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.28);
      padding: 24px 24px 20px;
    }
    h1 {
      margin-top: 0;
      font-size: 26px;
      margin-bottom: 8px;
    }
    .subtitle {
      margin-top: 4px;
      margin-bottom: 12px;
      color: #4b5563;
      font-size: 14px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      background: #e5e7eb;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #374151;
      margin-bottom: 12px;
    }
    .pill-dot {
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: #22c55e;
    }
    h2 {
      font-size: 17px;
      margin-bottom: 8px;
      margin-top: 20px;
    }
    p {
      margin-top: 4px;
      margin-bottom: 8px;
      color: #4b5563;
      font-size: 14px;
    }
    ul {
      padding-left: 18px;
      margin-top: 4px;
      color: #374151;
      font-size: 13px;
    }
    li {
      margin-bottom: 4px;
    }
    .taglist {
      margin-top: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .tag {
      font-size: 11px;
      padding: 3px 9px;
      border-radius: 999px;
      background: #e0f2fe;
      color: #0369a1;
    }
    .field {
      margin-bottom: 12px;
    }
    label {
      display: block;
      font-weight: 600;
      margin-bottom: 4px;
      font-size: 13px;
      color: #111827;
    }
    .hint {
      font-size: 11px;
      color: #6b7280;
      margin-top: 2px;
    }
    input[type="number"] {
      width: 100%;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid #d1d5db;
      box-sizing: border-box;
      font-size: 14px;
      background: #f9fafb;
    }
    input[type="number"]:focus {
      outline: none;
      border-color: #2563eb;
      box-shadow: 0 0 0 1px #2563eb33;
      background: #ffffff;
    }
    button {
      margin-top: 10px;
      padding: 10px 18px;
      background: #2563eb;
      color: #fff;
      border: none;
      border-radius: 999px;
      font-size: 14px;
      cursor: pointer;
      font-weight: 500;
    }
    button:hover {
      background: #1d4ed8;
    }
    .api-note {
      margin-top: 12px;
      font-size: 11px;
      color: #6b7280;
    }
    code {
      background: #f3f4f6;
      padding: 2px 4px;
      border-radius: 4px;
      font-size: 11px;
    }
    @media (max-width: 820px) {
      .layout {
        grid-template-columns: minmax(0, 1fr);
      }
      .wrapper {
        padding-top: 16px;
      }
      .card, .card-right {
        padding: 20px 18px;
      }
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-title">Melbourne Housing Classifier API</div>
    <div class="topbar-badge">Demo interna · Neural Network</div>
  </div>

  <div class="wrapper">
    <div class="layout">
      <div class="card">
        <div class="pill">
          <span class="pill-dot"></span>
          MODELO DE CLASIFICACIÓN
        </div>
        <h1>Predicción del segmento de precio de vivienda</h1>
        <p class="subtitle">
          Este servicio clasifica una propiedad en la ciudad de <strong>Melbourne</strong>  como <strong>Barata</strong>, <strong>Media</strong> o <strong>Cara</strong>
          en función de sus características físicas y de ubicación.
        </p>

        <h2>Descripción de los atributos</h2>
        <ul>
          <li><strong>Rooms:</strong> Número de habitaciones que tiene la propiedad.</li>
          <li><strong>Distance:</strong> Distancia de la casa al centro de la ciudad (en kilómetros).</li>
          <li><strong>Bathroom:</strong> Número de baños que tiene la propiedad.</li>
          <li><strong>Car:</strong> Número de espacios para automóviles disponibles.</li>
          <li><strong>Landsize:</strong> Tamaño del terreno de la propiedad (en metros cuadrados).</li>
        </ul>

        <div class="taglist">
          <span class="tag">Red Neuronal</span>
          <span class="tag">Clasificación Multiclase</span>
          <span class="tag">Demo de análisis inmobiliario</span>
        </div>
      </div>

      <div class="card-right">
        <h2>Ingresar datos de la propiedad</h2>
        <p>Completa el formulario y obtén la clase de precio estimada.</p>

        <form method="POST" action="/predict">
          <div class="field">
            <label for="rooms">Rooms</label>
            <input id="rooms" type="number" name="Rooms" step="1" min="0" required>
          </div>

          <div class="field">
            <label for="distance">Distance</label>
            <input id="distance" type="number" name="Distance" step="any" min="0" required>
            <div class="hint">Distancia al centro de la ciudad (km).</div>
          </div>

          <div class="field">
            <label for="bathroom">Bathroom</label>
            <input id="bathroom" type="number" name="Bathroom" step="1" min="0" required>
          </div>

          <div class="field">
            <label for="car">Car</label>
            <input id="car" type="number" name="Car" step="1" min="0" required>
          </div>

          <div class="field">
            <label for="landsize">Landsize</label>
            <input id="landsize" type="number" name="Landsize" step="any" min="0" required>
            <div class="hint">Tamaño del terreno en m².</div>
          </div>

          <button type="submit">Predecir segmento de precio</button>
        </form>

        <p class="api-note">
          También puedes consumir este servicio como API JSON mediante
          <code>POST /predict</code> con <code>Content-Type: application/json</code>.
        </p>
      </div>
    </div>
  </div>
</body>
</html>
"""
            self.wfile.write(html.encode("utf-8"))
        else:
            super().do_GET()

    # ------------ POST -------------
    def do_POST(self):
        if self.path != "/predict":
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Endpoint no encontrado")
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            ctype = self.headers.get("Content-Type", "")

            # ---- MODO API JSON ----
            if "application/json" in ctype:
                payload = json.loads(body.decode("utf-8"))

                if isinstance(payload, dict):
                    df = pd.DataFrame([payload])
                elif isinstance(payload, list):
                    df = pd.DataFrame(payload)
                else:
                    raise ValueError("El payload debe ser dict o lista de dicts.")

                missing_cols = [c for c in COLS_KEEP if c not in df.columns]
                if missing_cols:
                    raise ValueError(f"Faltan columnas en el JSON de entrada: {missing_cols}")

                X_raw = df[COLS_KEEP].copy()
                X_imp = imp.transform(X_raw)
                X_sc  = sc.transform(X_imp)

                probs   = model.predict(X_sc)
                classes = np.argmax(probs, axis=1)

                response = {
                    "classes": classes.tolist(),
                    "probabilities": probs.tolist()
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            # ---- MODO FORMULARIO HTML ----
            form_data = parse_qs(body.decode("utf-8"))

            sample = {}
            for col in COLS_KEEP:
                if col not in form_data:
                    raise ValueError(f"Falta el campo {col} en el formulario.")
                value_str = form_data[col][0]
                if col in ["Rooms", "Bathroom", "Car"]:
                    sample[col] = int(value_str)
                else:
                    sample[col] = float(value_str)

            df = pd.DataFrame([sample])

            X_raw = df[COLS_KEEP].copy()
            X_imp = imp.transform(X_raw)
            X_sc  = sc.transform(X_imp)

            probs   = model.predict(X_sc)
            classes = np.argmax(probs, axis=1)

            label_map = {0: "Barata", 1: "Media", 2: "Cara"}
            pred_class = int(classes[0])
            pred_label = label_map.get(pred_class, str(pred_class))
            p0, p1, p2 = probs[0]

            html = f"""
<html>
<head>
  <title>Resultado de predicción · Housing Classifier</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(135deg, #0f172a, #1e293b);
      margin: 0;
      padding: 20px;
      color: #0f172a;
    }}
    .wrapper {{
      max-width: 720px;
      margin: 40px auto;
    }}
    .card {{
      background: #f9fafb;
      border-radius: 16px;
      box-shadow: 0 18px 45px rgba(15, 23, 42, 0.38);
      padding: 24px 28px;
      box-sizing: border-box;
    }}
    h1 {{
      margin-top: 0;
      font-size: 24px;
      margin-bottom: 6px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      background: #e5e7eb;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #374151;
      margin-bottom: 10px;
    }}
    .badge-dot {{
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: #22c55e;
    }}
    .pill-class {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      margin-top: 6px;
      background: #dbeafe;
      color: #1d4ed8;
      font-weight: 500;
    }}
    .section-title {{
      font-size: 16px;
      margin-top: 18px;
      margin-bottom: 6px;
    }}
    ul {{
      padding-left: 18px;
      color: #374151;
      font-size: 14px;
    }}
    li {{
      margin-bottom: 4px;
    }}
    pre {{
      background: #f3f4f6;
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 13px;
      overflow-x: auto;
    }}
    a.button-back {{
      display: inline-block;
      margin-top: 16px;
      padding: 8px 14px;
      border-radius: 999px;
      background: #2563eb;
      color: #ffffff;
      text-decoration: none;
      font-size: 14px;
      font-weight: 500;
    }}
    a.button-back:hover {{
      background: #1d4ed8;
    }}
    .note {{
      font-size: 11px;
      color: #6b7280;
      margin-top: 6px;
    }}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="card">
      <div class="badge">
        <span class="badge-dot"></span>
        RESULTADO DE LA PREDICCIÓN
      </div>
      <h1>Segmento estimado de precio</h1>
      <p style="margin: 4px 0 8px; color:#4b5563;">
        Con base en las características ingresadas, el modelo de clasificación estima el siguiente segmento:
      </p>

      <p style="font-size:18px; margin-top:8px; margin-bottom:4px;">
        <strong>Clase predicha:</strong> {pred_label}
        <span style="color:#6b7280; font-size:14px;">(clase {pred_class})</span>
      </p>
      <span class="pill-class">Barata (0) · Media (1) · Cara (2)</span>

      <h2 class="section-title">Probabilidades por categoría</h2>
      <ul>
        <li>Barata (0): {p0:.3f}</li>
        <li>Media  (1): {p1:.3f}</li>
        <li>Cara   (2): {p2:.3f}</li>
      </ul>

      <h2 class="section-title">Datos ingresados</h2>
      <pre>{json.dumps(sample, indent=2)}</pre>

      <a href="/predict" class="button-back">&larr; Realizar otra predicción</a>

      <p class="note">
        Este resultado se basa en un modelo de Red Neuronal entrenado con datos históricos de viviendas.
        Los valores se interpretan como una aproximación estadística, no como una valoración comercial formal.
      </p>
    </div>
  </div>
</body>
</html>
"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = f"""
<html>
<head><title>Error</title></head>
<body style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#f4f5fb; margin:0; padding:20px;">
  <div style="max-width:600px; margin:40px auto; background:#ffffff; border-radius:12px; box-shadow:0 10px 25px rgba(0,0,0,0.08); padding:24px 28px;">
    <h1>Error en la predicción</h1>
    <p>{str(e)}</p>
    <p><a href="/predict" style="color:#2563eb;">Volver al formulario</a></p>
  </div>
</body>
</html>
"""
            self.wfile.write(html.encode("utf-8"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Servidor iniciando en puerto {port}...")
    with socketserver.TCPServer(("", port), PredictHandler) as httpd:
        print("Servidor listo, atendiendo solicitudes.")
        httpd.serve_forever()
