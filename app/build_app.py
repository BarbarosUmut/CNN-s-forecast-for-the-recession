"""Deneme uygulamasını tek bir HTML dosyası olarak derler.

Kullanım (depo kök klasöründen):
    python app/build_app.py

Girdi : app/template.html, app/model.js, model_export/app_model.json (notebook üretir)
Çıktı : docs/index.html  (GitHub Pages bu dosyayı yayınlar; tarayıcıda çift tıklayarak da açılır)
"""
import json
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
sablon = (KOK / "app" / "template.html").read_text(encoding="utf-8")
model_js = (KOK / "app" / "model.js").read_text(encoding="utf-8")
model = json.loads((KOK / "model_export" / "app_model.json").read_text(encoding="utf-8"))

# JSON'u <script> etiketinin içine güvenle gömmek için "</" dizisini kaçırıyoruz
model_json = json.dumps(model, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = sablon.replace("__MODEL_JS__", model_js).replace("__MODEL_JSON__", model_json)

cikti = KOK / "docs" / "index.html"
cikti.parent.mkdir(exist_ok=True)
cikti.write_text(html, encoding="utf-8")
print(f"Yazıldı: {cikti.relative_to(KOK)} ({len(html.encode('utf-8')) / 1024:.0f} KB)")
