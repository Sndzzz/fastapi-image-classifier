import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Modeli yükle
model = tf.keras.models.load_model("model/model.h5")

# Sınıf isimleri (Modeline göre değiştir)
class_labels = { 
    0: "Hız Sınırı (20 km/h)",
    1: "Hız Sınırı (30 km/h)",
    2: "Hız Sınırı (50 km/h)",
    3: "Hız Sınırı (60 km/h)",
    4: "Hız Sınırı (70 km/h)",
    5: "Hız Sınırı (80 km/h)",
    6: "Özel Hız Sınırı Sonu (80 km/h)",
    7: "Hız Sınırı (100 km/h)",
    8: "Hız Sınırı (120 km/h)",
    9: "Geçiş Yapılmaz",
    10: "Kamyonlar İçin Geçiş Yapılmaz",
    11: "Ana Yol Önceliği",
    12: "Dur",
    13: "Giriş Yapılmaz",
    14: "Giriş Yasak",
    15: "Trafik Lambası",
    16: "Yaya Geçidi",
    17: "Okul Geçidi",
    18: "Dikkat",
    19: "Viraj (Sağ)",
    20: "Viraj (Sol)",
    21: "Çift Viraj",
    22: "Dönel Kavşak",
    23: "Kaygan Yol",
    24: "Sağdan Daralan Yol",
    25: "İş Çalışması",
    26: "Trafik Işığı",
    27: "Yayalar İçin Geçiş Önceliği",
    28: "Sağa Dön",
    29: "Sola Dön",
    30: "Düz Git",
    31: "Düz veya Sağa Git",
    32: "Düz veya Sola Git",
    33: "Sağdan Geç",
    34: "Soldan Geç",
    35: "Sağdan ve Soldan Geç",
    36: "Dönüş Yasağı (Sağ)",
    37: "Dönüş Yasağı (Sol)",
    38: "Dönüş Yasağı",
    39: "Gidiş Yönü (Sağ)",
    40: "Gidiş Yönü (Sol)",
    41: "Ana Yol",
    42: "Kamyonlar İçin Gidiş Yönü (Sağ)"
}

def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((30, 30))  # Modeline uygun boyut
    image = np.array(image) / 255.0   # Normalizasyon
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    return {
        "class": class_labels[predicted_class],
        "probability": float(np.max(predictions))
    }
