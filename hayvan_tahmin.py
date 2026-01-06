import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from PIL import Image
import json
import sys

# Ayarlar
IMG_SIZE = 160
MODEL_PATH = 'hayvan_taniyici_model.keras'
CLASS_NAMES_PATH = 'class_names.json'


TURKCE_ISIMLER = {
    'antelope': 'Antilop',
    'badger': 'Porsuk',
    'bat': 'Yarasa',
    'bear': 'Ayı',
    'bee': 'Arı',
    'beetle': 'Böcek',
    'bison': 'Bizon',
    'boar': 'Yaban Domuzu',
    'butterfly': 'Kelebek',
    'cat': 'Kedi',
    'caterpillar': 'Tırtıl',
    'chimpanzee': 'Şempanze',
    'cockroach': 'Hamamböceği',
    'cow': 'İnek',
    'coyote': 'Çakal',
    'crab': 'Yengeç',
    'crow': 'Karga',
    'deer': 'Geyik',
    'dog': 'Köpek',
    'dolphin': 'Yunus',
    'donkey': 'Eşek',
    'dragonfly': 'Yusufçuk',
    'duck': 'Ördek',
    'eagle': 'Kartal',
    'elephant': 'Fil',
    'flamingo': 'Flamingo',
    'fly': 'Sinek',
    'fox': 'Tilki',
    'goat': 'Keçi',
    'goldfish': 'Japon Balığı',
    'goose': 'Kaz',
    'gorilla': 'Goril',
    'grasshopper': 'Çekirge',
    'hamster': 'Hamster',
    'hare': 'Tavşan',
    'hedgehog': 'Kirpi',
    'hippopotamus': 'Su Aygırı',
    'hornbill': 'Boynuzgaga',
    'horse': 'At',
    'hummingbird': 'Sinekkuşu',
    'hyena': 'Sırtlan',
    'jellyfish': 'Denizanası',
    'kangaroo': 'Kanguru',
    'koala': 'Koala',
    'ladybugs': 'Uğur Böceği',
    'leopard': 'Leopar',
    'lion': 'Aslan',
    'lizard': 'Kertenkele',
    'lobster': 'Istakoz',
    'mosquito': 'Sivrisinek',
    'moth': 'Güve',
    'mouse': 'Fare',
    'octopus': 'Ahtapot',
    'okapi': 'Okapi',
    'orangutan': 'Orangutan',
    'otter': 'Su Samuru',
    'owl': 'Baykuş',
    'ox': 'Öküz',
    'oyster': 'İstiridye',
    'panda': 'Panda',
    'parrot': 'Papağan',
    'pelecaniformes': 'Pelikan',
    'penguin': 'Penguen',
    'pig': 'Domuz',
    'pigeon': 'Güvercin',
    'porcupine': 'Kirpi',
    'possum': 'Keseli Sıçan',
    'raccoon': 'Rakun',
    'rat': 'Sıçan',
    'reindeer': 'Ren Geyiği',
    'rhinoceros': 'Gergedan',
    'sandpiper': 'Kumkuşu',
    'seahorse': 'Denizatı',
    'seal': 'Fok',
    'shark': 'Köpekbalığı',
    'sheep': 'Koyun',
    'snake': 'Yılan',
    'sparrow': 'Serçe',
    'squid': 'Kalamar',
    'squirrel': 'Sincap',
    'starfish': 'Denizyıldızı',
    'swan': 'Kuğu',
    'tiger': 'Kaplan',
    'turkey': 'Hindi',
    'turtle': 'Kaplumbağa',
    'whale': 'Balina',
    'wolf': 'Kurt',
    'wombat': 'Vombat',
    'woodpecker': 'Ağaçkakan',
    'zebra': 'Zebra'
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyası bulunamadı: {MODEL_PATH}")
        sys.exit(1)
    
    print("Model yükleniyor...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model başarıyla yüklendi!")
    return model


def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"HATA: Sınıf isimleri dosyası bulunamadı: {CLASS_NAMES_PATH}")
        sys.exit(1)
    
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    
    return {int(k): v for k, v in class_names.items()}


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")
    
    img = Image.open(image_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((IMG_SIZE, IMG_SIZE))   
    img_array = np.array(img) / 255.0    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_animal(model, class_names, image_path):
    """Görüntüdeki hayvanı tahmin et"""
    
    img_array = preprocess_image(image_path)   
    predictions = model.predict(img_array, verbose=0)   
    top_indices = np.argsort(predictions[0])[::-1][:5]
    
    results = []
    for idx in top_indices:
        english_name = class_names[idx]
        turkish_name = TURKCE_ISIMLER.get(english_name, english_name)
        probability = predictions[0][idx] * 100
        results.append({
            'english': english_name,
            'turkish': turkish_name,
            'probability': probability
        })
    
    return results


def print_results(results, image_path):
    """Sonuçları güzel formatta yazdır"""
    print("\n" + "="*50)
    print(f"Görüntü: {os.path.basename(image_path)}")
    print("="*50)
    
    print(f"\nTAHMİN: {results[0]['turkish'].upper()}")
    print(f"   ({results[0]['english']})")
    print(f"   Güven: %{results[0]['probability']:.2f}")
    
    print("\nDiğer olasılıklar:")
    for i, result in enumerate(results[1:], 2):
        bar_length = int(result['probability'] / 2)
        bar = "█" * bar_length
        print(f"   {i}. {result['turkish']:15} ({result['english']:15}) %{result['probability']:5.2f} {bar}")
    
    print("="*50 + "\n")


def interactive_mode(model, class_names):
    """İnteraktif mod - kullanıcıdan sürekli görüntü al"""
    print("\n" + "="*50)
    print("HAYVAN TANIYICI - İNTERAKTİF MOD")
    print("="*50)
    print("Görüntü yolunu girin veya çıkmak için 'q' yazın.\n")
    
    while True:
        image_path = input("Görüntü yolu: ").strip()
        
        if image_path.lower() in ['q', 'quit', 'exit', 'çık']:
            print("Güle güle!")
            break
        
        if not image_path:
            print("Lütfen bir görüntü yolu girin.\n")
            continue
        
        image_path = image_path.strip('"').strip("'")
        
        try:
            results = predict_animal(model, class_names, image_path)
            print_results(results, image_path)
        except FileNotFoundError as e:
            print(f"Hata: {e}\n")
        except Exception as e:
            print(f"Beklenmeyen hata: {e}\n")


def main():
    """Ana fonksiyon"""
    # Model ve sınıf isimlerini yükle
    model = load_model()
    class_names = load_class_names()
    
    # Komut satırı argümanı kontrolü
    if len(sys.argv) > 1:
        # Argüman olarak verilen görüntüyü tahmin et
        image_path = sys.argv[1]
        try:
            results = predict_animal(model, class_names, image_path)
            print_results(results, image_path)
        except FileNotFoundError as e:
            print(f"Hata: {e}")
            sys.exit(1)
    else:
        # İnteraktif mod
        interactive_mode(model, class_names)


if __name__ == "__main__":
    main()
