import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from keras import layers, models
from keras.applications import MobileNetV2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Scriptin bulunduğu dizine göre veri seti yolu
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'animals')

IMG_SIZE = 160 
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_SPLIT = 0.2

MODEL_PATH = 'hayvan_taniyici_model.keras'
CLASS_NAMES_PATH = 'class_names.json'

print("=" * 60)
print("🐾 HAYVAN TANIYICI - MODEL EĞİTİMİ")
print("=" * 60)

if not os.path.exists(DATA_DIR):
    print(f"HATA: Veri seti bulunamadı: {DATA_DIR}")
    exit(1)

class_names = sorted([d for d in os.listdir(DATA_DIR) 
                     if os.path.isdir(os.path.join(DATA_DIR, d))])
NUM_CLASSES = len(class_names)
print(f"\n Toplam {NUM_CLASSES} hayvan sınıfı bulundu:")
print(", ".join(class_names[:10]) + "..." if len(class_names) > 10 else ", ".join(class_names))

print("\n Veri yükleniyor")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names_list = train_ds.class_names
class_names_dict = {i: name for i, name in enumerate(class_names_list)}

with open(CLASS_NAMES_PATH, 'w') as f:
    json.dump(class_names_dict, f, indent=2)
print(f" Sınıf isimleri kaydedildi: {CLASS_NAMES_PATH}")

AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

normalization = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("\n Model oluşturuluyor")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

print("\n" + "=" * 60)
print("AŞAMA 1: Özellik Çıkarma Eğitimi")
print("=" * 60)

history1 = model.fit(
    train_ds,
    epochs=8,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 60)
print("AŞAMA 2: Fine-Tuning")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 60)
print("EĞİTİM SONUÇLARI")
print("=" * 60)

val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
print(f"\n Doğrulama Kaybı: {val_loss:.4f}")
print(f"Doğrulama Doğruluğu: %{val_accuracy*100:.2f}")

model.save(MODEL_PATH)
print(f"\n Model kaydedildi: {MODEL_PATH}")
print(f"Sınıf isimleri kaydedildi: {CLASS_NAMES_PATH}")

print("\n" + "=" * 60)
print("EĞİTİM TAMAMLANDI!")
print("=" * 60)