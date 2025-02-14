import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l2

# 設定隨機種子，便於結果重現
np.random.seed(10)

# 1. 數據增強：對圖片進行各種隨機變換
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    channel_shift_range=50.0
)

# 設定訓練和驗證數據的路徑
train_path = "C:/cnn_test/train/"
validation_path = "C:/cnn_test/validation/"

# 讀取訓練數據
train_generator = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),  # 增加解析度
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# 讀取驗證數據
validation_generator = datagen.flow_from_directory(
    validation_path,
    target_size=(224, 224),  # 增加解析度
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 2. 使用 VGG16 作為遷移學習基礎
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # 先凍結預訓練層

# 解凍 VGG16 的最後幾層
for layer in base_model.layers[-4:]:
    layer.trainable = True

# 3. 新增自訂層進行微調
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

# 編譯模型，使用較低的學習率
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# 顯示模型架構
model.summary()

# 4. 設定回調函數
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)

# 5. 訓練模型
train_history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,  # 增加訓練輪數
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=2
)

# 6. 評估模型
scores = model.evaluate(validation_generator)
print(f"驗證準確率: {scores[1]}")

# 儲存模型
model.save("cut_vs_abrasion_model.keras")

# 7. 將模型轉換為 TensorFlow Lite 格式（如需要）
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 儲存 TFLite 模型
with open("cut_vs_abrasion_model.tflite", "wb") as f:
    f.write(tflite_model)
