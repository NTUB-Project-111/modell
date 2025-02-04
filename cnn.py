import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 設定隨機種子，便於結果重現
np.random.seed(10)

# 1. 數據增強：對圖片進行各種隨機變換
datagen = ImageDataGenerator(
    rescale=1.0/255,          # 將像素值縮放至 [0, 1]
    rotation_range=20,        # 隨機旋轉
    width_shift_range=0.2,    # 隨機水平平移
    height_shift_range=0.2,   # 隨機垂直平移
    shear_range=0.2,          # 剪切變換
    zoom_range=0.2,           # 隨機縮放
    horizontal_flip=True,     # 隨機水平翻轉
    fill_mode='nearest'       # 填補方式
)

# 設定訓練和驗證數據的路徑
train_path = "C:/cnn/train/"
validation_path = "C:/cnn/validation/"

# 讀取訓練數據
train_generator = datagen.flow_from_directory(
    train_path,
    target_size=(80, 80),       # 調整圖片大小為 80x80
    batch_size=32,
    class_mode='categorical',   # 分類模式
    shuffle=True
)

# 讀取驗證數據
validation_generator = datagen.flow_from_directory(
    validation_path,
    target_size=(80, 80),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 2. 建立 CNN 模型
model = Sequential()

# 第一層卷積與池化
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 第二層卷積與池化
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 第三層卷積與池化
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 全連接層
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # 加入 Dropout 避免過擬合
model.add(Dense(5, activation='softmax'))  # 幾個類別：看放幾個類別之傷口做調整！

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 顯示模型架構
model.summary()

# 3. 設定回調函數
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

# 4. 訓練模型
train_history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,                   # 訓練 30 個 epoch
    callbacks=[early_stopping, model_checkpoint],
    verbose=2
)

# 5. 評估模型
scores = model.evaluate(validation_generator)
print(f"驗證準確率: {scores[1]}")

# 儲存模型
model.save("cut_vs_abrasion_model.keras")

# 6. 將模型轉換為 TensorFlow Lite 格式（如需要）
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 儲存 TFLite 模型
with open("cut_vs_abrasion_model.tflite", "wb") as f:
    f.write(tflite_model)
