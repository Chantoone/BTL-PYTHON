import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# Tạo bộ dữ liệu và áp dụng tăng cường dữ liệu
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('data/train', target_size=(224, 224),
                                                    batch_size=32, class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory('data/validation', target_size=(224, 224),
                                                batch_size=32, class_mode='categorical')

# Tải mô hình ResNet50 với trọng số đã được huấn luyện trước
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Thêm các lớp fully connected tùy chỉnh vào mô hình
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(6, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Tạo mô hình hoàn chỉnh
model = Model(inputs=base_model.input, outputs=predictions)

# Đóng băng các lớp của mô hình gốc ResNet50
for layer in base_model.layers:
    layer.trainable = False

# Biên dịch mô hình
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(train_generator, validation_data=val_generator, epochs=10)
