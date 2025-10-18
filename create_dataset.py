import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Создаем директории
os.makedirs('dataset/images/train', exist_ok=True)
os.makedirs('dataset/images/val', exist_ok=True)
os.makedirs('dataset/labels/train', exist_ok=True)
os.makedirs('dataset/labels/val', exist_ok=True)

def draw_smiley(img, center_x, center_y, size):
    """Рисует смайлик"""
    # Лицо
    cv2.circle(img, (center_x, center_y), size, (0, 255, 255), -1)
    cv2.circle(img, (center_x, center_y), size, (0, 0, 0), 2)
    
    # Глаза
    eye_shift = size // 3
    cv2.circle(img, (center_x - eye_shift, center_y - eye_shift), size//5, (0, 0, 0), -1)
    cv2.circle(img, (center_x + eye_shift, center_y - eye_shift), size//5, (0, 0, 0), -1)
    
    # Улыбка
    cv2.ellipse(img, (center_x, center_y + eye_shift), (size//2, size//3), 0, 0, 180, (0, 0, 0), 2)

def draw_heart(img, center_x, center_y, size):
    """Рисует сердце"""
    # Упрощенное сердце
    top_x, top_y = center_x, center_y - size//2
    left_x, left_y = center_x - size//2, center_y
    right_x, right_y = center_x + size//2, center_y
    bottom_x, bottom_y = center_x, center_y + size//2
    
    # Рисуем ромб (упрощенное сердце)
    points = np.array([[top_x, top_y], [right_x, right_y], 
                       [bottom_x, bottom_y], [left_x, left_y]], np.int32)
    cv2.fillPoly(img, [points], (255, 0, 0))
    cv2.polylines(img, [points], True, (0, 0, 0), 2)

def create_simple_dataset(num_images=100):
    """Создает простой датасет"""
    images = []
    labels = []
    
    for i in range(num_images):
        # Белый фон
        img = np.ones((320, 320, 3), dtype=np.uint8) * 255
        img_labels = []
        
        # Добавляем 1-2 объекта на изображение
        num_objects = np.random.randint(1, 3)
        
        for _ in range(num_objects):
            obj_type = np.random.choice(['smiley', 'heart'])
            size = np.random.randint(40, 80)
            x = np.random.randint(size, 320 - size)
            y = np.random.randint(size, 320 - size)
            
            if obj_type == 'smiley':
                draw_smiley(img, x, y, size)
                # YOLO формат: class x_center y_center width height
                x_center = x / 320
                y_center = y / 320
                width = (size * 2) / 320
                height = (size * 2) / 320
                img_labels.append(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}")
            else:
                draw_heart(img, x, y, size)
                x_center = x / 320
                y_center = y / 320
                width = (size * 2) / 320
                height = (size * 2) / 320
                img_labels.append(f"1 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}")
        
        images.append(img)
        labels.append(img_labels)
    
    return images, labels

print("Создаем датасет...")
images, labels = create_simple_dataset(80)  # 80 изображений
print("✅ Датсет создан!")


# Разделяем на train/val
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Сохраняем train
for i, (img, label_list) in enumerate(zip(train_images, train_labels)):
    cv2.imwrite(f'dataset/images/train/img_{i:03d}.jpg', img)
    with open(f'dataset/labels/train/img_{i:03d}.txt', 'w') as f:
        f.write('\n'.join(label_list))

# Сохраняем val
for i, (img, label_list) in enumerate(zip(val_images, val_labels)):
    cv2.imwrite(f'dataset/images/val/img_{i:03d}.jpg', img)
    with open(f'dataset/labels/val/img_{i:03d}.txt', 'w') as f:
        f.write('\n'.join(label_list))

print(f"📁 Train: {len(train_images)} изображений")
print(f"📁 Val: {len(val_images)} изображений")

# dataset.yaml
yaml_content = """
path: dataset
train: images/train
val: images/val

nc: 2
names: ['smiley', 'heart']
"""

with open('dataset.yaml', 'w') as f:
    f.write(yaml_content)

print("✅ Конфиг создан!")

from ultralytics import YOLO

# Берем самую маленькую модель
model = YOLO('yolov8n.pt')

# ЗАПУСКАЕМ ОБУЧЕНИЕ!
results = model.train(
    data='dataset.yaml',
    epochs=20,      # Всего 20 эпох
    imgsz=320,      # Маленький размер для скорости
    batch=8,        # Маленький батч
    patience=5,     # Остановка если нет улучшений
    device='cpu',   # Работает на любом компьютере!
    name='my_first_yolo',
    verbose=True    # Видим процесс
)

print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")

# Создаем тестовое изображение
test_img = np.ones((320, 320, 3), dtype=np.uint8) * 255
draw_smiley(test_img, 100, 100, 50)
draw_heart(test_img, 220, 200, 40)
cv2.imwrite('test.jpg', test_img)

# Загружаем нашу обученную модель
trained_model = YOLO('runs/detect/my_first_yolo/weights/best.pt')

# Предсказываем!
results = trained_model('test.jpg')

# Показываем результат
for r in results:
    im_array = r.plot()  # Рисуем bounding boxes
    cv2.imwrite('result.jpg', im_array)

print("📸 Результат сохранен в result.jpg")

import matplotlib.pyplot as plt
from PIL import Image

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Исходное изображение
ax1.imshow(Image.open('test.jpg'))
ax1.set_title('Исходное изображение')
ax1.axis('off')

# Результат
ax2.imshow(Image.open('result.jpg'))
ax2.set_title('YOLO детекция')
ax2.axis('off')

plt.show()

# Информация о детекциях
print("\n🔍 РЕЗУЛЬТАТЫ ДЕТЕКЦИИ:")
for i, box in enumerate(results[0].boxes):
    cls = int(box.cls[0])
    conf = box.conf[0]
    class_name = trained_model.names[cls]
    print(f"Объект {i+1}: {class_name} с уверенностью {conf:.2%}")

