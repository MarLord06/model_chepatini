import tensorflow as tf
from model_unet import build_unet
from losses import bce_dice_loss

# Ejemplo de dataset con tf.data
def load_dataset(images, masks, batch_size=4):
    ds = tf.data.Dataset.from_tensor_slices((images, masks))
    ds = ds.shuffle(200)
    ds = ds.map(lambda x, y: (tf.image.resize(x, (512,512)) / 255.0,
                              tf.image.resize(y, (512,512))))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Cargar tus rutas aquí
train_images = ...
train_masks  = ...
val_images   = ...
val_masks    = ...

train_ds = load_dataset(train_images, train_masks)
val_ds   = load_dataset(val_images, val_masks)

model = build_unet()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=bce_dice_loss,
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    epochs=40,
    validation_data=val_ds
)

model.save("unet_severstal.h5")
