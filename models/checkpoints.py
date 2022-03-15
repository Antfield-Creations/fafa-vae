from keras.callbacks import ModelCheckpoint


def get_checkpoint_callback(checkpoint_folder: str) -> ModelCheckpoint:
    return ModelCheckpoint(
        filepath=checkpoint_folder,
        monitor='loss',
        save_freq='epoch',
        verbose=1,
    )
