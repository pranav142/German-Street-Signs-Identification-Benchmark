from my_utils import split_data, order_test_set, create_generators
from DeepLearning_Models import street_signs_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    if False:
        path_to_data = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Signs_Data\Train"
        path_to_save_train = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Signs_Data\Training_Data\Train"
        path_to_save_val = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Signs_Data\Training_Data\Val"   
        split_data(path_to_data, path_to_save_train, path_to_save_val)
    
    if False:
        path_to_images = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Signs_Data\Test"
        path_to_csv = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Signs_Data\Test.csv"
        order_test_set(path_to_images, path_to_csv)
    
    TRAIN = False
    TEST = True
    
    train_data_path = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Signs_Data\Training_Data\Train"
    val_data_path = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Signs_Data\Training_Data\Val"
    test_data_path = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Signs_Data\Test"
    path_to_save_model = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Street Signs\SavedModels"
    batch_size = 64
    epochs = 15
    
    train_generator,val_generator,test_generator = create_generators(batch_size=batch_size, train_data_path=train_data_path, val_data_path= val_data_path, test_data_path=test_data_path)

    num_classes = train_generator.num_classes

    if TRAIN:

        chpkt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor="val_loss",
            mode="min",
            save_best_only = True,
            save_freq = 'epoch',
            verbose = 1,
        )

        early_stop = EarlyStopping(
            monitor = "val_accuracy",
            patience = 2,
        )

        model = street_signs_model(num_classes=num_classes)

        model.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics = ['accuracy'])

        model.fit(train_generator, 
                batch_size= batch_size, 
                epochs=epochs, 
                validation_data= val_generator,
                callbacks = [chpkt_saver, early_stop]
                )

    if TEST:
        model = tf.keras.models.load_model("C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Street Signs\SavedModels")
        model.summary()

        print("Evaluating validation set: ")
        model.evaluate(val_generator)

        print("Evaluating test set")
        model.evaluate(test_generator)
    




