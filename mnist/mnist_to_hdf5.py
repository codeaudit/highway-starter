import h5py

# wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

with h5py.File("mnist.hdf5", "w") as f:
    f.create_dataset("X_train", data=mnist.train.images)
    f.create_dataset("y_train", data=mnist.train.labels)

    f.create_dataset("X_validate", data=mnist.validation.images)
    f.create_dataset("y_validate", data=mnist.validation.labels)

    f.create_dataset("X_test", data=mnist.test.images)
    f.create_dataset("y_test", data=mnist.test.labels)
