import os
import argparse

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

mnist = tf.keras.datasets.mnist

parser = argparse.ArgumentParser(description='Basic MNIST NN')

parser.add_argument('--load_checkpoint', type=str, default='',
                    help='Relative path and filename to load weights from')
parser.add_argument('--save_checkpoint', type=str, default='',
                    help='Relative from CWD path to save weights to')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout Rate')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--decay', type=float, default=1e-6,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=1,
                    help='max number of epochs')
parser.add_argument('--FoM', action="store_true",
                    help="Print Feature of Merit for HPO")

args = parser.parse_args()

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

checkpoint_path_load = args.save_checkpoint
checkpoint_path_out = args.load_checkpoint
if args.load_checkpoint != '' and os.path.isfile(checkpoint_path_load):
    print('Loading model from checkpoint: ', checkpoint_path_load)
    model = tf.keras.models.load_model(checkpoint_path_load)
else:
    if not os.path.isfile(checkpoint_path_load):
        print("Warning: No checkpoint found, starting new model!", checkpoint_path_load)

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28,28,)),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dropout(args.dropout),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

callbacks = []

if args.save_checkpoint != '':
    checkpoint_dir = os.path.dirname(checkpoint_path_out)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print('Setting up checkpoint save callback: ', checkpoint_path_out)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_out)
    callbacks.append(cp_callback)

opt = tf.keras.optimizers.SGD(lr=args.lr, decay=args.decay)

model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=args.epochs, callbacks=callbacks)
losses = model.evaluate(x_test, y_test)

print("FoM: %e" % losses[0])
