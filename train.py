"""The training module for ArcFace face recognition."""

import os
from argparse import ArgumentParser

from tensorflow import keras

from dataset import build_dataset
from losses import ArcLoss
from network import ArcLayer, L2Normalization, resnet101
from training_supervisor import TrainingSupervisor

parser = ArgumentParser()
parser.add_argument("--softmax", default=False, type=bool,
                    help="Training with softmax loss.")
parser.add_argument("--epochs", default=60, type=int,
                    help="Number of training epochs.")
parser.add_argument("--batch_size", default=128, type=int,
                    help="Training batch size.")
parser.add_argument("--export_only", default=False, type=bool,
                    help="Save the model without training.")
parser.add_argument("--restore_weights_only", default=False, type=bool,
                    help="Only restore the model weights from checkpoint.")
parser.add_argument("--override", default=False, type=bool,
                    help="Manually override the training objects.")
args = parser.parse_args()

if __name__ == "__main__":
    # Deep neural network training is complicated. The first thing is making
    # sure you have everything ready for training, like datasets, checkpoints,
    # logs, etc. Modify these paths to suit your needs.

    # What is the model's name?
    name = "resnetv2_101"

    # Where are the training files?
    train_files = "/home/robin/data/face/faces_ms1m-refine-v2_112x112/faces_emore/train.record"

    # Where are the testing files?
    test_files = None

    # Where are the validation files? Set `None` if no files available.
    val_files = None

    # What is the shape of the input image?
    input_shape = (112, 112, 3)

    # What is the size of the embeddings that represent the faces?
    embedding_size = 512

    # How many identities do you have in the training dataset?
    num_ids = 85742

    # How many examples do you have in the training dataset?
    num_examples = 5822653

    # That should be sufficient for training. However if you want more
    # customization, please keep going.

    # Where is the training direcotory for checkpoints and logs?
    training_dir = os.getcwd()

    # Where is the exported model going to be saved?
    export_dir = os.path.join(training_dir, 'exported', name)

    # Any weight regularization?
    regularizer = keras.regularizers.L2(5e-4)

    # How often do you want to log and save the model, in steps?
    frequency = 1000

    # All sets. Now it's time to build the model. There are two steps in ArcFace
    # training: 1, training with softmax loss; 2, training with arcloss. This
    # means not only different loss functions but also fragmented models.

    # First model is base model which outputs the face embeddings.
    base_model = resnet101(input_shape=input_shape, output_size=embedding_size,
                           trainable=True, training=True,
                           kernel_regularizer=regularizer,
                           name="embedding_model")

    # Then build the second model for training.
    if args.softmax:
        print("Building training model with softmax loss...")
        model = keras.Sequential([keras.Input(input_shape),
                                  base_model,
                                  keras.layers.Dense(num_ids,
                                                     kernel_regularizer=regularizer),
                                  keras.layers.Softmax()],
                                 name="training_model")
        loss_fun = keras.losses.CategoricalCrossentropy()
    else:
        print("Building training model with ARC loss...")
        model = keras.Sequential([keras.Input(input_shape),
                                  base_model,
                                  L2Normalization(),
                                  ArcLayer(num_ids, regularizer)],
                                 name="training_model")
        loss_fun = ArcLoss()

    # Summary the model to find any thing suspicious at early stage.
    model.summary()

    # Construct an optimizer. This optimizer is different from the official
    # implementation which use SGD with momentum.
    optimizer = keras.optimizers.Adam(0.1)

    # Construct training datasets.
    dataset_train = build_dataset(train_files,
                                  batch_size=args.batch_size,
                                  one_hot_depth=num_ids,
                                  training=True,
                                  buffer_size=4096)

    # Construct dataset for validation. The loss value from this dataset can be
    # used to decide which checkpoint should be preserved.
    if val_files:
        dataset_val = build_dataset(val_files,
                                    batch_size=args.batch_size,
                                    one_hot_depth=num_ids,
                                    training=False,
                                    buffer_size=4096)
    else:
        dataset_val = None

    # The training adventure is long and full of traps. A training supervisor
    # can help us to ease the pain.
    supervisor = TrainingSupervisor(model,
                                    optimizer,
                                    loss_fun,
                                    dataset_train,
                                    training_dir,
                                    frequency,
                                    "categorical_accuracy",
                                    'max',
                                    name)

    # If training accomplished, save the base model for inference.
    if args.export_only:
        print("The best model will be exported.")
        supervisor.restore(args.restore_weights_only, True)
        supervisor.export(base_model, export_dir)
        quit()

    # Restore the latest model if checkpoints are available.
    supervisor.restore(args.restore_weights_only)

    # Sometimes the training process might go wrong and we would like to resume
    # training from manually selected checkpoint. In this case some training
    # objects should be overridden before training started.
    if args.override:
        supervisor.override(0, 1)
        print("Training process overridden by user.")

    # Now it is safe to start training.
    supervisor.train(args.epochs, num_examples // args.batch_size)

    # Export the model after training.
    supervisor.export(base_model, export_dir)
