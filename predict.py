"""This module demonstrates how to run face recognition with ArcFace model."""
import tensorflow as tf
from preprocessing import normalize


class WanderingAI():
    """A lightweight class implementation of face recognition."""

    def __init__(self, model_path):
        """Initialize an AI to recognize faces.

        Args:
            model_path: the exported model file path.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.identities = None

    def _get_embeddings(self, inputs):
        """Return the face embeddings of the inputs tensor.

        Args:
            inputs: a batch of processed input tensors..

        Returns:
            the embeddings.
        """
        embeddings = self.model(inputs)
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)

        return embeddings

    def _get_distances(self, embeddings_1, embeddings_2):
        """Return the distances between input embeddings.

        Args:
            embeddings_1: the first batch of embeddings.
            embeddings_2: the second batch of embeddings.

        Returns:
            the distances.
        """
        distances = []
        for embedding_1 in tf.unstack(embeddings_1):
            s_diff = tf.math.squared_difference(embedding_1, embeddings_2)
            distances.append(tf.reduce_sum(s_diff, axis=1))

        return distances

    def remember(self, images):
        """Let AI remember the input faces.

        Args:
            images: the face images of the targets to remember.
        """
        inputs = normalize(images)
        self.identities = self._get_embeddings(inputs)

    def identify(self, images):
        """Find the most similar persons from the input images.

        Args:
            images: a batch of images to be investgated.

        Returns:
            the coresponding person's index
        """
        inputs = normalize(images)
        embeddings = self._get_embeddings(inputs)
        distances = self._get_distances(self.identities, embeddings)
        indices = tf.math.argmin(distances, axis=1)

        return indices


def _read_in(image_path):
    """A helper function to read and decode images from file."""
    image_string = tf.io.read_file(image_path)
    return tf.image.decode_jpeg(image_string, 3)


if __name__ == '__main__':
    # Test files list.
    image_files_1 = ["/home/robin/Pictures/resources/img_2.jpg",
                     "/home/robin/Pictures/resources/img_16.jpg"]
    image_files_2 = ["/home/robin/Pictures/resources/img_3.jpg",
                     "/home/robin/Pictures/resources/img_6.jpg",
                     "/home/robin/Pictures/resources/img_16.jpg"]

    # Read and decode the image files.
    group_1 = [_read_in(x) for x in image_files_1]
    group_2 = [_read_in(x) for x in image_files_2]

    # Call an AI to recognize these faces.
    ai = WanderingAI('exported/hrnetv2')

    # Tell the AI to remember the faces in the first group.
    ai.remember(group_1)

    # Find out the most similar faces in the second image group.
    indices = ai.identify(group_2).numpy()

    # What result do we have?
    for target, id in zip(image_files_1, indices):
        print("Similar pairs: {} - {}".format(target, image_files_2[id]))
