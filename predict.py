"""This module demonstrates how to run face recognition with ArcFace model."""
from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf

from preprocessing import normalize
from face_detector.detector import Detector

parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
parser.add_argument("--write_video", type=bool, default=False,
                    help="Write processing video file.")
args = parser.parse_args()


class WanderingAI():
    """A lightweight class implementation of face recognition."""

    def __init__(self, model_path):
        """Initialize an AI to recognize faces.

        Args:
            model_path: the exported model file path.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.identities = None

    def _preprocess(self, inputs):
        """Preprocess the input images.

        Args:
            inputs: a batch of raw images.

        Returns:
            a batch of processed images as tensors.
        """
        return normalize(inputs)

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

    def _get_distances(self, embeddings_1, embeddings_2, element_wise=False):
        """Return the distances between input embeddings.

        Args:
            embeddings_1: the first batch of embeddings.
            embeddings_2: the second batch of embeddings.
            element_wise: get distances element wise.

        Returns:
            the distances list.
        """
        if element_wise:
            s_diff = tf.math.squared_difference(embeddings_1, embeddings_2)
            distances = tf.unstack(tf.reduce_sum(s_diff, axis=1))
        else:
            distances = []
            for embedding_1 in tf.unstack(embeddings_1):
                s_diff = tf.math.squared_difference(embedding_1, embeddings_2)
                distances.append(tf.reduce_sum(s_diff, axis=1))

        return distances

    def _match(self, distances, threshold):
        """Find out the matching result from the distances array.

        Args:
            distances: the distances array.
            threshold: the threshold to filter the negative samples.

        Returns:
            the matching results [[person, candidate], ...].
        """
        matched_pairs = []
        distances = np.array(distances, dtype=np.float32)
        num_results = np.min(distances.shape)

        for _ in range(num_results):
            min_distance = np.min(distances, axis=None)
            if min_distance > threshold:
                break
            arg_min = np.argmin(distances, axis=None)
            row, col = np.unravel_index(arg_min, distances.shape)
            matched_pairs.append([row, col, min_distance])
            distances[row, :] = 666
            distances[:, col] = 666

        return matched_pairs

    def remember(self, images):
        """Let AI remember the input faces.

        Args:
            images: the face images of the targets to remember.
        """
        inputs = self._preprocess(images)
        self.identities = self._get_embeddings(inputs)

    def identify(self, images, threshold):
        """Find the most similar persons from the input images.

        Args:
            images: a batch of images to be investgated.
            threshold: a threshold value to filter the results.

        Returns:
            the coresponding person's index
        """
        inputs = self._preprocess(images)
        embeddings = self._get_embeddings(inputs)
        distances = self._get_distances(self.identities, embeddings)
        results = self._match(distances, threshold)

        return results


def _read_in(image_path):
    """A helper function to read and decode images from file."""
    image_string = tf.io.read_file(image_path)
    return tf.image.decode_jpeg(image_string, 3)


if __name__ == '__main__':
    # Summon an AI to recognize faces.
    ai = WanderingAI('exported/hrnetv2')

    # Tell the AI to remember these faces.
    faces_to_remember = ["/home/robin/Pictures/wanda.jpg",
                         "/home/robin/Pictures/vision.jpg",
                         "/home/robin/Pictures/monica.jpg"]
    targets = [_read_in(x) for x in faces_to_remember]
    ai.remember(targets)

    # Setup the names and the bounding boxes colors.
    names = ["Wanda", "Vision", "Monica"]
    color_palette = [(54, 84, 160), [160, 84, 54], [84, 160, 54]]

    # What is the threshold value for face detection.
    threshold = 0.7

    # Construct a face detector.
    detector_face = Detector('assets/face_model')

    # Setup the video source. If no video file provided, the default webcam will
    # be used.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)

    # If reading frames from a webcam, try setting the camera resolution.
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Get the real frame resolution.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Video output by video writer.
    if args.write_video:
        video_writer = cv2.VideoWriter(
            'output.avi', cv2.VideoWriter_fourcc(*'avc1'), frame_rate,
            (frame_width, frame_height))

    # Introduce a metter to measure the FPS.
    tm = cv2.TickMeter()

    # Loop through the video frames.
    while True:
        # Start the metter.
        tm.start()

        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if not frame_got:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Preprocess the input image.
        _image = detector_face.preprocess(frame)

        # Run the model
        boxes, _, _ = detector_face.predict(_image, threshold)

        # Transform the boxes into squares.
        boxes = detector_face.transform_to_square(boxes, 0.9)

        # Clip the boxes if they cross the image boundaries.
        boxes, _ = detector_face.clip_boxes(
            boxes, (0, 0, frame_height, frame_width))
        boxes = boxes.astype(np.int32)
        num_boxes = boxes.shape[0]

        if num_boxes > 0:
            faces = []
            for facebox in boxes:
                # Crop the face image
                top, left, bottom, right = facebox
                face_image = frame[top:bottom, left:right]

                # Preprocess it.
                face_image = cv2.resize(face_image, (112, 112))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                faces.append(face_image)

            faces = np.array(faces, dtype=np.float32)

            # Do prediction.
            results = ai.identify(faces, 0.9)

            # Draw the names on image.
            labels = ["Nobody"] * num_boxes
            values = [0] * num_boxes
            colors = [(54, 54, 54)] * num_boxes

            for p_id, c_id, distance in results:
                labels[c_id] = names[p_id]
                values[c_id] = distance
                colors[c_id] = color_palette[p_id]

            for label, value, box, color in zip(labels, values, boxes, colors):
                y1, x1, y2, x2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_8)
                cv2.rectangle(frame, (x1-2, y1),
                              (x2+2, y1-40), color, cv2.FILLED)
                cv2.putText(frame, "{}: {:.2f}".format(label, value), (x1+7, y1-10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        tm.stop()

        # Show the result in windows.
        cv2.imshow('image', frame)

        # Write video file.
        if args.write_video:
            video_writer.write(frame)

        if cv2.waitKey(1) == 27:
            break
