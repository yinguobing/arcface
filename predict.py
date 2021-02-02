"""This module demonstrates how to run face recognition with ArcFace model."""
from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf

from preprocessing import normalize

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
        inputs = normalize(images)
        self.identities = self._get_embeddings(inputs)

    def identify(self, images, threshold):
        """Find the most similar persons from the input images.

        Args:
            images: a batch of images to be investgated.
            threshold: a threshold value to filter the results.

        Returns:
            the coresponding person's index
        """
        inputs = normalize(images)
        embeddings = self._get_embeddings(inputs)
        distances = self._get_distances(self.identities, embeddings)
        results = self._match(distances, threshold)

        return results


def _read_in(image_path):
    """A helper function to read and decode images from file."""
    image_string = tf.io.read_file(image_path)
    return tf.image.decode_jpeg(image_string, 3)


class FaceDetector:
    """Detect human face from image"""

    def __init__(self,
                 dnn_proto_text='./assets/deploy.prototxt',
                 dnn_model='./assets/res10_300x300_ssd_iter_140000.caffemodel'):
        """Initialization"""
        self.face_net = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)
        self.detection_result = None

    def get_faceboxes(self, image, threshold=0.5):
        """
        Get the bounding box of faces in image using dnn.
        """
        rows, cols, _ = image.shape

        confidences = []
        faceboxes = []

        self.face_net.setInput(cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
        detections = self.face_net.forward()

        for result in detections[0, 0, :, :]:
            confidence = result[2]
            if confidence > threshold:
                x_left_bottom = int(result[3] * cols)
                y_left_bottom = int(result[4] * rows)
                x_right_top = int(result[5] * cols)
                y_right_top = int(result[6] * rows)
                confidences.append(confidence)
                faceboxes.append(
                    [x_left_bottom, y_left_bottom, x_right_top, y_right_top])

        self.detection_result = [faceboxes, confidences]

        return confidences, faceboxes

    def draw_all_result(self, image):
        """Draw the detection result on image"""
        for facebox, conf in self.detection_result:
            cv2.rectangle(image, (facebox[0], facebox[1]),
                          (facebox[2], facebox[3]), (0, 255, 0))
            label = "face: %.4f" % conf
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                          (facebox[0] + label_size[0],
                           facebox[1] + base_line),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (facebox[0], facebox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]), box_color, 2)

    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:                   # Already a square.
            return box
        elif diff > 0:                  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:                           # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    def scale_box(self, box, ratio):
        """Scale the box by ratio."""
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        margin_x = box_width * ratio / 2
        margin_y = box_height * ratio / 2

        center_x = (left_x + right_x) / 2
        center_y = (top_y + bottom_y) / 2

        return [int(center_x - margin_x),
                int(center_y - margin_y),
                int(center_x + margin_y),
                int(center_y + margin_x)]

    def extract_cnn_faceboxes(self, image, threshold=0.8, scale=1.0):
        """Extract face area from image."""
        # Detect all faces.
        _, raw_boxes = self.get_faceboxes(
            image=image, threshold=0.9)

        # Filter all valid boxes.
        valid_boxes = []
        for box in raw_boxes:
            # Move box down.
            # diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            offset_y = int(abs((box[3] - box[1]) * 0.0))
            box_moved = self.move_box(box, [0, offset_y])

            # Make box square.
            facebox = self.get_square_box(box_moved)

            # Scale the box.
            box_scaled = self.scale_box(facebox, scale)

            if self.box_in_image(box_scaled, image):
                valid_boxes.append(box_scaled)

        return valid_boxes


if __name__ == '__main__':
    # Summon an AI to recognize faces.
    ai = WanderingAI('exported/hrnetv2')

    # Tell the AI to remember these faces.
    faces_to_remember = ["/home/robin/Pictures/wanda.jpg",
                         "/home/robin/Pictures/vision.jpg",
                         "/home/robin/Pictures/monica.jpg"]
    names = ["Wonda", "Vision", "Monica"]
    targets = [_read_in(x) for x in faces_to_remember]
    ai.remember(targets)

    # Video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _, sample_frame = cap.read()

    # Video output by video writer.
    if args.write_video:
        fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        video_writer = cv2.VideoWriter(
            "output.avi", fourcc, frame_rate, (width, height))

    # Construct a face detector.
    face_detector = FaceDetector()

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[:480, :640]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Get face area images.
        faceboxes = face_detector.extract_cnn_faceboxes(frame, 0.5, 1.0)

        color_palette = [(54, 84, 160), [160, 84, 54], [84, 160, 54]]

        if faceboxes:
            faces = []
            for facebox in faceboxes:
                # Preprocess the sample image
                face_img = frame[facebox[1]: facebox[3],
                                 facebox[0]: facebox[2]]
                face_img = cv2.resize(face_img, (112, 112))
                img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                faces.append(img_rgb)

            faces = np.array(faces, dtype=np.float32)

            # Do prediction.
            results = ai.identify(faces, 0.90)

            # Draw the names on image.
            labels = ["Nobody"] * len(faceboxes)
            values = [0] * len(faceboxes)
            colors = [(54, 54, 54)] * len(faceboxes)

            for p_id, c_id, distance in results:
                labels[c_id] = names[p_id]
                values[c_id] = distance
                colors[c_id] = color_palette[p_id]

            for label, value, box, color in zip(labels, values, faceboxes, colors):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_8)
                cv2.rectangle(frame, (x1-2, y1),
                              (x2+2, y1-40), color, cv2.FILLED)
                cv2.putText(frame, "{}: {:.2f}".format(label, value), (x1+7, y1-10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the result in windows.
        cv2.imshow('image', frame)

        # Write video file.
        if args.write_video:
            video_writer.write(frame)

        if cv2.waitKey(1) == 27:
            break
