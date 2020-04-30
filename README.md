# Sign language interpretation

This project aims to lower the barrier for communicating for non-speech persons. The projects original focus has been on interpreting the Swedish hand alphabet but the models can be retrained with data for other languages as well.

The project has two main ways of preprocessing the images for the convolutional neural network. One of which is Canny edge detection and the other is just using plain grayscale images.

### Using the pretrained models

1. Install the necessary dependencies by running ``pip install -r requirements.txt``
2. Run either the ``predict.py``, ``predict_edge.py`` or ``predict_edge_128.py``script depending on which model you would like to use.

### Model stats

| Model     | Loss   | Accuracy |
|-----------|--------|----------|
| Canny_64  | 0.1686 | 0.9598   |
| Gray_64   | 0.2164 | 0.9502   |
| Gray_128  | 0.3537 | 0.9282   |
| Canny_128 | 0.4059 | 0.9234   |

### Adding more data to the corpus

 - The ``process_images.py`` script can be used to add more data to the existing corpus. Edit the script so that the output path matches the path to which the corpus has been extracted.
 - The ``capture.py``script can be used to generate more images of the Swedish hand alphabet
 

 