
# Traffic Sign Classifier

This code builds and trains a neural network on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, and then saves the model for use in identifying traffic signs from real world images.

## Usage

### Training

A pre-trained model has already been provided at `traffic_ai_v2.h5`.

If you want to train your own model and adjust parameters, you must first download the GTSRB dataset and save it to a directory named "gtrsb".

Then run the following command to train and save the model:
```bash
  python traffic_ai.py
```

### Testing

To test the model on your own image, save the image to `stopsign2.JPG` and run:
```bash
  python sign_recogniser.py
```