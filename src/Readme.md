
## Installation

To install the package, clone the repo and execute the following command in the rvd root folder:

```bash
$ pip install .
```

<!-- The package will be made available on PyPI. To install it, execute the following command within your Python environment:

```bash
$ pip install rvd
``` -->

<!-- We tested the code with Python version 3.8.18 and pip version 23.3.1. -->

## Usage

### Computing the voicing decision from audio

```python
import multipitch


# Assign the path of the audio file
audio_filename = ...

# Specify the path to the pre-trained model file
model_file = ...

# Set the voicing threshold
voicing_threshold = ...

# Choose a device to use for inference
device = 'cuda:0'

# Create the model with the specified model file, voicing threshold, and device
model = rvd.Model(model_file, voicing_threshold, device)

# Compute voicing decision using first gpu
vd = model.predict(audio_filename)

```


## References

[1] D. Talkin and W. B. Kleijn, “A robust algorithm for pitch tracking (RAPT),” Speech Coding and Synthesis, vol. 495, p. 518, 1995.

[2] P. C. Bagshaw, S. M. Hiller, and M. A. Jack, “Enhanced pitch tracking and the processing of F0 contours for computer aided intonation teaching,” in Proc. Eurospeech, 1993.

[3] G. Pirker, M. Wohlmayr, S. Petrik, and F. Pernkopf, “A pitch tracking corpus with evaluation on multipitch tracking scenario,” in Proc. Interspeech, 2011.

[4] F. Plante, G. Meyer, and W. Ainsworth, “A pitch extraction reference database,” in Proc. Eurospeech, 1995.

[5] https://data.cstr.ed.ac.uk/mocha/

[6] J. Kominek and A. W. Black, “The CMU arctic speech databases,” in Fifth ISCA workshop on speech synthesis, 2004.


