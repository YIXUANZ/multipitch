# Source Code Overview

This folder contains the source code and supporting files for the multipitch tracking project. Below is an overview of the key files and their functionalities:

## File Descriptions

- **crossnet.py**: TF-CrossNet for speaker separation.
- **dccrn.py**: DC-CRN for pitch tracking.
- **stft.py**
- **utils.py**
- **test.py**: Test script.
- **test_reverb_ref.sh**: A shell script for testing the model with reverberant speech as a reference.
- **test_anechoic_ref.sh**: A shell script for testing the model with anechoic speech as a reference.
- **environment.yml**

## Setup Instructions

To set up the environment and run the project, follow these steps:

1. Install dependencies using Conda:
   ```bash
   conda env create -f environment.yml
   conda activate multipitch
   ```

2. Run the test scripts:
   - For reverberant reference:
     ```bash
     bash test_reverb_ref.sh
     ```
   - For anechoic reference:
     ```bash
     bash test_anechoic_ref.sh
     ```