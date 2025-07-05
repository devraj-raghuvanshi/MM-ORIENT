# MM-ORIENT: A Multimodal-Multitask Framework for Semantic Comprehension

The multimodal learning methods have constantly focused on acquiring a proficient joint multimodal representation. However, the intricate fusion techniques employed to create multimodal features result in the neglect of discriminative information contained within the monomodal features. Moreover, monomodal representation inherently contains noise, which influences latent multimodal representations when these representations are obtained through explicit cross-modal interaction among different modalities. To this end, we propose a multimodal-multitask framework with cross-modal relation and hierarchical interactive attention that is effective for multiple tasks.

## Project Structure

```
MM-ORIENT-main/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization
│   └── model.py                 # Main model implementation
├── utils/                        # Utility functions
│   ├── __init__.py              # Package initialization
│   └── label_utils.py           # Label processing utilities
├── scripts/                      # Processing scripts
│   ├── augmentation/            # Data augmentation scripts
│   │   ├── few_shot_prompting.py
│   │   └── image_transformation.py
│   ├── feature_extraction/      # Feature extraction scripts
│   │   ├── img_mrcnnx152_features.py
│   │   ├── imgtxt_clip_features.py
│   │   └── txt_bert_features.py
│   └── preprocessing/           # Data preprocessing scripts
│       ├── mask_inpaint.py
│       └── preprocess_text.py
├── notebooks/                   # Jupyter notebooks
│   └── model.ipynb             # Original model notebook
├── setup.py                     # Package setup configuration
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```


## Usage

### Running the Model
```python
from src.model import train_and_eval
from utils.label_utils import get_train_labels, get_test_labels

# Load your data and run training
# See src/model.py for detailed usage
```

### Feature Extraction
```bash
# Extract CLIP features
python scripts/feature_extraction/imgtxt_clip_features.py

# Extract BERT features
python scripts/feature_extraction/txt_bert_features.py

# Extract MRCNN features
python scripts/feature_extraction/img_mrcnnx152_features.py
```

### Data Processing
```bash
# Preprocess text data
python scripts/preprocessing/preprocess_text.py

# Apply data augmentation
python scripts/augmentation/few_shot_prompting.py
python scripts/augmentation/image_transformation.py
```


## Tasks Supported

- **Task A**: Overall sentiment classification (positive, negative, neutral)
- **Task B**: Binary classification for humor, sarcasm, offense, motivation
- **Task C**: Multi-class classification with intensity levels

## Data Paths Configuration

Update the placeholder paths in `src/model.py` with your actual data paths:

```python
# Training data
dataframe = pd.read_csv("path/to/dataset/train_labels.csv")
img_glob = np.load("path/to/clip_features/image_clip_train.npy")
text_glob = np.load("path/to/clip_features/text_clip_train.npy")

# Monomodal features
img_monomodal_features = np.load("path/to/monomodal_features/image_features_train.npy")
text_monomodal_features = np.load("path/to/monomodal_features/text_features_train.npy")

# Additional features
toxicity = np.load("path/to/additional_features/roberta_features_train.npy")
nrclex = np.load("path/to/additional_features/nrclex_features_train.npy")
stanford = np.load("path/to/additional_features/corenlp_features_train.npy")
```


## Code Structure
- `src/`: Main model implementation
- `utils/`: Reusable utility functions
- `scripts/`: Individual processing scripts


## Requirements

- Python >= 3.7
- TensorFlow >= 2.8.0
- TensorFlow Hub >= 0.12.0
- Keras >= 2.8.0
- Pandas >= 1.3.0
- NumPy >= 1.21.0
- Scikit-learn >= 1.0.0
- Spektral >= 1.2.0
- Scikit-image >= 0.19.0

For complete dependencies, see `requirements.txt`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.