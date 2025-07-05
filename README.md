# MM-ORIENT: A Multimodal-Multitask Framework for Semantic Comprehension

The multimodal learning methods have constantly focused on acquiring a proficient joint multimodal representation. However, the intricate fusion techniques employed to create multimodal features result in the neglect of discriminative information contained within the monomodal features. Moreover, monomodal representation inherently contains noise, which influences latent multimodal representations when these representations are obtained through explicit cross-modal interaction among different modalities. To this end, we propose a multimodal-multitask framework with cross-modal relation and hierarchical interactive attention that is effective for multiple tasks.

## Project Structure

```
MM-ORIENT/
├── src/                               
│   ├── __init__.py                    
│   └── model.py                       
├── utils/                             
│   ├── __init__.py                    
│   └── label_utils.py                 
├── scripts/                           
│   ├── augmentation/                  
│   │   ├── few_shot_prompting.py
│   │   └── image_transformation.py
│   ├── feature_extraction/            
│   │   ├── img_mrcnnx152_features.py
│   │   ├── imgtxt_clip_features.py
│   │   └── txt_bert_features.py
│   └── preprocessing/                 
│       ├── mask_inpaint.py
│       └── preprocess_text.py
├── requirements.txt                   
└── README.md                          
```


## Running the Model
```python
from src.model import train_and_eval
from utils.label_utils import get_train_labels, get_test_labels

# Load your data and run training
# See src/model.py for detailed usage
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