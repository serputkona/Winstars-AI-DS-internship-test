# Animal Recognition System

This repository contains a dual-modality animal recognition system that combines:
1. **Named Entity Recognition (NER)** to extract animal mentions from text
2. **Image Classification** to identify animals in images

The system can validate whether animals mentioned in a text description match what appears in an accompanying image.

## Project Overview

This project integrates natural language processing and computer vision to create a coherent animal recognition pipeline. The system includes:

- A fine-tuned BERT-based NER model that detects animal mentions in text
- A ResNet50-based image classifier trained to recognize 13 animal categories
- A unified pipeline that combines both components for cross-modal validation

## Animal Classes

The system recognizes the following 13 animal classes:
- Cat
- Bear
- Goose
- Squirrel
- Fox
- Elk
- Flamingo
- Owl
- Frog
- Beaver
- Bee
- Dove
- Ladybug

## Dataset

The image dataset includes:
- 1200-1500 images per class (approximately 15,600-19,500 total images)
- Images sourced from Images.cv and iNaturalist
- Dataset split: 80% training, 20% validation

The NER training data was created from a CSV file containing sentences with animal mentions, tagged with labels for animal entities.

## Installation

```bash
# Clone the repository
git clone https://github.com/serputkona/Winstars-AI-DS-internship-test.git
cd Task2

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
 See requirements.txt
```

## Usage

### Download Required Files

1. Download the Jupyter notebook file: `Task2_pipeline.ipynb`
2. Download the pre-trained models:
   - `animal_ner_model/` directory (containing the NER model files)
   - `animal_classifier_final.pth` (the image classification model)
3. Place all files in the same directory structure as shown in the notebook

### Run the Notebook

1. Open the `Task2_pipeline.ipynb` notebook in Jupyter:
   ```bash
   jupyter notebook Task2_pipeline.ipynb
   ```

2. Make sure the paths to the models are correct in the notebook:
   ```python
   pipeline = AnimalPipeline(
       ner_model_path="animal_ner_model",       # Update if your path is different
       classifier_model_path="animal_classifier_final.pth"  # Update if your path is different
   )
   ```

3. Run all cells in the notebook to load the pipeline

### Example Usage in the Notebook

```python
# Extract animals from text
text = "There is a beautiful fox hiding in the grass."
animals = pipeline.extract_animals_from_text(text)
print(f"Detected animals: {animals}")  # Output: ['fox']

# Classify an animal in an image
image_path = "path/to/your/image.jpg"  # Update with your image path
animal = pipeline.predict_from_image(image_path)
print(f"Predicted animal: {animal}")  # Output: 'fox'

# Verify if the text and image match
is_match = pipeline.is_same_animal(image_path, text)
print(f"Text and image match: {is_match}")  # Output: True
```

## Model Architecture

### NER Model
- Based on pre-trained BERT model
- Fine-tuned for token classification task
- Maps "B-ANIMAL" and "I-ANIMAL" tags to detect animal mentions

### Image Classification Model
- Based on pre-trained ResNet50
- Custom classifier head with the architecture:
  - Linear(2048 → 512) → ReLU → Dropout(0.3)
  - Linear(512 → 256) → ReLU → Dropout(0.2)
  - Linear(256 → 13)
- Two-phase training:
  1. Train only the custom classifier (7 epochs)
  2. Fine-tune with 5 ResNet layers unfrozen (7 epochs)

    
## Project Structure

```
animal-recognition-system/
├── animal_ner_model/               # Saved NER model and tokenizer
├── animal_classifier_final.pth     # Saved image classifier
├── EDA_images.ipynb                # Exploritary data analysis of image dataset
├── EDA_text.ipynb                  # Exploritary data analysis of text dataset
├── README.md                       # This documentation
├── requirements.txt                # Project dependencies
├── Task2_Demo.ipynb                # Pipeline implementation demo
├── Task2_Image.ipynb               # Image classifier training and utilities
├── Task2_NER.ipynb                 # NER model training and utilities
└── Task2_pipeline.ipynb            # Combined pipeline implementation

```
