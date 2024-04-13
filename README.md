# Advanced Sarcasm Detection with Transformer Models

## Abstract
This repository is an extension of ongoing research into the detection of sarcasm using transformer-based models. This project explores the application of advanced deep learning techniques to enhance sarcasm detection across diverse textual datasets. Utilizing state-of-the-art models like BERT and RoBERTa, this study aims to refine the understanding and prediction of sarcasm, addressing the challenge posed by its subtle and context-dependent nature.

## Motivation
Sarcasm detection remains a challenging area in natural language processing due to its reliance on contextual cues and cultural nuances. Effective detection can significantly improve the interaction between humans and AI, particularly in fields like sentiment analysis and automated response systems.

## Models
### BERT
We employ the BERT (Bidirectional Encoder Representations from Transformers) model extensively in our experiments. BERT's ability to process words in relation to all other words in a sentence (rather than sequentially) allows it to capture the contextual nuances essential for accurate sarcasm detection.

## Data
The primary dataset used in this project is the Isarcasm dataset (Oprea and Magdy, 2019), which is designed specifically for sarcasm detection.

### Data Cleaning
Cleaning text data is crucial to remove noise and reduce the dimensionality of the dataset, which can enhance the performance of models.

1. **Lowercasing**: Convert all characters in the text to lowercase to maintain uniformity.
2. **Removing User Tags**: Eliminate any user mentions in the text, typically found in tweets or other social media posts.
3. **Removing Hashtags**: Strip out words prefixed with '#' which are commonly used in social media.
4. **Removing URLs**: Get rid of any web links as they are not useful for text analysis.
5. **Removing Punctuations**: Delete characters that are marks of punctuation since they don't contribute to word meanings.
6. **Removing Non-Alphabetic Characters**: Exclude any numbers and special characters from the text.
7. **Removing Extra Spaces**: Reduce multiple spaces to a single space and remove leading and trailing spaces to clean up the text formatting.
8. **Stemming**: Reduce words to their root form, simplifying variations of the same word (e.g., "running" to "run").
9. **Lemmatization**: Convert words into their base form according to their part of speech, improving the contextual accuracy of text data.
10. **Removing Stopwords**: Filter out commonly used words (such as “the”, “a”, “an”, “in”) that do not add significant meaning to the text.
11. **De-emojifying**: Remove emojis from text as they can introduce ambiguity in text analysis and are not handled uniformly by text processing models.


## Data Augmentation Using GPT-2
###
We enhance our dataset using generative models to increase the diversity and volume of training data. This section outlines the use of Generative Pre-trained Transformer 2 (GPT-2) to augment our dataset with synthetic sarcastic expressions. The aim is to enhance the diversity and volume of our training set, thereby improving the accuracy and robustness of our sarcasm detection models.

### Process Overview

#### 1. **Model Selection**
- **Model**: GPT-2 Medium
- **Justification**: Selected for its advanced linguistic capabilities and balance between performance and computational demands, making it ideal for generating nuanced sarcastic sentences.

#### 2. **Training and Fine-Tuning**
- **Source Material**: The model was fine-tuned on a diverse corpus collected from sources known for their rich sarcastic content, including social media and literary texts.
- **Objective**: To adapt GPT-2's generative capabilities to better reflect the stylistic and contextual nuances of sarcastic language.

#### 3. **Data Generation**
- **Initialization**: Generation processes are started with common sarcastic seed phrases to guide the output.
- **Control Measures**: Mechanisms are in place to ensure the generated text adheres to logical and stylistic consistency with existing data.
- **Post-Processing**: All generated text undergoes a stringent quality check to remove any repetitive, irrelevant, or low-quality outputs.

#### 4. **Integration with Existing Data**
- **Method**: Synthetic sentences are carefully integrated with the original dataset, maintaining balanced proportions to prevent biases.
- **Stratification**: Careful blending is performed to ensure that the presence of augmented data complements rather than overwhelms the original dataset.


## Experimental Setup

### Preprocessing Techniques
We explore a variety of preprocessing strategies to optimize model performance. These include tokenization, normalization, and noise reduction (e.g., removal of URLs and user mentions).

### Training
Models are fine-tuned on the augmented dataset using a cross-entropy loss function, optimized with Adam optimizer. Training configurations are meticulously documented to ensure reproducibility.

## Results


## Discussion
The results underscore the effectiveness of RoBERTa over BERT in sarcasm detection when trained under optimized conditions. Our findings suggest that tailored pre-training and larger mini-batches contribute significantly to performance improvements in detecting sarcasm.

## Conclusion and Future Work
This project highlights the potential of transformer-based models in understanding complex linguistic constructs like sarcasm. Future research will explore unsupervised learning techniques and the integration of multimodal data to further enhance sarcasm detection capabilities.

## Repository Structure

- `/data`: Scripts and links to datasets.
- `/models`: Model training scripts and configuration files.
- `/results`: Output logs and performance metrics.
- `/notebooks`: Jupyter notebooks for exploratory data analysis and results visualization.
- `/augmentation`: Scripts and models for data augmentation using GPT-2.

## How to Use This Repository

1. Clone the repository to your local machine.
2. Install dependencies listed in `requirements.txt`.
3. Explore the Jupyter notebooks for a hands-on look at the data and model outputs.
4. Train the models using the scripts provided in the `/models` directory.


Feel free to contribute to this repository by suggesting improvements or by extending the range of experiments to cover more aspects of sarcasm detection.
