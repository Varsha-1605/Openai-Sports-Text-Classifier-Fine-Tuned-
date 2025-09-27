# Sports Text Classifier ⚾🏒

A fine-tuned GPT-3.5-turbo model for classifying sports-related text as either **Baseball** or **Hockey**, complete with a comprehensive evaluation framework and interactive Streamlit web application.

## 🎯 Project Overview

This project demonstrates the complete machine learning pipeline for text classification using OpenAI's fine-tuning capabilities:

1. **Data Preparation**: Using the 20 Newsgroups dataset
2. **Model Fine-tuning**: Training GPT-3.5-turbo on sports text
3. **Comprehensive Evaluation**: Testing on training, validation, and unseen test sets
4. **Edge Case Analysis**: Evaluating model robustness on ambiguous inputs
5. **Interactive Web App**: Streamlit interface for real-time classification

## 🚀 Key Features

- **High Accuracy**: 99-100% accuracy on sports classification tasks
- **Robust Evaluation**: Multi-tier testing including edge cases
- **Interactive Interface**: User-friendly Streamlit web application
- **Batch Processing**: Handle multiple texts simultaneously
- **Analytics Dashboard**: Visualization of prediction patterns
- **Real-time Testing**: Immediate feedback on model predictions

## 📊 Model Performance

### Training Results
- **Training Accuracy**: 100% (240 examples)
- **Validation Accuracy**: 100% (60 examples)
- **Test Accuracy**: 99% (100 unseen examples)

### Evaluation Metrics
```
              precision    recall  f1-score   support
    baseball       0.98      1.00      0.99        49
      hockey       1.00      0.98      0.99        51
    accuracy                           0.99       100
   macro avg       0.99      0.99      0.99       100
weighted avg       0.99      0.99      0.99       100
```

## 🛠 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Required Python packages

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sports-text-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install openai scikit-learn pandas streamlit plotly seaborn matplotlib
   ```

3. **Set up your OpenAI API key**
   - Get your API key from [OpenAI Platform](https://platform.openai.com/)
   - Keep it secure for use in the application

## 📁 Project Structure

```
sports-text-classifier/
│
├── README.md                 # This file
├── app.py                   # Streamlit web application
├── training_notebook.ipynb  # Complete training and evaluation code
├── sports_train_small.jsonl # Training data (generated)
├── sports_valid_small.jsonl # Validation data (generated)
└── requirements.txt         # Python dependencies
```

## 🏃‍♂️ Quick Start

### 1. Run the Streamlit App

```bash
streamlit run app.py
```

### 2. Configure the Application

1. Enter your OpenAI API key in the sidebar
2. Use the default fine-tuned model ID: `ft:gpt-3.5-turbo-0125:personal::CK2Uuq9e`
3. Click "Connect to Model"

### 3. Start Classifying

- **Single Prediction**: Enter text and get instant classification
- **Batch Processing**: Upload files or enter multiple texts
- **Analytics**: View prediction history and patterns
- **Test Cases**: Evaluate model on predefined edge cases

## 🔬 Model Training Process

### Data Preparation
```python
# Load 20 Newsgroups sports categories
categories = ['rec.sport.baseball', 'rec.sport.hockey']
sports_dataset = fetch_20newsgroups(subset='train', categories=categories)

# Create reduced dataset (300 examples, max 1500 chars)
# Split into training (80%) and validation (20%)
```

### Fine-tuning Configuration
- **Base Model**: `gpt-3.5-turbo-0125`
- **Training Examples**: 240
- **Validation Examples**: 60
- **Epochs**: 2
- **Format**: ChatML conversation format

### Training Data Format
```json
{
  "messages": [
    {"role": "user", "content": "The pitcher threw a fastball..."},
    {"role": "assistant", "content": "baseball"}
  ]
}
```

## 📈 Evaluation Framework

### 1. Training Set Evaluation
- Perfect accuracy on training data
- Validates model learning

### 2. Validation Set Evaluation
- 100% accuracy on held-out validation set
- Confirms no overfitting

### 3. Unseen Test Set Evaluation
- 99% accuracy on completely new data
- Demonstrates generalization capability

### 4. Edge Case Analysis
Tests model robustness on ambiguous inputs:
- Generic sports phrases
- Ambiguous contexts
- Out-of-domain scenarios

## 🎮 Web Application Features

### Single Prediction Mode
- Real-time text classification
- Confidence scoring
- Token usage tracking
- Random example generator

### Batch Processing
- Upload CSV or text files
- Process multiple texts simultaneously
- Download results
- Progress tracking

### Analytics Dashboard
- Prediction distribution charts
- Timeline visualization
- Token usage statistics
- Recent predictions history

### Test Cases
- Predefined test categories
- Clear vs. ambiguous examples
- Batch testing capabilities
- Performance comparison

## 🔍 Edge Case Findings

The model shows interesting behavior on ambiguous inputs:

| Input | Prediction | Notes |
|-------|------------|-------|
| "The player scored in the final seconds." | Hockey | Reasonable sports guess |
| "He was injured during the match." | Soccer | Out-of-domain hallucination |
| "The referee made a controversial call." | Soccer | Unexpected label |
| "The match ended in overtime." | Hockey | Good contextual guess |

### Recommendations for Improvement
1. **Add "uncertain" class** for ambiguous inputs
2. **Include label constraints** in fine-tuning
3. **Expand training diversity** with edge cases
4. **Post-processing validation** to handle unexpected labels

## 🛡 Robustness Features

- **Error handling**: Graceful handling of API failures
- **Out-of-domain detection**: Identifies unexpected predictions
- **Post-processing**: Maps invalid labels to "uncertain"
- **Rate limiting**: Controlled API usage
- **Progress tracking**: User feedback during batch processing

## 💡 Usage Examples

### Basic Classification
```python
classifier = SportsClassifier(api_key, model_id)
result = classifier.predict("The goalie made an incredible save.")
# Returns: {'prediction': 'hockey', 'confidence': 'high'}
```

### Batch Processing
```python
texts = ["The pitcher struck out the batter.", "The puck hit the post."]
results = classifier.batch_predict(texts)
# Returns: List of prediction results
```

## 📋 Requirements

```txt
openai>=1.0.0
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=2.0.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 🔐 Security Notes

- Never commit API keys to version control
- Use environment variables or secure input methods
- Monitor API usage to avoid unexpected charges
- Implement rate limiting for production use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for the fine-tuning platform and GPT models
- **20 Newsgroups Dataset** for providing the training data
- **Streamlit** for the excellent web app framework
- **Scikit-learn** for evaluation metrics and data handling


## 🔄 Future Improvements

- [ ] Add support for more sports categories
- [ ] Implement confidence scoring improvements
- [ ] Add model comparison features
- [ ] Include data augmentation techniques
- [ ] Deploy to cloud platforms
- [ ] Add automated testing pipeline

---

**Made with ❤️ by Varsha Dewangan**

For more details about the training process and evaluation results, check out the complete Jupyter notebook included in this repository.
