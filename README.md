A machine learning-based application to detect and classify fake news articles using NLP techniques and classification models.
Table of Contents

Overview
Features
Installation
Usage
Project Structure
Model Details
Technologies
Results
Future Improvements
License

Overview
This project implements a fake news detection system that analyzes news articles and determines their authenticity using machine learning. The classifier is trained on a labeled dataset of real and fake news articles to identify patterns and linguistic indicators of misinformation.
Features

Text Classification: Classifies news articles as real or fake
Multiple Models: Includes both basic and enhanced machine learning models
Web Application: Python Streamlit Interface
TF-IDF Vectorization: Uses term frequency-inverse document frequency for feature extraction
Pre-trained Models: Includes serialized models for quick deployment.
Data Set : From Kaggle

Enter the article text into the web application
Click the "Analyze" button
View the classification result (Real or Fake)

Model Details
Training Data

Dataset: Labeled fake news articles combined with real news
Format: CSV file with columns for text content and labels

Approach

Text Preprocessing: Clean and prepare text data
Vectorization: Convert text to numerical features using TF-IDF
Classification: Using logical regression
Serialization: Save models for production use

Models Included

fake_news_model.pkl: Binary classifier (Real vs Fake)
enhanced_model_multi.pkl: Enhanced classifier with additional features

Technologies

Python 3: Programming language
scikit-learn: Machine learning library
pandas: Data manipulation and analysis
Streamlit: Web framework
pickle: Model serialization
NumPy: Numerical computing


Future Improvements

 Add deep learning models (LSTM, BERT)
 Implement real-time fact-checking integration
 Expand to multi-language support
 Add confidence scores to predictions
 Create API endpoint for external integrations
 Build dashboard with analytics
 Improve model accuracy with more training data

Owner :
Simrithi
This project is licensed under the MIT License - see the LICENSE file for details.
Author
Simrithi
