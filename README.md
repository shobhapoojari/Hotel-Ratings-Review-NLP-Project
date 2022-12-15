Hotel Rating Classification-Internship Project
In this project, our goal is to examine how travelers are communicating their positive and negative experiences in online platforms for staying in a specific hotel. Our major objective is what are the attributes that travelers are considering while selecting a hotel. With this manager can understand which elements of their hotel influence more in forming a positive review or improves hotel brand image.

Datset
Dataset contains customer reivew and ratings to the perticular reviews.

Requirements
python
pandas
sklearn
numpy
nltk
neattext
pickle(save model)
Approach
We have approached in 2 way for text preprocessing

neattext preprocessing method
using nltk preprocessing method
1) Using neattext library
NeatText is a simple Natural Language Processing package for cleaning text data and pre-processing text data.

Cleaning of unstructured text data
Reduce noise [special characters,stopwords]
Reducing repetition of using the same code for text preprocessing
2) Using NLTK(Natural language toolkit) library
There are many libraries and algorithms used to deal with NLP-based problems. A regular expression(re) is mostly used library for text cleaning. NLTK(Natural language toolkit) and spacy are the next level library used for performing Natural language tasks like removing stopwords, named entity recognition, part of speech tagging, phrase matching, etc.

Model Building
We have created the model like Logistic Regression,Random Forest and XGboost, out of these we have selected LR is the final model.Our model rendered accuracy of 89.73%,Saved this model using pickle.

Deployment
To deploy this project run using streamlit
