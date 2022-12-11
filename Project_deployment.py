import streamlit as st
import numpy as np
import pandas as pd
import pickle
import altair as alt


with open(file="Hotel_Final_model.pkl",mode="rb") as f:
    model = pickle.load(f)

#function to read the emotion
def predict_emotions(docx):
    results=model.predict([docx] )
    return results

def get_prediction_proba(docx):
    results=model.predict_proba([docx] )
    return results

sentiment_emoji_dict = {"negative", "positive"}

def main():
    st.title('Hotel Rating Classifier App')
    menu=["Home", "Monitor", "About"]
    choice=st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home-rating in text")

        with st.form(key='sentiment_clf_form'):
            raw_text = st.text_area("Please enter your text")
            submit_text = st.form_submit_button(label="Submit")

        if submit_text:
            col1,col2 = st.columns(2)
            prediction=predict_emotions(raw_text)
            probability=get_prediction_proba(raw_text)
            with col1:
                st.success('Original text')
                st.write(raw_text)

                st.success("Prediction")
                ## emoji_icon= sentiment_emoji_dict[prediction[0]]
                st.write("{}".format(prediction[0]))
                st.write("COnfidence: {}".format(np.max(probability)))

            with col2:
                st.success('Prediction Probability')
                st.write(probability)
                proba_df=pd.DataFrame(probability,columns=model.classes_)
                st.write(proba_df.transpose())
                proba_df_clean=proba_df.transpose().reset_index()
                proba_df_clean.columns=["sentiment","probability"]

            fig=alt.Chart(proba_df_clean).mark_bar().encode(x='sentiment', y='probability',color='sentiment')
            st.altair_chart(fig,use_container_width=True)
                

    elif choice == "Monitor":
        st.subheader("Monitor App")
    else:
        st.subheader("About")
        st.write("This is an NLP powered webapp that can predict emotion from text recognition with 87.52 percent accuracy, Many python libraries like Numpy, Pandas, Seaborn, Scikit-learn, Scipy, Streamlit, eli5, lime, altair, streamlit was used. Streamlit was mainly used for the front-end development, Logistic regression model from the scikit-learn library was used to train a dataset containing speeches and their respective emotions. Streamlit was used for storing and using the trained model in the website")
        st.caption('Created by:  Shobha M')

if __name__ == "__main__":
    main()