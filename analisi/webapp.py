import streamlit as st
import pandas as pd
import numpy as np

import streamlit.components.v1 as components
import random
from glove import Glove
from Polarization.w2v_polarity import w2vPolarity
from Polarization.glove_polarity import glovePolarity
from Polarization.bert_polarity import bertPolarity
from gensim.models import Word2Vec
from glove import Glove
import numpy as np
import csv
import ast
#from glove import  Glove

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def main():
    # Register your pages
    pages = {
        "Trump": page_first,
        "Clinton": page_second,
        "Polarization": page_third,
        "Word2vec": page_fourth
    }

    st.sidebar.title("Menù")

    # Widget to select your page, you can choose between radio buttons or a selectbox
    page = st.sidebar.selectbox("Select your page", tuple(pages.keys()))
    #page = st.sidebar.radio("Select your page", tuple(pages.keys()))
    # Display the selected page with the session state
    pages[page]()

def page_first():

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    def icon(icon_name):
        st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

    local_css("style.css")

    st.title("Trump")
    selected = st.text_input("", "Search by label...")
    print("SELECTED:" ,selected)
    button_clicked = st.button("search")
    #st.write(button_clicked)
    entity_trump = load_data('entities_trump.csv')
    list_labels = entity_trump["label"]
    tweet_marked= pd.read_csv('highlight_trump.csv')
    tweets = tweet_marked["text"]
    col1,colm,col2 = st.beta_columns([4,1,3])
    with col1:
        st.header("Tweets")
        if(button_clicked):
                for index, row in entity_trump.iterrows():
                    if selected == row["label"]:
                        plain = ast.literal_eval(row["plain_text"])
                for ent in plain:
                    for tweet in tweets:
                        if ent in tweet:
                            try:
                                st.markdown(tweet, unsafe_allow_html=True)
                            except:
                                st.write("")
        else:
            for tweet in tweets:
                try:
                    st.markdown(tweet, unsafe_allow_html=True)
                except:
                    st.write("")             
                    
            
                    
            

    with colm:
        st.write("")

    with col2:
        person_df = pd.read_csv('colab/person_entity.csv')
        location_df = pd.read_csv('colab/location_entity.csv')
        organisation_df =  pd.read_csv('colab/organisation_entity.csv')
        st.markdown("<h3 class= 'person'>Person</h3>",unsafe_allow_html=True )
        st.dataframe(person_df)
        st.markdown("<h3 class = 'location'>Location</h3>", unsafe_allow_html=True)
        st.dataframe(location_df)
        st.markdown("<h3 class = 'organisation'>Organisation</h3>", unsafe_allow_html=True)
        st.dataframe(organisation_df)
   # t = "<div>Hello there my <span class='highlight blue'>name <span class='bold'>yo</span> </span> is <span class='highlight red'>Fanilo <span class='bold'>Name</span></span></div>"

    #st.markdown(t, unsafe_allow_html=True)
   
def page_second():
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    def icon(icon_name):
        st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)
    

    local_css("style.css")
    st.title("Clinton")
    selected = st.text_input("", "Search by entity label...")
    print("SELECTED:" ,selected)
    button_clicked = st.button("search")
    #st.write(button_clicked)
    entity_clinton = load_data('entities_clinton.csv')
    list_labels = entity_clinton["label"]
    tweet_marked= pd.read_csv('highlight_clinton.csv')
    tweets = tweet_marked["text"]
    col1, colm ,col2 = st.beta_columns([4,1,3])
    if(button_clicked):
                for index, row in entity_clinton.iterrows():
                    if selected == row["label"]:
                        plain = ast.literal_eval(row["plain_text"])
                for ent in plain:
                    for tweet in tweets:
                        if ent in tweet:
                            try:
                                st.markdown(tweet, unsafe_allow_html=True)
                            except:
                                st.write("")
    else:
        for tweet in tweets:
            try:
                st.markdown(tweet, unsafe_allow_html=True)
            except:
                st.write("")         


    with col1:
        st.header("Tweets")
        if(button_clicked):
                for index, row in entity_clinton.iterrows():
                    if selected == row["label"]:
                        plain = ast.literal_eval(row["plain_text"])
                for ent in plain:
                    for tweet in tweets:
                        if ent in tweet:
                            try:
                                st.markdown(tweet, unsafe_allow_html=True)
                            except:
                                st.write("")
        else:
            for tweet in tweets:
                try:
                    st.markdown(tweet, unsafe_allow_html=True)
                except:
                    st.write("")
    with colm:
        st.write("")
    with col2:
        person_df_clinton = pd.read_csv('colab/person_entity_clinton.csv', )
        location_df_clinton = pd.read_csv('colab/location_entity_clinton.csv')
        organisation_df_clinton =  pd.read_csv('colab/organisation_entity_clinton.csv')
        st.markdown("<h3 class= 'person'>Person</h3>",unsafe_allow_html=True )
        st.dataframe(person_df_clinton)
        st.markdown("<h3 class = 'location'>Location</h3>", unsafe_allow_html=True)
        st.dataframe(location_df_clinton)
        st.markdown("<h3 class = 'organisation'>Organisation</h3>", unsafe_allow_html=True)
        st.dataframe(organisation_df_clinton)

    #text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously."

    #nlp = spacy.load("en_core_web_sm")
    #doc = nlp(text)
    #displacy.serve(doc, style="ent")
    #models = ["en_core_web_sm"]
   # nlp = spacy.load("en_core_web_sm")
   # doc = "Sundar Pichai is the CEO of Google."
    #visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
    # ...
  #  t = "<div>Hello there my    <a href='http://dbpedia.org/resource/Alberto_Gonzales'> <span class='highlight blue'>name <span class='bold'>yo</span> </span> </a> is <span class='highlight red'>Fanilo <span class='bold'>Name</span></span></div>"

   # st.markdown(t, unsafe_allow_html=True)
  
def page_third():
    choice = st.radio(
     "Choose embedding",
     ('Trump', 'Clinton',))
    
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
 
    local_css("style.css")

    col1, colm ,col2 = st.beta_columns([4,1,3])
    with col1:
        topic_input = st.text_input('topic vector', "hillaryclinton hillary democratic dem")
        positive_input = st.text_input('positive vector', "good great nice positive love")
        negative_input = st.text_input('negative vector', "bad badly negative false wrong")

        button_polarity = st.button("Polarity")
        print(button_polarity)
    
        topic = topic_input.split()
        positive = positive_input.split()
        negative = negative_input.split()
        
    
        if choice == 'Trump':
            trump_w2v = Word2Vec.load("Polarization/models/trump_w2v.model")
            trump_glove = Glove.load("Polarization/models/trump_glove.model")
            with open('Polarization/models/tokens_trump_bert.csv', newline='') as f:
                reader = csv.reader(f)
                trump_bert_tokens = list(reader)
            trump_bert_emb = np.load('Polarization/models/embedding_trump_bert.npy', allow_pickle=True) 
            if button_polarity:
                positive_count = 0
                w2vec_pol = w2vPolarity(trump_w2v, topic, positive, negative)
                print(w2vec_pol)
                if  w2vec_pol > 0:
                    w2vec_rank = "POSITIVE"
                    style_w2v = "'positive'"
                    positive_count = positive_count + 1
                else:
                    w2vec_rank = "NEGATIVE"
                    style_w2v = "'negative'"

                glove_pol = glovePolarity(trump_glove, topic, positive, negative)
                print(glove_pol)
                if glove_pol > 0:
                    glove_rank = "POSITIVE"
                    style_glove = "'positive'"
                    positive_count = positive_count + 1
                else:
                    glove_rank = "NEGATIVE"
                    style_glove = "'negative'"

                #a = np.loadtxt('./models/embeddings_clinton_bert.csv', delimiter=',')  
                bert_pol = bertPolarity(trump_bert_tokens, trump_bert_emb, topic, positive, negative)
                print(bert_pol)
                if bert_pol > 0 :
                    positive_count = positive_count + 1
                    bert_rank = "POSITIVE"
                    style_bert = "'positive'"
                else:
                    bert_rank = "NEGATIVE"
                    style_bert = "'negative'"
            
                if positive_count > 1:
                    majority_rank = "<h4 style = 'color:green';>POSITIVE</h4>"
                    #st.markdown("<h4 style = 'color:green'>Majority polarity: POSITIVE</h4> ",unsafe_allow_html=True)
                else:
                    majority_rank = "<h4 style = 'color:red';>NEGATIVE</h4>"
                    #st.markdown("<h4 style = 'color:red'; text-align:'center' >Majority polarity: NEGATIVE</h4> ",unsafe_allow_html=True)
                st.markdown("<table><tr><th>Model</th><th>Polarity</th></tr><tr><td>Word2vec</td><td><h4 class = "+style_w2v+">"+w2vec_rank+"</h4></tr><tr><td>Glove polarity</td><td><h4 class = "+style_glove+"> "+glove_rank+"</h4></td></tr><tr><td>Bert polarity</td><td><h4 class = "+style_bert+">"+bert_rank+"</h4></td></tr><tr><td>Majority polarity</td><td>"+majority_rank+"</td></tr></table>", unsafe_allow_html= True)               
                
        elif choice == 'Clinton':
            clinton_w2v = Word2Vec.load("Polarization/models/clinton_w2v.model")
            clinton_glove = Glove.load("Polarization/models/clinton_glove.model")
            with open('Polarization/models/tokens_clinton_bert.csv', newline='') as f:
                reader = csv.reader(f)
                clinton_bert_tokens = list(reader)
            clinton_bert_emb = np.load('Polarization/models/embedding_clinton_bert.npy', allow_pickle=True)
            if button_polarity:
                positive_count = 0
                w2vec_pol = w2vPolarity(clinton_w2v, topic, positive, negative)
                print(w2vec_pol)
                if  w2vec_pol > 0:
                    w2vec_rank = "POSITIVE"
                    style_w2v = "'positive'"
                    positive_count = positive_count + 1
                else:
                    w2vec_rank = "NEGATIVE"
                    style_w2v = "'negative'"

                glove_pol = glovePolarity(clinton_glove, topic, positive, negative)
                print(glove_pol)
                if glove_pol > 0:
                    glove_rank = "POSITIVE"
                    style_glove = "'positive'"
                    positive_count = positive_count + 1
                else:
                    glove_rank = "NEGATIVE"
                    style_glove = "'negative'"

                #a = np.loadtxt('./models/embeddings_clinton_bert.csv', delimiter=',')  
                bert_pol = bertPolarity(clinton_bert_tokens, clinton_bert_emb, topic, positive, negative)
                print(bert_pol)
                if bert_pol > 0 :
                    positive_count = positive_count + 1
                    bert_rank = "POSITIVE"
                    style_bert = "'positive'"
                else:
                    bert_rank = "NEGATIVE"
                    style_bert = "'negative'"
                
                

                
            
            
                if positive_count > 1:
                        majority_rank = "<h4 style = 'color:green';> POSITIVE</h4>"
                        #st.markdown("<h4 style = 'color:green';>Majority polarity: POSITIVE</h4> ",unsafe_allow_html=True)
                else:
                        #st.markdown("<h4 style = 'color:red'; text-align:'center' >Majority polarity: NEGATIVE</h4>",unsafe_allow_html=True)
                        majority_rank = "<h4 style = 'color:red';>NEGATIVE</h4>"              
                st.markdown("<table><tr><th>Model</th><th>Polarity</th></tr><tr><td>Word2vec</td><td><h4 class = "+style_w2v+">"+w2vec_rank+"</h4> </td></tr><tr><td>Glove polarity</td><td><h4 class = "+style_glove+"> "+glove_rank+"</h4></td></tr><tr><td>Bert polarity</td><td><h4 class = "+style_bert+">"+bert_rank+"</h4></td></tr><tr><td>Majority polarity</td><td>"+majority_rank+"</td></tr></table>", unsafe_allow_html= True)               
    with col2:
        st.text_area("Tips", "Trump: \n topic : [immigration illegalimmigration] \n topic: [cnn foxnews thenweyorktimes]\n Clinton: \n topic: [immigration deportation trump]\n topic: [rttvnetwork thenweyorktimes]", height= 300)
def page_fourth():
    col1,colm,col2 = st.beta_columns([4,1,4])
    with col1:
        trump_w2v = Word2Vec.load("Polarization/models/trump_w2v.model")
        vector_tr = st.text_input('trump vector', "clinton")
        compute_tr = st.button("sim trump")
        similarity_tr = trump_w2v.wv.most_similar(vector_tr)
        if compute_tr and similarity_tr:
            st.table(similarity_tr)
    with col2:
        clinton_w2v = Word2Vec.load("Polarization/models/clinton_w2v.model")
        vector_cli = st.text_input('clinton vector', "election")
        compute_cli = st.button("sim clinton")
        similarity_cli = clinton_w2v.wv.most_similar(vector_cli)
        if compute_cli and similarity_cli:
            st.table(similarity_cli)
            print(similarity_cli)
        



 


    
    


   


 

        
         
if __name__ == "__main__":
    main()



