
import streamlit as st
import pandas as pd
import altair as alt
import keras
from PIL import Image
import numpy as np
from keras.utils.vis_utils import plot_model
import seaborn as sns

df = pd.read_csv('sales_data/vgsales.csv')
#option at the side bar
analysis = st.sidebar.selectbox('Select an Option',['Image Classification','Data Analysis & Visualization'])
# analysis
#title
st.set_option('deprecation.showfileUploaderEncoding', False)
if analysis=='Data Analysis & Visualization':
    st.title('Video Game Sales Analysis')
    # simple description
    st.write('In this dashboard we will analyze the Video Game Sales data from Kaggle. '
             'These charts are interactive')

    st.write('')
    st.write('Mario was my favorite game when i was a kid. ')
    # display media


    image = Image.open('mario-1557240_640.jpg')

    st.image(image, caption='source: https://pixabay.com/photos/mario-luigi-yoschi-figures-funny-1557240/',
             use_column_width=True)

    # 2 types of heading - header and subheader
    st.header('Exploratory Data Analysis')
    # markdown similar to github md and also has support for some cool graphics
    # in form of emojis full list here: https://raw.githubusercontent.com/omnidan/node-emoji/master/lib/emoji.json

    st.markdown('Before we get started with the analysis, lets have a quick look at the raw data :sunglasses:')

    df_sample = df.head()
    df_sample

    st.subheader('Platform wise sales')
    # drop down for unique value from a column
    platform_name = st.selectbox('Select a Platform', options=df.Platform.unique())
    # line chart after applying filter from above drop down menu
    st.write('Sales across years')
    basic_chart = alt.Chart(df.loc[df.Platform == platform_name].groupby(['Platform', 'Year']). \
                            agg({'Global_Sales': 'sum'}).reset_index()).mark_line().encode(
        x='Year',
        y='Global_Sales'
        # legend=alt.Legend(title='Animals by year')
    )
    st.altair_chart(basic_chart)
    # st.line_chart(df.loc[df.Platform==platform_name]['Global_Sales'])

    st.write('Geography wise sales')
    temp = pd.melt(df.loc[df.Platform == platform_name, ['Platform', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', \
                                                         'Other_Sales']], id_vars=['Platform', 'Year'], var_name='Geo',
                   value_name='Sales')
    # sns.barplot(x="Year", y="Sales", hue="Geo", data=temp)
    temp = temp.groupby(['Platform', 'Year', 'Geo']).agg({'Sales': 'sum'}).reset_index()
    stacked_bar = alt.Chart(temp).mark_bar().encode(
        x='Year',
        y='Sales',
        color='Geo'
    )
    st.altair_chart(stacked_bar)
    st.write('I had a bit of hard time, reshaping the data for the above chart. \
    We had to bring the values from multiple sales columns into a single column.'
             ' Luckily Streamlit allows you to create a section for showing code ')




    with st.echo():
        # Code used to unpivot the sales columns into a single column.

        unpivot_df = pd.melt(df.loc[df.Platform == platform_name, ['Platform', 'Year', \
                            'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']],
                             id_vars=['Platform', 'Year'], \
                             var_name='Geo', value_name='Sales')
        unpivot_df_agg = unpivot_df.groupby(['Platform', 'Year', 'Geo']). \
            agg({'Sales': 'sum'}).reset_index()

    st.write(unpivot_df_agg.head())
else:
    st.title('MNIST Image Classification')
    st.write('This is to showcase how quickly image classification apps can be built ')
    # st.header('Identifying digits from Images')
    st.subheader('Please upload an image to identify the digit')
    file_uploader = st.file_uploader("Select an image for Classification", type="png")
    print(file_uploader)
    model = keras.models.load_model('mnist_model', compile=False)
    if file_uploader:
        image = Image.open(file_uploader)
        st.image(image, caption='Selected Image')
    if st.button('Predict'):
        #add warning for image not selected
        image = np.asarray(image)

        pred = model.predict(image.reshape(1,28,28,1))



        import time
        my_bar = st.progress(0)
        with st.spinner('Predicting'):
            time.sleep(2)

        st.write(pred)







