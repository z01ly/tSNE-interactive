import os
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


@st.cache_data
def load_data():
    df = pd.read_pickle(os.path.join('data/tSNE-32-dims', 'embedded-z-reduced.pkl'))
    return df


def main():
    df = load_data()

    st.title("t-SNE Plot Interactive")

    fig = px.scatter(df, x='f0', y='f1', color='label', hover_name='filename')

    current_dict = st.plotly_chart(fig, on_select='rerun', selection_mode='points')

    click_check = current_dict['selection']['point_indices']
    if click_check:
        image_path = current_dict['selection']['points'][0]['hovertext']
        with Image.open(image_path) as img:
            st.image(img, caption=f'{image_path}')

        st.write(current_dict)



if __name__ == "__main__":
    main()
