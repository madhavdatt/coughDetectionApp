# streamlit_audio_recorder by stefanrmmr (rs. analytics) - version January 2023

from coughDetectionModel import predict_xTest
import os
import streamlit as st
from st_custom_components import st_audiorec
import wave


# DESIGN implement changes to the standard streamlit UI/UX
# --> optional, not relevant for the functionality of the component!
st.set_page_config(page_title="streamlit_audio_recorder")
# Design move app further up and remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
            unsafe_allow_html=True)
# Design change st.Audio to fixed height of 45 pixels
st.markdown('''<style>.stAudio {height: 45px;}</style>''',
            unsafe_allow_html=True)
# Design change hyperlink href link color
st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
            unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
            unsafe_allow_html=True)  # lightmode


rec_sound_filepath = "/Users/madhavdatt/CSE5349/coughDetectionApp/rec_sound.wav"


def audiorec_demo_app():

    # TITLE and Creator information
    st.title("Team 2: Cough Classification")
    st.markdown('Implemented by:')
    st.markdown('-[Madhav Datt](https://www.linkedin.com/in/madhavanildatt/)')
    st.markdown('-[Sanjivni Rana](https://www.linkedin.com/in/sanjivnirana/)')
    st.markdown('-[Baby Mahitha Thumma](https://www.linkedin.com/in/baby-mahitha-thumma/)')
    st.markdown('view project source code on [GitHub](https://github.com/madhavdatt/coughDetectionApp)')
    st.write('\n\n')

    # TUTORIAL: How to use STREAMLIT AUDIO RECORDER?
    # by calling this function an instance of the audio recorder is created
    # once a recording is completed, audio data will be saved to wav_audio_data


    wav_bytes = st_audiorec() # tadaaaa! yes, that's it! :D

    # add some spacing and informative messages
    col_info, col_space = st.columns([0.57, 0.43])
    with col_info:
        st.write('\n')  # add vertical spacer
        st.write('\n')  # add vertical spacer
    

    if wav_bytes is not None:

        # fc.LeftRightCheck.run()
        # display audio data as received on the Python side

        f = wave.open("rec_sound.wav", "wb")
        f.setnchannels(1)
        f.setsampwidth(4)
        f.setframerate(44100)
        f.writeframes(wav_bytes)
        f.close()

        result = predict_xTest(rec_sound_filepath)
        st.write("# {}".format(result))


if __name__ == '__main__':
    # call main function
    audiorec_demo_app()