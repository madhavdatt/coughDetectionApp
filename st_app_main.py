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
    st.title('streamlit audio recorder')
    st.markdown('Implemented by '
        '[Stefan Rummer](https://www.linkedin.com/in/stefanrmmr/) - '
        'view project source code on '
        '[GitHub](https://github.com/stefanrmmr/streamlit_audio_recorder)')
    st.write('\n\n')

    # TUTORIAL: How to use STREAMLIT AUDIO RECORDER?
    # by calling this function an instance of the audio recorder is created
    # once a recording is completed, audio data will be saved to wav_audio_data

    try:
        os.remove(rec_sound_filepath)
    except OSError:
        pass

    wav_bytes = st_audiorec() # tadaaaa! yes, that's it! :D

    # add some spacing and informative messages
    col_info, col_space = st.columns([0.57, 0.43])
    with col_info:
        st.write('\n')  # add vertical spacer
        st.write('\n')  # add vertical spacer
        st.write('The .wav audio data, as received in the backend Python code,'
                 ' will be displayed below this message as soon as it has'
                 ' been processed. [This informative message is not part of'
                 ' the audio recorder and can be removed easily] 🎈')
    

    if wav_bytes is not None:

        # fc.LeftRightCheck.run()
        # display audio data as received on the Python side


        f = wave.open("rec_sound.wav", "w")
        # 2 Channels.
        f.setnchannels(1)
        # # 2 bytes per sample.
        f.setsampwidth(1)
        f.setframerate(22050)
        f.writeframes(wav_bytes)
        f.close()

        # wave_obj = sa.WaveObject.from_wave_file(rec_sound_filepath)
        # play_obj = wave_obj.play()
        # play_obj.wait_done()
        # model = runCD_Train()
        result = predict_xTest(rec_sound_filepath)
        st.write(result)

        # st.write(prediction)

        # col_playback, col_space = st.columns([0.58,0.42])
        # with col_playback:
        #     abc = st.audio(wav_audio_data, format='audio/wav')


if __name__ == '__main__':
    # call main function
    audiorec_demo_app()
