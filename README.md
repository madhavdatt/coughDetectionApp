# coughDetectionApp

Download ESC-50 Data from below repository
- https://github.com/karolpiczak/ESC-50

Make below chnages before running app:
- In 'st_app_main.py'
  - change variable 'rec_sound_filepath' to a local path for your recorded audio
  
- In 'coughDetectionModel.py':
  - change variable 'trainingDataDirectory' to point to local path where contents of ESC-50 are stored
  - change variable 'xTest_path' to point to local path for the 'stockCoughSound.wav' file

- Run st_app_main.py
