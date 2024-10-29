Semi-supervised Random Forest Classifier for Voice Activity Detection in Diverse Noisy Environments with Limited Labelled Data.

This Project Aims to perform VAD with only 20% of the labelled data available. The audio data has noisy speech mixed at different SNR values as well as non-speech segments. 
We aim to perform VAD for short utterances.

To run the Project successfully, follow these Steps:
1. Audio Augmentation : Pitch Shift and Time Stretch
2. Audio Overlap : Perform speech and noise overlap at various SNR values (-10 dB to 10 dB)
3. Combined Feature Extraction: Extract MFCC, GTCC, ZCR, Pitch, STE from the speech/non-speech audio segments and store in a csv file
4. Supervised Random Forest: Perform Supervised Random forest VAD to establish baseline
5. SSRF: Semi-supervised random forest classification using different percentages of labelled data with 5-fold cross-validation
