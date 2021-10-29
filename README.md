# Natural-Language-Process-Team-Project
2021 COSE461 Team 14   
[@noparkee](https://github.com/noparkee) [@jooeun9199](https://github.com/jooeun9199) [@Xenor99](https://github.com/Xenor99)

# SER with TextEmbedding
감정 분류를 하자!

## Environment
- Python3
- PyTorch

## Data Set
[IEMOCAP](https://sail.usc.edu/iemocap/)

## Reference Paper
[Empirical Interpretation of Speech Emotion Perception with Attention Based Model for Speech Emotion Recognition](http://www.interspeech2020.org/uploadfile/pdf/Thu-2-2-8.pdf)
- 현재 SOTA
- 음성만 이용해서 감정 분류
- BiLSTM + attention

[Multimodal Speech Emotion Recognition and Ambiguity Resolution](https://arxiv.org/pdf/1904.06022v1.pdf)
- audio와 text 데이터 모두 이용
- audio data의 경우, 8개의 hand-crafted features 사용
- textual data의 경우, TF-IDF 이용
- ML models, simple LSTM, etc.
