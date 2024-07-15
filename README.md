# Sign-language to text and speech convertor
### Overview
Sign language is an essential tool for the deaf and dumb community to communicate effectively with the world. To enhance this communication, I developed a real-time Sign Language to Text and Speech Converter. This project leverages OpenCV and advanced machine learning techniques, primarily Convolutional Neural Networks (CNNs) and Random Forest classifiers. The system captures human gestures via a camera and identifies hand landmarks using MediaPipe, which employs CNNs. The data is then trained and tested using Random Forest classifiers. The recognized gestures are first converted to text, which is subsequently transformed into speech using pyttsx3. This innovative solution aims to bridge communication gaps and facilitate seamless interaction for individuals with hearing and speech impairments.

The result of the project is 

https://github.com/user-attachments/assets/f78ab689-af63-4a6e-bf56-66ef77eb3c14


### Inroduction
Sign language is a crucial tool for individuals who are deaf and dumb, allowing them to communicate effectively through hand signs since they cannot speak. Communication involves the exchange of messages, which can be done through signals, speech, or text. For those who are deaf and dumb, sign language simplifies communication. However, not everyone understands sign language. To bridge this gap, I developed a system that can detect hand signs representing letters, convert these signs into text, and subsequently transform the text into speech. This makes communication more accessible and seamless for individuals with hearing and speech disabilities.
The below attached photo consists of the Aplhabet sign Language

![Screenshot (57)](https://github.com/user-attachments/assets/a04267ea-f89b-4219-9f41-bb3d873eb4b9)
and open plam for spaces as shown in the video.

### Installation
You can clone this project into the visual studio code 
and then you have to run the test.py file then you can will get the screen where you can show the signs and that will be converted to sentence after getting the screen you can use it got get sentences by showing signs and if you want to clear thescreen you can press 'c' and if you want the sentnce you can press 's' and if you want to quit from the program you can click 'q'.


This software can be used not only for recognizing individual letter signs but also for gestures like "thank you" and "hello". To use it for additional gestures, follow these steps:

Collect Data: Run the data collection script to capture and store the new hand signs in the data directory. Adjust the number of classes to match the number of gestures you want to include.

![Screenshot (59)](https://github.com/user-attachments/assets/aa60c171-3e03-416a-8269-c8316d86ebe9)

Process Data: Execute the create_data script to process the collected data.
Train Model: Use the train.py script to train the model with the processed data.
Update Label Dictionary in test file: Modify the label dictionary to reflect the new gestures. For example, if the gesture for "hi" is stored in class 0, update the dictionary entry from 0: 'A' to 0: 'hi'.

![Screenshot (60)](https://github.com/user-attachments/assets/9e2f640d-23bf-4ef0-923d-654fb4c0d013)

By following these steps, you can extend the software to recognize and convert a variety of hand signs into text and speech.

