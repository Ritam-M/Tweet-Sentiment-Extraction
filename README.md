# Tweet-Sentiment-Extraction

Very often we want to identify the portion of the sentence which tries to convey some meaning into the sentence, and this project is a use case of the same. Introducing the Kaggle Competition, Tweet-Sentiment Extraction to identify phrases which are happy, neutral or sad.

I used Bert Architecture, using the template of Question-Answering Models, and reduced the problem to finding start and end indexes to classify the portion of the sentence responsible for a particular tweet. Metric used in the competition was Jaccard, and it is the equivalent of IoU score used in Object Detection Tasks.

Further Experiments: Using Roberta, Albert and make appropriate changes in the preprocessing function

Scope of Improvements: Using Text Summarization to shorten the length of the entire statement, and expanding the problem to large documents and possible applications into Visual Question Answering.
