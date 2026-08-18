# Nasij
AI-powered civic platform for reporting infrastructure issues and analyzing citizen feedback in Lebanon
# 🧶 Nasij

Nasij is an AI-powered civic engagement platform designed to help citizens report infrastructure problems and communicate their concerns to municipalities.
The platform combines Computer Vision and Natural Language Processing (NLP) to transform citizen reports and suggestions into structured, actionable information for local authorities.

##The Problem
Municipalities often receive citizen complaints and suggestions through scattered channels such as phone calls, social media, messaging applications, and informal reports.
This makes it difficult to:
- Organize infrastructure complaints
- Identify urgent problems
- Understand citizens' most common concerns
- Prioritize municipal work
- Track reported issues
- Provide a structured view of community needs

Citizens may also have little visibility into how their reports are processed.
Nasij aims to create a clearer bridge between citizens and municipalities using Artificial Intelligence.

##The Solution

Nasij provides three main features:

1) 📷 Report an Infrastructure Issue
Citizens can upload an image of an infrastructure problem.
A trained YOLO object detection model analyzes the image and detects supported civic issues.
The system then automatically creates a maintenance ticket containing:
- Detected issue
- AI confidence
- Location
- Citizen notes
- Submission time
- Ticket status

2) 💬 Submit a Suggestion
Citizens can write suggestions or concerns in English or Arabic.
Nasij analyzes the text and assigns it to one of five civic topics:
- 🗑️ Waste management
- 🕳️ Roads / potholes
- 💡 Street lighting
- 🌳 Green spaces
- 💧 Water supply
The NLP system uses multilingual sentence embeddings and semantic similarity.
A small Lebanese Arabic keyword layer is also used for clear civic expressions such as:
- `زبالة`
- `حفر`
- `انقطاع المي`
- `إنارة`
- `شجر`
This improves the handling of common Lebanese Arabic expressions.

3) 🏛️ Municipality Dashboard
The municipality dashboard automatically summarizes incoming citizen activity.
It displays:
- Number of infrastructure reports
- Number of citizen suggestions
- Most requested civic topic
- Topic frequency table
- Topic frequency chart
- Representative citizen concerns
- Infrastructure maintenance tickets
- AI confidence values
- Ticket status
This allows municipalities to quickly understand which issues are receiving the most attention.

##Artificial Intelligence Components

### Computer Vision
Nasij uses a fine-tuned **YOLOv8 Nano** object detection model.
The model was trained using a public civic-infrastructure dataset containing **5,454 images**.
The dataset contains five classes:
1. Fallen tree
2. Garbage
3. Pothole
4. Streetlight
5. Water leak
The model was fine-tuned for 25 epochs using transfer learning from pretrained YOLOv8 weights.

### Computer Vision Pipeline
Citizen Image
      ↓
YOLOv8
      ↓
Object Detection
      ↓
Issue + Confidence + Bounding Box
      ↓
Maintenance Ticket
      ↓
Municipality Dashboard

## Natural Language Processing NLP
The Natural Language Processing component of Nasij processes citizen suggestions and organizes them into meaningful civic categories. Citizens may submit suggestions in English or Arabic, and each submission is assigned to one of five topics: waste management, roads and potholes, street lighting, green spaces, and water supply.
The system uses the multilingual Sentence Transformer model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2. Sentence Transformers convert text into numerical representations known as embeddings. In this case, each suggestion is represented by a 384-dimensional vector that captures its semantic meaning.
This representation allows the system to identify similarities between sentences even when they use different wording. For example, a suggestion referring to potholes and another referring to damaged roads may still have similar embeddings because they describe closely related civic concerns.
The multilingual nature of the model also makes it suitable for a Lebanese context, where citizen feedback may be written in English, Arabic, or colloquial Lebanese Arabic.

### NLP Dataset:
A balanced evaluation dataset of 30 citizen suggestions was used to assess the NLP methods.
The dataset contained six suggestions for each of the five civic topics. For every category, three suggestions were written in English and three in Arabic or Lebanese Arabic.
The five categories were waste management, roads and potholes, street lighting, green spaces, and water supply.
The known topic associated with each suggestion was retained as an evaluation label. These labels were not provided to the clustering algorithms and were only used afterward to measure how well the discovered clusters corresponded to the intended civic categories.

### NLP Experiments:
Three NLP configurations were evaluated to compare different embedding and clustering approaches.
- -Experiment 1: Multilingual MiniLM with K-Means
The first experiment used embeddings generated by paraphrase-multilingual-MiniLM-L12-v2 together with K-Means clustering.
Since the project focuses on five civic topics, K-Means was configured to produce five clusters.
The clustering achieved a cosine silhouette score of approximately 0.202 and an Adjusted Rand Index of approximately 0.355.
The results showed that several civic categories formed meaningful groups, particularly suggestions related to roads and green spaces. However, some multilingual suggestions were grouped according to linguistic similarity rather than their intended civic category.
Among the clustering configurations evaluated, this approach achieved the strongest overall performance.
- Experiment 2: Multilingual MiniLM with Agglomerative Clustering
The second experiment used the same MiniLM sentence embeddings with Agglomerative Clustering.
Agglomerative Clustering creates a hierarchy of groups by progressively merging similar samples. Cosine distance was used to measure similarity between the sentence embeddings.
This configuration achieved a cosine silhouette score of approximately 0.195 and an Adjusted Rand Index of approximately 0.318.
Its performance was slightly lower than that of K-Means with the same MiniLM embeddings.
- Experiment 3: Multilingual E5 with K-Means
A second multilingual embedding model, multilingual-e5-small, was also evaluated.
The generated embeddings were normalized and clustered using K-Means with five clusters.
This configuration achieved a cosine silhouette score of approximately 0.164 and an Adjusted Rand Index of approximately 0.132.
The MiniLM embeddings therefore produced stronger clustering results for the civic suggestion dataset.
Overall, the best unsupervised clustering configuration was multilingual MiniLM combined with K-Means.

### Clustering Evaluation:
Two metrics were used to evaluate the clustering results: the silhouette score and the Adjusted Rand Index.
The silhouette score measures how closely a sample belongs to its own cluster compared with other clusters. Higher values generally indicate better separation between groups.
The Adjusted Rand Index compares the clusters generated by the algorithm with the known civic categories in the evaluation dataset. It therefore provides a measure of how closely the discovered groups correspond to the expected topics.
The clustering experiments demonstrated that meaningful patterns could be extracted from multilingual citizen feedback. However, unsupervised clustering alone does not guarantee that every cluster corresponds directly to a predefined civic category.
For the deployed application, the clustering experiments were therefore complemented by semantic topic assignment.

### Semantic Topic Assignment:
The deployed Nasij application assigns every citizen suggestion directly to one of the five civic topics using semantic similarity.
A representative description is defined for each topic. For example, the representation of waste management includes concepts such as garbage collection, trash, waste, garbage bins, and dirty streets. The roads and potholes category includes concepts related to potholes, damaged roads, road repairs, and street maintenance.
Equivalent topic descriptions are defined for street lighting, green spaces, and water supply.
Each topic description is converted into an embedding using the same multilingual MiniLM model.
When a citizen submits a suggestion, the suggestion is converted into a 384-dimensional embedding. Cosine similarity is then calculated between that embedding and the embeddings representing the five civic topics.
The civic topic with the highest similarity score is selected as the predicted category.
This produces interpretable categories that can be displayed directly on the municipality dashboard.

### Semantic Topic Evaluation:
The semantic topic assignment approach was evaluated on the same balanced dataset of 30 English and Arabic suggestions.
Twenty-six of the thirty suggestions were assigned to the correct civic category, corresponding to an accuracy of approximately 86.67 percent.
The method performed particularly well for waste management, roads and potholes, and green spaces.
Water-related suggestions were the most challenging category, with some examples being assigned to other civic topics.
The results indicate that semantic topic matching provides a practical method for organizing multilingual civic feedback into predefined categories.

### Lebanese Arabic Processing:
Nasij also includes a lightweight lexical layer designed for common Lebanese Arabic civic expressions.
Although the Sentence Transformer provides multilingual semantic representations, short colloquial expressions may sometimes contain highly specific local vocabulary that can be identified more reliably through lexical matching.
The system therefore checks for civic terms associated with the five supported topics.
Examples include words related to زبالة for waste management, حفر for roads and potholes, expressions related to إنارة and street lighting, terms such as شجر and حديقة for green spaces, and expressions such as انقطاع المي for water supply.
Arabic text is normalized before matching to reduce the effect of common orthographic variations.
When a clear civic expression is detected, the associated category can be assigned directly. Otherwise, the multilingual semantic similarity model is used.
This produces a hybrid NLP architecture combining multilingual sentence embeddings with a small rule-based layer adapted to commonly used Lebanese civic vocabulary.

### Citizen Feedback Aggregation:
Once citizen suggestions have been assigned to civic categories, the municipality dashboard aggregates them by topic.
The number of submissions associated with each category is calculated automatically.
This allows the platform to identify which civic concern is being mentioned most frequently at a given time.
For example, if roads and potholes receives the largest number of submissions, it is displayed as the current top citizen concern.
The dashboard therefore transforms individual messages into a broader representation of community priorities.

### Representative Citizen Feedback:
For each civic topic, Nasij also identifies a representative citizen suggestion.
All suggestions assigned to the same topic are converted into embeddings and compared with the embedding representing that civic category.
The suggestion with the highest semantic similarity to the topic representation is selected as the representative concern.
This allows municipal staff to quickly understand the type of feedback associated with a category without having to read every individual submission.
The method is extractive because it selects an existing citizen statement instead of generating new text. This ensures that the displayed feedback remains directly grounded in actual citizen submissions.

### NLP Architecture:
The NLP workflow begins when a citizen submits a suggestion.
The text is first processed and checked for recognizable Lebanese Arabic civic expressions. When a clear lexical match is available, the corresponding civic topic is assigned.
Otherwise, the suggestion is converted into a multilingual MiniLM embedding and compared with the embeddings representing the five civic topics using cosine similarity.
The topic with the highest similarity is selected.
The classified suggestions are then aggregated by category. Their frequencies are calculated to identify the most common civic concerns, and representative suggestions are selected for display on the municipality dashboard.
This pipeline combines multilingual language understanding, semantic similarity, local linguistic adaptation, and feedback aggregation in a single system.

### NLP Limitations:
The NLP evaluation dataset is relatively small, containing 30 suggestions. It provides an initial measure of system behavior but does not represent the full variety of language that may appear in real municipal feedback.
Lebanese users may write in formal Arabic, Lebanese Arabic, English, Arabizi, or combinations of several languages. Spelling and dialectal expressions can also vary considerably between users.
The lexical component currently focuses on a limited set of common civic expressions and therefore cannot cover every possible formulation.
The semantic topic evaluation also showed that water-related suggestions were more difficult to classify than the other categories.
The reported 86.67 percent accuracy should therefore be interpreted as performance on the project evaluation dataset rather than as a general estimate of performance on all possible citizen submissions.

### Future NLP Development:
A larger dataset of real Lebanese citizen feedback would allow the NLP component to be evaluated and improved more extensively.
Future datasets could include English, formal Arabic, Lebanese Arabic, Arabizi, and mixed-language submissions.
With sufficient labeled data, a dedicated civic topic classifier could be trained or fine-tuned specifically for Lebanese municipal feedback.
The system could also be extended to recognize additional civic topics, detect multiple topics within the same submission, identify duplicate suggestions, estimate urgency, and analyze sentiment.
More advanced summarization methods could also be introduced for large volumes of feedback, allowing the municipality to receive concise summaries of hundreds of related citizen submissions.
The current NLP component provides the foundation for this functionality by converting multilingual and unstructured citizen suggestions into organized civic categories and actionable dashboard information.
