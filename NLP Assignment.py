import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from random import sample
import numpy as np

def generate_mcq(paragraph, num_questions=6, num_options=4):
    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(paragraph)

    # Tokenize the sentences into words
    word_tokens = [word_tokenize(sentence) for sentence in sentences]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [[word for word in words if word.lower() not in stop_words] for words in word_tokens]

    # Convert filtered words back to sentences
    filtered_sentences = [' '.join(words) for words in filtered_words]

    # Calculate TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_sentences)

    # Perform clustering using K-means
    num_clusters = min(num_questions, len(sentences))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0,  n_init=10).fit(tfidf_matrix)

    # Generate questions and options
    questions = []
    for cluster in range(num_clusters):
        indices = [i for i, label in enumerate(kmeans.labels_) if label == cluster]
        if len(indices) >= num_options:
            centroid = np.asarray(tfidf_matrix[indices].mean(axis=0))
            similarities = cosine_similarity(centroid, tfidf_matrix[indices])
            closest_sentences = [sentences[indices[i]] for i in similarities.argsort()[0][-num_options:]]
            correct_option = sample(range(num_options), 1)[0]
            options = [closest_sentences[i] for i in range(num_options)]
            #options[correct_option] = closest_sentences[-1]  # Set last sentence as the correct answer
            questions.append((closest_sentences[correct_option], options))

    return questions


# Example usage
paragraph = '''
Photosynthesis is a process used by plants and other organisms to convert light energy into
chemical energy that, through cellular respiration, can later be released to fuel the organism's
activities. Some of this chemical energy is stored in carbohydrate molecules, such as sugars and
starches, which are synthesized from carbon dioxide and water – hence the name photosynthesis,
from the Greek phōs, "light", and synthesis , "putting together". Most plants, algae, and
cyanobacteria perform photosynthesis; such organisms are called photoautotrophs. Photosynthesis
is largely responsible for producing and maintaining the oxygen content of the Earth's atmosphere,
and supplies most of the energy necessary for life on Earth.
Although photosynthesis is performed differently by different species, the process always begins
when energy from light is absorbed by proteins called reaction centers that contain green chlorophyll
(and other colored) pigments/chromophores. In plants, these proteins are held inside organelles
called chloroplasts, which are most abundant in leaf cells, while in bacteria they are embedded in
the plasma membrane. In these light-dependent reactions, some energy is used to strip electrons
from suitable substances, such as water, producing oxygen gas. The hydrogen freed by the splitting
of water is used in the creation of two further compounds that serve as short-term stores of energy,
enabling its transfer to drive other reactions: these compounds are reduced nicotinamide adenine
dinucleotide phosphate (NADPH) and adenosine triphosphate (ATP), the "energy currency" of cells.
In plants, algae and cyanobacteria, sugars are synthesized by a subsequent sequence of
light-independent reactions called the Calvin cycle. In the Calvin cycle, atmospheric carbon dioxide
is incorporated into already existing organic carbon compounds, such as ribulose bisphosphate
(RuBP).[5] Using the ATP and NADPH produced by the light-dependent reactions, the resulting
compounds are then reduced and removed to form further carbohydrates, such as glucose. In other
bacteria, different mechanisms such as the reverse Krebs cycle are used to achieve the same end.
The first photosynthetic organisms probably evolved early in the evolutionary history of life and most
likely used reducing agents such as hydrogen or hydrogen sulfide, rather than water, as sources of
electrons. Cyanobacteria appeared later; the excess oxygen they produced contributed directly to
the oxygenation of the Earth, which rendered the evolution of complex life possible. Today, the
average rate of energy capture by photosynthesis globally is approximately 130 terawatts, which is
about eight times the current power consumption of human civilization. Photosynthetic organisms
also convert around 100–115 billion tons (91–104 Pg petagrams, or billion metric tons), of carbon
into biomass per year. That plants receive some energy from light –
'''

generated_questions = generate_mcq(paragraph, num_questions=4 , num_options=4)

# Print the generated questions and options
for i, (question, options) in enumerate(generated_questions):
    print(f"Question {i + 1}: {question}")
    for j, option in enumerate(options):
        print(f"Option {j + 1}: {option}")
    print()
    
    









