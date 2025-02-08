# Create your API token from your Hugging Face Account. Make sure to save it in text file or notepad for future use.
# Will need to add it once per section
from huggingface_hub import login

from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import load_dataset

import gdown
import os
import random
import torch
from torch.utils.data import DataLoader

from sympy import false


# Create your API token from your Hugging Face Account. Make sure to save it in text file or notepad for future use.
# Will need to add it once per section
class TextSimilarityModel:
    def __init__(self, corpus_name, rel_name, model_name='all-MiniLM-L6-v2', top_k=10):
        """
        Initialize the model with datasets and pre-trained sentence transformer.
        """
        self.model = SentenceTransformer(model_name)
        # "/content/finetuned_senBERT_train_v2" model_name
        self.corpus_name = corpus_name
        self.rel_name = rel_name
        self.top_k = top_k

        ### Add Init
        self.query_id_to_ranked_doc_ids = {}
        self.glove_embedding_dict = {}
        self.add_negative = False

        ###
        self.load_data()


    def load_data(self):
        """
        Load and filter datasets based on test queries and documents.
        """
        # Load query and document datasets
        dataset_queries = load_dataset(self.corpus_name, "queries")
        dataset_docs = load_dataset(self.corpus_name, "corpus")

        # Extract queries and documents
        self.queries = dataset_queries["queries"]["text"]
        self.query_ids = dataset_queries["queries"]["_id"]
        self.documents = dataset_docs["corpus"]["text"]
        self.document_ids = dataset_docs["corpus"]["_id"]


        # Filter queries and documents and build relevant queries and documents mapping based on test set
        test_qrels = load_dataset(self.rel_name)["test"]
        self.filtered_test_query_ids = set(test_qrels["query-id"])
        self.filtered_test_doc_ids = set(test_qrels["corpus-id"])

        self.test_queries = [q for qid, q in zip(self.query_ids, self.queries) if qid in self.filtered_test_query_ids]
        self.test_query_ids = [qid for qid in self.query_ids if qid in self.filtered_test_query_ids]
        self.test_documents = [doc for did, doc in zip(self.document_ids, self.documents) if did in self.filtered_test_doc_ids]
        self.test_document_ids = [did for did in self.document_ids if did in self.filtered_test_doc_ids]

        self.test_query_id_to_relevant_doc_ids = {qid: [] for qid in self.test_query_ids}
        for qid, doc_id in zip(test_qrels["query-id"], test_qrels["corpus-id"]):
            if qid in self.test_query_id_to_relevant_doc_ids:
                self.test_query_id_to_relevant_doc_ids[qid].append(doc_id)

        ## Code Below this is used for creating the training set
        # Build query and document id to text mapping
        self.query_id_to_text = {query_id:query for query_id, query in zip(self.query_ids, self.queries)}
        self.document_id_to_text = {document_id:document for document_id, document in zip(self.document_ids, self.documents)}

        # Build relevant queries and documents mapping based on train set
        train_qrels = load_dataset(self.rel_name)["train"]
        self.train_query_id_to_relevant_doc_ids = {qid: [] for qid in train_qrels["query-id"]}

        for qid, doc_id in zip(train_qrels["query-id"], train_qrels["corpus-id"]):
            if qid in self.train_query_id_to_relevant_doc_ids:
                # Append the document ID to the relevant doc mapping
                self.train_query_id_to_relevant_doc_ids[qid].append(doc_id)

        # Filter queries and documents and build relevant queries and documents mapping based on validation set
        #TODO Put your code here. Done by Tianyi Li on 02/06/2025
         ###########################################################################
        # Build relevant queries and documents mapping based on validation set
        validate_qrels = load_dataset(self.rel_name)["validation"]
        self.validate_query_id_to_relevant_doc_ids = {qid: [] for qid in validate_qrels["query-id"]}

        for qid, doc_id in zip(validate_qrels["query-id"], validate_qrels["corpus-id"]):
            if qid in self.validate_query_id_to_relevant_doc_ids:
                # Append the document ID to the relevant doc mapping
                self.validate_query_id_to_relevant_doc_ids[qid].append(doc_id)
        ###########################################################################

    def load_glove_embeddings(self, glove_path) -> None :
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                self.glove_embedding_dict[word] = vector

    #Task 1: Encode Queries and Documents (10 Pts)
    def encode_with_glove(self, glove_file_path: str, sentences: list[str]) -> list[np.ndarray]:

        """
        # Inputs:
            - glove_file_path (str): Path to the GloVe embeddings file (e.g., "glove.6B.50d.txt").
            - sentences (list[str]): A list of sentences to encode.

        # Output:
            - list[np.ndarray]: A list of sentence embeddings

        (1) Encodes sentences by averaging GloVe 50d vectors of words in each sentence.
        (2) Return a sequence of embeddings of the sentences.
        Download the glove vectors from here.
        https://nlp.stanford.edu/data/glove.6B.zip
        Handle unknown words by using zero vectors
        """
        #TODO Put your code here. Done by Tianyi Li on 02/05/2025
        ###########################################################################

        if not self.glove_embedding_dict:
            self.load_glove_embeddings(glove_file_path)
        embeddings_sentences = []
        for sentence in sentences:
            embeddings_sentence = np.zeros(50)
            words = sentence.split()
            for word in words:
                if word.lower() in self.glove_embedding_dict:
                    embeddings_sentence += self.glove_embedding_dict[word.lower()]
                else:
                    embeddings_sentence += np.zeros(50)
            embeddings_sentence /= len(words)
            embeddings_sentences.append(embeddings_sentence)
        return embeddings_sentences
        ###########################################################################

    #Task 2: Calculate Cosine Similarity and Rank Documents (20 Pts)

    def rank_documents(self, encoding_method: str = 'sentence_transformer') -> None:
        """
         # Inputs:
            - encoding_method (str): The method used for encoding queries/documents.
                             Options: ['glove', 'sentence_transformer'].

        # Output:
            - None (updates self.query_id_to_ranked_doc_ids with ranked document IDs).

        (1) Compute cosine similarity between each document and the query
        (2) Rank documents for each query and save the results in a dictionary "query_id_to_ranked_doc_ids"
            This will be used in "mean_average_precision"
            Example format {2: [125, 673], 35: [900, 822]}
        """
        if encoding_method == 'glove':
            query_embeddings = self.encode_with_glove("glove.6B.50d.txt", self.queries)
            document_embeddings = self.encode_with_glove("glove.6B.50d.txt", self.documents)
        elif encoding_method == 'sentence_transformer':
            query_embeddings = self.model.encode(self.queries)
            document_embeddings = self.model.encode(self.documents)
        else:
            raise ValueError("Invalid encoding method. Choose 'glove' or 'sentence_transformer'.")

        #TODO Put your code here. Done by Tianyi Li on 02/05/2025
        ###########################################################################
         # define a dictionary to store the ranked documents for each query

        query_embeddings = np.array(query_embeddings)
        document_embeddings = np.array(document_embeddings)
        cosine_similarities = cosine_similarity(query_embeddings,document_embeddings)
        for query_id, similarity_scores in zip(self.query_ids, cosine_similarities):
            sorted_indices = np.argsort(similarity_scores)[::-1][:self.top_k]  # Sort in descending order
            ranked_doc_ids = [self.document_ids[idx] for idx in sorted_indices]  # Map indices to document IDs
            self.query_id_to_ranked_doc_ids[query_id] = ranked_doc_ids
        ###########################################################################

    @staticmethod
    def average_precision(relevant_docs: list[str], candidate_docs: list[str]) -> float:
        """
        # Inputs:
            - relevant_docs (list[str]): A list of document IDs that are relevant to the query.
            - candidate_docs (list[str]): A list of document IDs ranked by the model.

        # Output:
            - float: The average precision score

        Compute average precision for a single query.
        """
        y_true = [1 if doc_id in relevant_docs else 0 for doc_id in candidate_docs]
        precisions = [np.mean(y_true[:k+1]) for k in range(len(y_true)) if y_true[k]]
        return np.mean(precisions) if precisions else 0

    #Task 3: Calculate Evaluate System Performance (10 Pts)

    def mean_average_precision(self) -> float:
        """
        # Inputs:
            - None (uses ranked documents stored in self.query_id_to_ranked_doc_ids).

        # Output:
            - float: The MAP score, computed as the mean of all average precision scores.

        (1) Compute mean average precision for all queries using the "average_precision" function.
        (2) Compute the mean of all average precision scores
        Return the mean average precision score

        reference: https://www.evidentlyai.com/ranking-metrics/mean-average-precision-map
        https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2
        """
         #TODO Put your code here. Done by DanieL Chen on 02/06/2025
        ###########################################################################
        average_precisions = [] # Create an empty list average_precisions to store the AP of each query

        for qid in self.test_query_ids:
            relevant_docs = self.test_query_id_to_relevant_doc_ids[qid]
            ranked_docs = self.query_id_to_ranked_doc_ids[qid]
            average_precisions.append(self.average_precision(relevant_docs, ranked_docs))

        return np.mean(average_precisions) if average_precisions else 0.0

        ###########################################################################

    #Task 4: Ranking the Top 10 Documents based on Similarity Scores (10 Pts)

    def show_ranking_documents(self, example_query: str) -> None:

        """
        # Inputs:
            - example_query (str): A query string for which top-ranked documents should be displayed.

        # Output:
            - None (prints the ranked documents along with similarity scores).

        (1) rank documents with given query with cosine similarity scores
        (2) prints the top 10 results along with its similarity score.

        """
        #TODO Put your code here. Done by DanieL Chen on 02/06/2025
        # query_embedding = self.model.encode(example_query)
        query_embedding = self.model.encode([example_query])[0] # encode() requires a List format
        document_embeddings = self.model.encode(self.documents)
        ###########################################################################
        cosine_similarities = cosine_similarity([query_embedding], document_embeddings)[0]
        sorted_indices = np.argsort(cosine_similarities)[::-1][:10]
        ranked_docs = [(self.documents[i], cosine_similarities[i]) for i in sorted_indices]

        print(f"Top 10 documents for query: {example_query}\n")

        for i, (doc, score) in enumerate(ranked_docs, 1):
            print(f"{i}. Score: {score:.4f}, Document: {doc}")


        ###########################################################################

    #Task 5:Fine tune the sentence transformer model (25 Pts)
    # Students are not graded on achieving a high MAP score.
    # The key is to show understanding, experimentation, and thoughtful analysis.

    def fine_tune_model(self, batch_size: int = 32, num_epochs: int = 3, save_model_path: str = "finetuned_senBERT") -> None:

        """
        Fine-tunes the model using MultipleNegativesRankingLoss.
        (1) Prepare training examples from `self.prepare_training_examples()`
        (2) Experiment with [anchor, positive] vs [anchor, positive, negative]
        (3) Define a loss function (`MultipleNegativesRankingLoss`)
        (4) Freeze all model layers except the final layers
        (5) Train the model with the specified learning rate
        (6) Save the fine-tuned model
        """
        #TODO Put your code here.
        ###########################################################################
        self.add_negative = True
        train_examples = self.prepare_training_examples()

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        for name, param in self.model.named_parameters():
          if "encoder.layer.5" in name or "pooler" in name:  
              param.requires_grad = True
          else:
              param.requires_grad = False
        trainable_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
        print("Trainable parameters:", trainable_params)

        self.model.train()
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            show_progress_bar=True,
            optimizer_params={'lr': 5e-5}
        )

        self.model.save(save_model_path)
        ###########################################################################

    # Take a careful look into how the training set is created
    def prepare_training_examples(self) -> list[InputExample]:

        """
        Prepares training examples from the training data.
        # Inputs:
            - None (uses self.train_query_id_to_relevant_doc_ids to create training pairs).

         # Output:
            Output: - list[InputExample]: A list of training samples containing [anchor, positive] or [anchor, positive, negative].

        """
        train_examples = []
        for qid, doc_ids in self.train_query_id_to_relevant_doc_ids.items():
            negative_candidates = list(set(self.document_ids) - set(doc_ids))
            for doc_id in doc_ids:
                anchor = self.query_id_to_text[qid]
                positive = self.document_id_to_text[doc_id]
                # TODO: Select random negative examples that are not relevant to the query.
                # TODO: Create list[InputExample] of type [anchor, positive, negative]
                if self.add_negative:
                    negative_id = random.choice(negative_candidates)
                    negative = self.document_id_to_text[negative_id]
                    train_examples.append(InputExample(texts=[anchor, positive, negative]))
                else:
                    train_examples.append(InputExample(texts=[anchor, positive]))
                # train_examples.append(InputExample(texts=[anchor, positive]))

        return train_examples

# 0. Login to huggingface
login()

# 1. Initialize the model
# Initialize and use the model
model = TextSimilarityModel("BeIR/nfcorpus", "BeIR/nfcorpus-qrels")

# 2. Rank documents using sentence transformer
# Compare the outputs 
print("Ranking with sentence_transformer...")
model.rank_documents(encoding_method='sentence_transformer')
map_score = model.mean_average_precision()
print("Mean Average Precision:", map_score)


# # 3. Rank documents using glove
# # Download glove txt
# glove_file_path = 'glove.6B.50d.txt'
# embeddings_id = "1sX7UOmk8dGQfGhe8s1qOyN10XvL-8qHx"
# if not os.path.exists(glove_file_path):
#     print("Donwloading embedings...\n\n")
#     gdown.download(id=embeddings_id, output=glove_file_path, quiet=False)
# # Compare the outputs
# print("Ranking with glove...")
# model.rank_documents(encoding_method='glove')
# map_score = model.mean_average_precision()
# print("Mean Average Precision:", map_score)


# # 4. show top_k rankings using sentence transformer
model.show_ranking_documents("Breast Cancer Cells Feed on Cholesterol")

# 5. Finetune all-MiniLM-L6-v2 sentence transformer model
model.fine_tune_model(batch_size=32, num_epochs=3, save_model_path="finetuned_senBERT_train_v2")  # Adjust batch size and epochs as needed
model.rank_documents()
map_score = model.mean_average_precision()
print("Mean Average Precision:", map_score)
model.show_ranking_documents("Breast Cancer Cells Feed on Cholesterol")