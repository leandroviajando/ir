# Information Retrieval and Search

## 1. Introduction

Information retrieval is concerned with the _non-deterministic_ matching of a _query_ and _documents_ in large collections of _unstructured_ data (cf. data retrieval in a structured, deterministic context):

- Input: _Representation of text_.
- System: Formulation `q`, index generation `d` yielding a `Ranking(q, d)` of relevant documents.
- Output: _Retrieval model_.

### Text Preprocessing

1. Preparation of the document, e.g. removal of tags.
2. Lexical analysis: tokenisation.
3. Removal of stopwords (optional).
4. Stemming (optional), e.g. Porter algorithm.
5. Term weighting (optional).

### Text Representation and Similarity

Bag of Words (BoW) does not capture word order (and consequently similarity measures will be the same for some distinct sentences). Alternatives include character n-grams or word n-grams.

### Term Weighting

A weight is an importance indicator of a term regarding content:

- _Term frequency_: frequency of occurrence of the index term $i$ in document $j$. _Assumption_: High-frequency content signals main topics in the document!

  $$w_{ij} = \text{tf}_{ij}$$

**(Constant Rank-Frequency) Law of Zipf**: rank-frequency exhibit a log-linear relationship; highly frequent words are frequent in a lot of documents and might not necessarily be informative about a particular document.

- _Inverse document frequency_: $N$ = no. of documents in the reference collection, $n_i$ = no. of documents in the reference collection having index term $i$. Solves the previous issue!

  $$w_{ij} = \text{idf}_i = \log\Big( \frac{N}{n_i} \Big)$$

- _TfIdf_:

  $$w_{ij} = \text{tf-idf}_{ij} = \text{tf}_{ij} \times \text{idf}_i$$

- _Length normalisation_: $l$ = no. of distinct index terms in the document. Without normalisation, long documents would have a higher weight purely by virtue of containing more words!

  $$w_{ij} = \frac{\text{tf}_{ij}}{\max_{1 < k < l}(\text{tf}_{kj})}$$

- _Augmented normalised term frequency_: smoothing term $\alpha$ usually equal to $0.5$.

  $$w_{ij} = \alpha + (1 - \alpha) \frac{\text{tf}_{ij}}{\max_{1 < k < l}(\text{tf}_{kj})}$$

  $\alpha = 0$ is equivalent to standard length normalisation:

  ```python
  import numpy as np

  tf = np.array([2., 1., 2., 1., 0.])

  def augmented_normalised_tf(tf, alpha):
      return alpha + (1 - alpha) * (tf / max(tf))

  for alpha in [0., 0.25, 0.5, 0.75, 1.]:
      print(augmented_normalised_tf(tf, alpha))
  [1., 0.5,   1., 0.5,   0.  ]
  [1., 0.625, 1., 0.625, 0.25]
  [1., 0.75,  1., 0.75,  0.5 ]
  [1., 0.875, 1., 0.875, 0.75]
  [1., 1.,    1., 1.,    1.  ]
  ```

### Ranked Retrieval Evaluation

More informative than accuracy:

$$\text{Precision (P)} = \frac{\textcolor{blue}{\text{retrieved}} \land \textcolor{red}{\text{relevant}}}{\textcolor{blue}{\text{retrieved}}}
\qquad \qquad
\text{Recall (R)} = \frac{\textcolor{blue}{\text{retrieved}} \land \textcolor{red}{\text{relevant}}}{\textcolor{red}{\text{relevant}}}$$

```python
import numpy as np
import matplotlib.pyplot as plt

docs = np.array([1., 1., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.])

retrieved_and_relevant = docs.cumsum()
relevant = docs.nonzero()[0] + 1  # zero-indexing
precision = lambda retrieved: retrieved_and_relevant[retrieved-1] / retrieved
recall = lambda retrieved: retrieved_and_relevant[retrieved-1] / relevant.size

all = np.arange(docs.size) + 1
np.round(np.stack((all, precision(all), recall(all))), 2)
array([[ 1.  ,  2.  ,  3.  ,  4.  ,  5.  ,  6.  ,  7.  ,  8.  ,  9.  , 10.  , 11.  , 12.  , 13.  , 14.  , 15.  , 16.  , 17.  , 18.  , 19.  , 20.  , 21.  , 22.  , 23.  , 24.  ],
       [ 1.  ,  1.  ,  1.  ,  0.75,  0.8 ,  0.67,  0.71,  0.75,  0.67,  0.6 ,  0.64,  0.58,  0.62,  0.57,  0.53,  0.5 ,  0.47,  0.5 ,  0.47,  0.45,  0.43,  0.41,  0.39,  0.42],
       [ 0.1 ,  0.2 ,  0.3 ,  0.3 ,  0.4 ,  0.4 ,  0.5 ,  0.6 ,  0.6 ,  0.6 ,  0.7 ,  0.7 ,  0.8 ,  0.8 ,  0.8 ,  0.8 ,  0.8 ,  0.9 ,  0.9 ,  0.9 ,  0.9 ,  0.9 ,  0.9 ,  1.  ]])

pr_recall_levels = np.round(np.stack((relevant, recall(relevant), precision(relevant))), 2)
pr_recall_levels
array([[ 1.  ,  2.  ,  3.  ,  5.  ,  7.  ,  8.  , 11.  , 13.  , 18.  ,  24. ],
       [ 0.1 ,  0.2 ,  0.3 ,  0.4 ,  0.5 ,  0.6 ,  0.7 ,  0.8 ,  0.9 ,  1.  ],
       [ 1.  ,  1.  ,  1.  ,  0.8 ,  0.71,  0.75,  0.64,  0.62,  0.5 ,  0.42]])
plt.plot(pr_recall_levels[1], pr_recall_levels[2])
plt.show()
```

A single trade-off measure: $$F = \frac{(\beta^2 + 1) PR}{\beta^2 P + R}$$

- $\beta < 1$: emphasise precision
- $\beta = 1$: harmonic mean (F1 score)
- $\beta > 1$: emphasis recall

Breakeven point: point in the PR graph where $P = R$.

MAP, AUC-ROC, confusion matrix, etc.

## 2. Retrieval Models

Information retrieval (/ ranking / relevance) models are defined by:

- The **representation** of the document and query.
- The **ranking** function that uses these as arguments.

Model taxonomy:

- Simple (based on co-ocurrence / frequencies), _unsupervised_ (and can be applied to large datasets), focused on _topical relevance_ (necessary but not sufficient!) - therefore useful for an initial filtering of a large set of documents:
  - Set-theoretic (e.g. boolean): simple and efficient representation (as a set of index terms) but no ranking (only whether relevant or not).
  - Probabilistic (e.g. probabilistic language model): model the probability of relevance given a document and query.
  - Algebraic (e.g. vector space): simple and efficient representation (as vectors), ranking (similarity function between `d` and `q`).
- Link-based (e.g. PageRank)
- Neural network-based: learned vector representations of `q` and `d` through _supervision_ (need annotated data such as user profiles and behaviour to model intentional and motivational relevance).

A **retrieval model** $\langle D, Q, F, R(d_j, q_i) \rangle$:

- $D$ is the set of _representations_ of the _documents_ in the collection.
- $Q$ is the set of _representations_ of the _queries_.
- $F$ is a framework for modelling document and query _representations_, and their relationships.
- $R(d_j , q_i)$ is a _ranking function_ that takes a document and query representations $d_j \in D, q_i \in Q$ and returns a real number that expresses the potential relevance of $d_j$ to $q_i$ by which documents can be ordered.
- $K = k_1, \dots, k_t$ is the set of all index terms.
- $t$ is the number of index (_vocabulary_) terms in the collection.
- $w_{ij} \text{ or } w_{iq}$ is a weight associated with each index term $k_i$ of a document representation $d_j$ or query representation $q$:
  - $\mathbf{d_j} = [w_{1j}, w_{2j}, \dots, w_{tj}]$ is the term vector of $d_j$.
  - $\mathbf{q} = [w_{1q}, w_{2q}, \dots, w_{tq}]$ is the term vector of $q$.

### Boolean Models

The index term weight variables are all binary: $w_{ij}, w_{iq} \in \{0, 1\}$.

A query $q$ is a Boolean expression in _disjunctive normal form (DNF)_:

$$\text{Query: } k_1 \land \lnot k_4$$
$$\text{DNF: } (0, 0, 0, 1) \lor (0, 0, 1, 1) \lor (0, 1, 0, 1) \lor (0, 1, 1, 1)$$

| Advantages | Disadvantages                                                          |
| ---------- | ---------------------------------------------------------------------- |
| Simplicity | Relative importance of index terms is ignored.                         |
|            | No ranking (only matching; may be that no document matches the query). |

### Extended Boolean Models

Hybrid model with properties of set theoretic and algebraic models: takes into account _partial fulfilment_ of the query and _sorts_ documents by relevance to obtain a ranking.

Note: disjunctive queries are good if far from the origin, conjunctive queries are good if close to $(1, 1)$.

### Bag of Words (BoW) Vector Space Models

Document and query are represented as term vectors with term weights $\geq 0$ in a $t$-dimensional space, where $t$ is the number of features (here terms) measured. Vector of concepts / topics with number of dimensions $k \ll$ number of index terms $t$.

A ranking of the documents is obtained with a distance / _similarity_ measure between document $d_j$ and query $q$: Manhattan distance, Euclidean distance, inner product similarity, cosine similarity (most popular), etc.

$$\cos(d_j, q) = \frac{d_j^T \cdot q}{\lVert d_j \rVert \lVert q \rVert} = \frac{\sum_{i=1}^t{w_{ij} w_{iq}}}{\sum_{i=1}^t{w_{ij}^2} \, \sum_{i=1}^t{w_{iq}^2}}$$

TODO: disadvantage of BoW vector space models (below) and picture in this lecture with disadvantage mentioned in lecture 1.

| Disadvantages                                                                                  | Advantages                                                                          |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Simplifying assumption that terms are not correlated and term vectors are pair-wise orthogonal | Partial matching: retrieval of documents that approximate the query conditions      |
|                                                                                                | Simple, efficient model with relatively good results → popular (e.g., SMART system) |

### Probabilistic Retrieval Models

Probabilistically retrieval is a problem of estimating the probability of relevance given a query, document, collection, etc. and ranking the retrieved documents in decreasing order of this probability.

#### Generative Relevance Models

Consider the random variables $D$ = document, $Q$ = query, $R \in \{r, \bar{r}\}$ = relevance (relevant, not relevant).

$$
P(R = r \mid D, Q)
= 1 - P(R = \bar{r} \mid, D, Q)
\approxeq \log{ \frac{P(D \mid Q, R = r)}{P(D \mid Q, R = \bar{r})} }
$$

In a classical probabilistic model, the document is represented by a collection of its attributes $D = W_i, \dots, W_n$ (e.g. words) and the probabilities are factorised across these attributes - estimated from the set of relevant and non-relevant documents (e.g. MLE):

$$P(D \mid Q, R) = \prod_{i=1}^n{ P(W_i \mid Q, R) }$$

A language retrieval model ranks a document $D$ according to the probability that the document generates the query, i.e. $P(Q \mid D)$ where $C$ is the document collection and $\lambda$ is the Jelinek-Mercer smoothing parameter:

$$P(q_1, \dots, q_m \mid D ) = \prod_{i=1}^m{ \big( \lambda P_{MLE}(q_i \mid D) + (1-\lambda) P_{MLE}(q_i \mid C) \big) }$$

Evidence from multiple sources can easily be combined (with $cq_i$ = conceptual term, $w_l$ = content pattern such as a word image pattern):

$$P(cq_1, \dots, cq_m \mid D) = \prod_{i=1}^m{ \big( \alpha \sum_{i=1}^k{P(cq_i \mid w_l)P(w_l \mid D)} + \beta P(cq_i \mid D) + (1-\alpha-\beta) P(cq_i \mid C) \big) }$$

#### Inference Network Models

Representation as Bayesian Networks.

| Advantages                                                                                           | Disadvantages                                               |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Elegantly combines multiple sources of evidence and probabilistic dependencies                       | Computationally not scalable for querying large collections |
| Easy integration of representations of different media, domain knowledge, semantic information, etc. |                                                             |
| Good retrieval performance                                                                           |                                                             |

## 3. Probabilistic Representations

_Topic modelling_: Unsupervised representation learning of latent topics.

_Realisational chain_: Ideas → broad conceptual components of a text → sub-ideas →
sentences → set of semantic roles → set of grammatical and
lexical concepts → character sequences

**Generative model for documents**:

- Select a document $d_j$ with probability $P(d_j)$.
- Pick a latent class / concept $z_k$ with probability $P(d_j)$.
- Generate a word $w_i$ with probability $P(w_i \mid z_k)$.

Trained on a large corpus, learn:

- Per-document topic distributions.
- Per-topic word distributions.

### [Latent Semantic Analysis (LSA)](https://www.geeksforgeeks.org/latent-semantic-analysis/)

Weakness: cannot capture polysemy. Probabilistic topic models are a solution to this.

### Probabilistic latent semantic analysis (pLSA)

Maximum likelihood: parameters that maximise the likelihood of the observed data. Exact likelihood is intractable, we have to _approximate_ it, e.g., by:

- Expectation-maximization: iteratively estimate the probability of unobserved, latent variables until convergence.
- Gibbs sampling: update parameters sample-wise.
- Variational inference: approximate the model by an easier one.

These models are _unsupervised_ and learn from _raw_ data!

- Per-document topic distributions: $P(w_i \mid z_k)$.
- Per-topic word distributions: $P(z_k \mid d_j)$.

EM: TODO.

### Latent Dirichlet allocation (LDA)

Gibbs sampling:

Inference: he updating and sampling cycle from Gibbs sampling can be directly used for inference. DA can be applied to unseen documents!

## 4. Algebraic Representations

### Latent Semantic Indexing (LSI)

### Neural Network-Based Word Embeddings

### Representing documents and queries

## 5. Multimedia Information Retrieval

Representation and retrieval with multimedia: text, images, video, etc.

Multimedia data types and features

Concept detection

Cross-modal indexing of content: latent Dirichlet allocation and deep learning methods

Cross-modal and multimodal retrieval and recommendation models

Illustrations with spoken document, image, video and music search

## 6. Learning to Rank

Supervised machine learning to learn to rank documents.

Relevance feedback, personalized and contextualized information needs, user profiling

Pointwise, pairwise and listwise approaches

Structured output support vector machines, loss functions, most violated constraints

End-to-end neural network models

Optimization of retrieval effectiveness and of diversity of search results

## 7. Web Information Retrieval

Issues specific to web, such as:

- **Scalability**
- Heterogeneous content: multimedia, user generated content
- **Link** based retrieval models

Web search engines, crawler-indexer architecture, query processing

Link analysis retrieval models: PageRank, HITS, personalized PageRank and variants

Behaviur and credibility based retrieval models

Social search, mining and searching user generated content

## 8. Indexing, Compression and Search

Data structures and techniques for efficient storage and search:

- In-memory: SkipList, hash index.
- Document search: Inverted index.

Inverted files, nextword indices, taxonomy indices, distributed indices

Compression

Learning of hashing functions, cross-modal hashing

Scalability and efficiency challenges

Architectural optimizations

## 9. Clustering

Clustering similar content together to detect similar content or organise information.

Distance and similarity functions in Euclidean and hyperbolic spaces, proximity functions

Sequential and hierarchical cluster algorithms, algorithms based on cost-function optimization, number of clusters

Term clustering for query expansion, document clustering, multiview clustering

## 10. Categorization

Semantic labelling of documents for filtering, using supervised learning, e.g. spam detection.

Feature selection, naive Bayes model, support vector machines, (approximate) k-nearest neighbor models

Deep learning methods

Multilabel and hierarchical categorization

Convolutional neural network (CNN) based hierarchical categorization

## 11. Dynamic Retrieval and Search

Reinforcement learning from user interactions.

Static versus dynamic models

Markov decision processes

Multi-armed bandit models

Modelling sessions

Online advertising

Document segmentation, maximum marginal relevance

Summarization based on latent Dirichlet allocation models and long short-term memory (LSTM) networks

Abstractive summarization with attention models

Multidocument summarization, search results fusion and visualization

## 12. Question Answering, Conversational Search and Recommendations

- IR-based (visual) QA
- Conversational search and recommendation (including LLM-based chatbots)

Retrieval based question answering

Deep learning methods including attention models

Cross-modal question answering

E-commerce search and recommendation
