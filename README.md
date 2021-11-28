# rec-NN
Recommender system for songs using different neural networks: MLP, VAE and flow. 

The dataset used is the [Million Song Dataset](http://millionsongdataset.com/), in particular the user's data and the song's metadata (i.e. genre, tags, author...) which can be found in the [additional datasets](http://millionsongdataset.com/pages/additional-datasets/). 

## Models and Training

All models are located inside [`main/models`](main/models), which include MLP and VAE with different hidden layers, the reversible flow (flow) and the recommender flow (MLP+flow). 

To train the MLP and VAE models, run
```
python train.py --TODO TRAIN --mod <mode_name> --loss <function_name> 
```

The complete description of all the arguments and their default values can be found in [`main/wrapper.py`](main/wrapper.py), see also [`variables_definitions.pdf`](variables_definitions.pdf). 

To train flow models, run
```
python train_flow.py --TODO TRAIN --mod <mode_name> --loss <function_name> 
```

The complete description of all the arguments and their default values can be found in [`main/wrapper.py`](main/wrapper.py), see also [`variables_definitions.pdf`](variables_definitions.pdf). 

## Recommend

Given a model and an input vector of listened songs, it returns the recommendation calculated by the model. 
It can perform the following types of recommendations: 
- *standard* = regular recommendation, without tunning
- *postfiltering* = increases probability (from uniform probability) of the selected tags with alpha factor
- *tunning* = moves user vector in the latent space with linear combination using alpha as factor

## Predict 

It tests the recommendation from `recommend.py` using NDCG ranking, the distance in the latent space and the number of recommended songs with the desired metadata. 

## References

1. Liang, Dawen, et al. "Variational autoencoders for collaborative filtering." Proceedings of the 2018 world wide web conference. 2018.
2. Zhang, Shuai, et al. "Deep learning based recommender system: A survey and new perspectives." ACM Computing Surveys (CSUR) 52.1 (2019): 1-38.
3. Kingma, Diederik P., and Prafulla Dhariwal. "Glow: Generative flow with invertible 1x1 convolutions." arXiv preprint arXiv:1807.03039 (2018).
4. Rezende, Danilo, and Shakir Mohamed. "Variational inference with normalizing flows." International conference on machine learning. PMLR, 2015.
5. Chen, Yifan, and Maarten de Rijke. "A collective variational autoencoder for top-n recommendation with side information." Proceedings of the 3rd Workshop on Deep Learning for Recommender Systems. 2018.
6. Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." arXiv preprint arXiv:1205.2618 (2012).
7. Hidasi, Balázs, et al. "Session-based recommendations with recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
8. Hidasi, Balázs, and Alexandros Karatzoglou. "Recurrent neural networks with top-k gains for session-based recommendations." Proceedings of the 27th ACM international conference on information and knowledge management. 2018.
9. Karamanolakis, Giannis, et al. "Item recommendation with variational autoencoders and heterogeneous priors." Proceedings of the 3rd Workshop on Deep Learning for Recommender Systems. 2018.
