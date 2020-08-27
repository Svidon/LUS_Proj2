# LUS_Proj2

Second project for Language Understanding System course about *Concept tagging*.

The dataset used is about movies and it can be found [here](https://github.com/esrel/NL2SparQL4NLU).

Here some basic information is explained. For more details look at `report.pdf`.

The first project used WFSTs and has an analysis of the dataset. The repository can be found [here](https://github.com/Svidon/LUS_Project1).

---

## Project Structure and How to Use

This project requires having installed python3, sklearn_crfsuite, pytorch (along with torchtext and pytorch-crf) and spacy (with the large english dataset, `en_web_core_lg`).

The files `conlleval.pl` and `conll.py` are utilities (given by the lecturers) to evaluate the model on the test set.

The file `LSTM+CRF/utils.py` contains some utilities used for the batch training of the NNs. It allows to split the data in batches so that each batch has only sentences of the same length. This allows to avoid using padding. The code has not been written by me, and it can be found [here](https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/utils/data/text/dataset.py) (credits to user *chrisvdweth*).

There are 3 different models. In all of the different folder there will be a *model_evaluations* folder, in which there is the best resulting tags for each model (`result_<model_info>.txt`) and a series of files containing the evaluations for different parameters tested. These files are not generated from the python evaluation script (which doesn't give the accuracy), but from the perl script (also for consistency with the previous project), thanks to the `complete_evaluation.sh` script. For the NNs models the folder also contains `learning_curve_<model>_<embedding>.png` with the plot of the learninig curve of the best model and `learning_curve_lstm_spacy_150ep.png`, used in the report to explain the decision on the number of epochs for the training.

The models are the following:

* **CRF model:** it is in the *CRF* folder. To run the model and tag the test set just execute `python3 crf.py`. The parameter you can tune is the **window size**: you have to modify the `WIN` variable in the python script. The resulting tagged sentences can be found in `result.txt`. An additional file is generated, `evaluation_window{size}_python`, containing both the evaluation and the best hyperparameters of the algorithm (along with the window size). The evaluation files in the proper folder will have this same structure. If you want to evaluate the accuracy as well simply run `complete_evaluation.sh`: the results will be in the newly generated `evaluation.txt`. The model takes quite a long time to train, because it searches the hyperparameters in a parameter space. If you want to see the optimal results without the search you have to modify once again the script and assign the optimal parameters to the `crf` object.

* **biLSTM:** it is in the *LSTM+CRF* folder. To run the model execute `python3 main.py <embedding> lstm`. The last argument is to specify that the model to use is the LSTM, while embedding can be one of `{default, glove, spacy}` and it corresponds to different word embeddings used. The file `models/lstm.py` contains the network model. The output is the evaluation file `evaluation_lstm_<embed_name>_bi.txt`. Again to have a more complete evaluation use `complete_evaluation.sh`.

* **biLSTM+CRF:** it is always in the LSTM+CRF* folder. The model can be found in `models/lstm_crf.py`. The execution and the outputs are the same as above, with the evaluation file being named `evaluation_lstm+crf_<embed_name>_bi.txt`. The execution is slower than the biLSTM because the forward pass only returns the NLL: there is then also a predict pass to be able to compute the F1 score during the training.

*Note:* The CRF results might slightly change if you used the saved parameters in the evaluation files because of approximations.

*Note:* All the NNs generated results have the suffix `_bi` because the LSTM is bidirectional by default (it achieves better results). If you want to run the non-biLSTM set the parameter `bi` in `main.py` to `False`. To get the best results for `default` you have to set `lemma` in `main.py` to `True`.

*Note:* The results for the NNs could change on different machines. This is due to the pytorch architecture. The results will be consistent on the same machine.