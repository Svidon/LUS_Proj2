# LUS_Proj2

Second project for Language Understanding System course about *Concept tagging*.

The dataset used is about movies and it can be found [here](https://github.com/esrel/NL2SparQL4NLU).

Here some basic information is explained. For more details look at `report.pdf`.

The first project used WFSTs and has an analysis of the dataset. The repository can be found [here](https://github.com/Svidon/LUS_Project1).

---

# Project Structure and How to Use

This project requires having installed python3, sklearn_crfsuite and pytorch.

The files `conlleval.pl` and `conll.py` are utilities (given by the lecturers) to evaluate the model on the test set.

There are 3 different models. In all of the different folder there will be a *model_evaluations* folder, in which there is the best resulting tags for each model (`result.txt`) and a series of files containing the evaluations for different parameters I tried. These files are not generated from the python evaluation script (which doesn't give the accuracy), but from the perl script (also for consistency with the previous project).

The models are the following:

* **CRF model:** it is in the *CRF* folder. To run the model and tag the test set just execute `python3 crf.py`. The parameter you can tune is the **window size**: you have to modify the `WIN` variable in the python script. The resulting tagged sentences can be found in `result.txt`. An additional file is generated, `evaluation_window{size}_python`, containing both the evaluation and the best hyperparameters of the algorithm (along with the window size). The evaluation files in the proper folder will have this same structure. If you want to evaluate the accuracy as well simply run `complete_evaluation.sh`: the results will be in the newly generated `evaluation.txt`. The model takes quite a long time to train, because it searches the hyperparameters in a parameter space. If you want to see the optimal results without the search you have to modify once again the script and assign the optimal parameters to the `crf` object.

* **biLSTM:**

## Status

The CRF model is fully built, results are under evaluation.
