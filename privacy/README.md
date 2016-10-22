# Learning private models with multiple teachers

This repository contains code to create a setup for learning privacy-preserving 
student models by transferring knowledge from an ensemble of teachers trained 
on disjoint subsets of the data for which privacy guarantees are to be provided.

Knowledge acquired by teachers is transferred to the student in a differentially
private manner by noisily aggregating the teacher decisions before feeding them
to the student during training.

A paper describing the approach is in preparation. A link will be added to this 
README when available.

## Dependencies

This model uses `TensorFlow` to perform numerical computations associated with 
machine learning models, as well as common Python libraries like: `numpy`, 
`scipy`, and `six`. Instructions to install these can be found in their 
respective documentations. 

## How to run

This repository supports the MNIST, CIFAR10, and SVHN datasets. The following
instructions are given for MNIST but can easily be adapted by replacing the 
flag `--dataset=mnist` by `--dataset=cifar10` or `--dataset=svhn`.
There are 2 steps: teacher training and student training. Data will be 
automatically downloaded when you start the teacher training. 

The following is a two-step process: first we train an ensemble of teacher
models and second we train a student using predictions made by this ensemble.

**Training the teachers:** first run the `train_teachers.py` file with at least
three flags specifying (1) the number of teachers, (2) the ID of the teacher
you are training among these teachers, and (3) the dataset on which to train. 
For instance, to train teacher number 10 among an ensemble of 100 teachers for 
MNIST, you use the following command:

```
python train_teachers.py --nb_teachers=100 --teacher_id=10 --dataset=mnist
```

Other flags like `train_dir` and `data_dir` should optionally be set to
respectively point to the directory where model checkpoints and temporary data
(like the dataset) should be saved. The flag `max_steps` (default at 3000) 
controls the length of training. See `train_teachers.py` and `deep_cnn.py` 
to find available flags and their descriptions.

**Training the student:** once the teachers are all trained, e.g., teachers 
with IDs `0` to `99` are trained for `nb_teachers=100`, we are ready to train
the student. The student is trained by labeling some of the test data with 
predictions from the teachers. The predictions are aggregated by counting the
votes assigned to each class among the ensemble of teachers, adding Laplacian 
noise to these votes, and assigning the label with the maximum noisy vote count
to the sample. This is detailed in function `noisy_max` in the file 
`aggregation.py`. To learn the student, use the following command:

```
python train_student.py --nb_teachers=100 --dataset=mnist --stdnt_share=5000
```

The flag `--stdnt_share=5000` indicates that the student should be able to
use the first `5000` samples of the dataset's test subset as unlabeled
training points (they will be labeled using the teacher predictions). The 
remaining samples are used for evaluation of the student's accuracy, which
is displayed upon completion of training.

## Alternative deeper convolutional architecture

Note that a deeper convolutional model is available. Both the default and 
deeper models graphs are defined in `deep_cnn.py`, respectively by 
functions `inference` and `inference_deeper`. Use the flag `--deeper=true` 
to switch to that model when launching `train_teachers.py` and 
`train_student.py`. 

## Contact

To ask questions, please email `nicolas@papernot.fr` or open an issue on 
the `tensorflow/models` issues tracker. Please assign issues to 
[(@npapernot)](https://github.com/npapernot).
