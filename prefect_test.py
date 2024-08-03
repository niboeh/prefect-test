import glob
import pandas as pd
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from prefect import flow, task
from prefect.client import get_client
import asyncio
from prefect.deployments import Deployment
from prefect.infrastructure import Process
#from prefect.schedules import Schedule
#from prefect.schedules.models import IntervalSchedule
from datetime import timedelta
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#mlflow library
import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
from mlflow.models.signature import infer_signature

import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
#logging.getLogger("mlflow").setLevel(logging.WARNING)
#logging.basicConfig(level=logging.DEBUG)

#machine learning library
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Conv2D, MaxPooling2D

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, Convolution3D, MaxPool3D, GlobalMaxPool3D, GlobalAveragePooling3D, Flatten

import librosa
import soundfile as sf
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
import seaborn as sns
# 기본 작업 정의
@task
def load_dataset():
    # Load Dataset
    train_files = glob.glob("input/joint_sound_train/*.wav")
    train_csv = pd.read_csv("input/joint_sound_train.csv")

    audio_data_list = []
    sample_rate_list = []
    for file in train_files:
        audio_data, sample_rate = librosa.load(file)
        audio_data_list.append(audio_data)
        sample_rate_list.append(sample_rate)

    y_train = train_csv['label'].values
    label_encoder = LabelEncoder()
    Y_train = label_encoder.fit_transform(y_train)

    wn_files = glob.glob("input_wn/white_noise/*.wav")
    wn_csv = pd.read_csv("input_wn/wn_sound.csv")
    
    wn_audio_data_list = []
    wn_sample_rate_list = []
    for file in wn_files:
        wn_audio_data, wn_sample_rate = librosa.load(file)
        wn_audio_data_list.append(wn_audio_data)
        wn_sample_rate_list.append(wn_sample_rate)

    wn_y_train = wn_csv['label'].values
    wn_Y_train = label_encoder.fit_transform(wn_y_train)

    minus_files = glob.glob("input_aug/augmented_train/*.wav")
    minus_csv = pd.read_csv("input_aug/aug_sound.csv")

    minus_audio_data_list = []
    minus_sample_rate_list = []
    for file in minus_files:
        minus_audio_data, minus_sample_rate = librosa.load(file)
        minus_audio_data_list.append(minus_audio_data)
        minus_sample_rate_list.append(minus_sample_rate)

    minus_y_train = minus_csv['label'].values
    minus_Y_train = label_encoder.fit_transform(minus_y_train)

    return (audio_data_list, sample_rate_list, 
            Y_train, wn_Y_train, minus_Y_train, 
            wn_audio_data_list, wn_sample_rate_list, 
            minus_audio_data_list, minus_sample_rate_list)

@task
def change_mfcc_x(audio_data_list, sample_rate_list, wn_audio_data_list, wn_sample_rate_list, minus_audio_data_list, minus_sample_rate_list):
    audio_mfcc = []
    for y in audio_data_list:
        ret = librosa.feature.mfcc(y=y, sr=sample_rate_list[0])
        audio_mfcc.append(ret)

    wn_audio_mfcc = []
    for y in wn_audio_data_list:
        ret = librosa.feature.mfcc(y=y, sr=wn_sample_rate_list[0])
        wn_audio_mfcc.append(ret)

    minus_audio_mfcc = []
    for y in minus_audio_data_list:
        ret = librosa.feature.mfcc(y=y, sr=minus_sample_rate_list[0])
        minus_audio_mfcc.append(ret)

    mfcc_np = np.array(audio_mfcc, np.float32)
    wn_mfcc_np = np.array(wn_audio_mfcc, np.float32)
    minus_mfcc_np = np.array(minus_audio_mfcc, np.float32)

    mfcc_x = np.vstack([mfcc_np, wn_mfcc_np, minus_mfcc_np])
    mfcc_arr = np.array(mfcc_x, np.float32)

    print("MFCC Shape:", mfcc_arr.shape)
    return mfcc_arr

@task
def change_mfcc_y(Y_train, wn_Y_train, minus_Y_train):
    mfcc_y = np.hstack([Y_train, wn_Y_train, minus_Y_train])
    return mfcc_y

    
@task
def scaling_data(mfcc_arr):
    from sklearn.preprocessing import MinMaxScaler
    print('task-scaling_data')
    
    mfcc_scale = mfcc_arr.reshape(1547, 20 * 44)
    scaler = MinMaxScaler()
    scaler.fit(mfcc_scale)

    scaler_data = scaler.transform(mfcc_scale)
    scaler_data = pd.DataFrame(scaler_data)
    
    mfcc_array = mfcc_scale.reshape(1547, 20, 44)
    print(np.shape(mfcc_array))
    mfcc_x = np.expand_dims(mfcc_array, -1)
    print(mfcc_x.shape)
    
    return mfcc_x
    
@task
def data_split(mfcc_x, mfcc_y):
    print("tast_data_split")
    
    mfcc_x, mfcc_y = shuffle(mfcc_x, mfcc_y, random_state=10)
    train_x, test_x, train_y, test_y = train_test_split(mfcc_x, mfcc_y, test_size = 0.2, random_state=10)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.2, random_state=10)
    signature = infer_signature(train_x, train_y)

    return train_x, train_y, valid_x, valid_y, test_x, test_y

#@task
def train_model(params, train_x, train_y, valid_x, valid_y, test_x, test_y):
    input_shape = (20, 44, 1)
    # Define model architecture
    model = Sequential()

    model.add(Input(shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    
    model.add(Flatten())
    
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(params["dropout"]))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(params["dropout"]))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(params["dropout"]))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=params["lr"],
            #momentum=params["momentum"]
        ),
        loss="binary_crossentropy",
        metrics=['accuracy'],
    )

    # Train model with MLflow tracking
    with mlflow.start_run(nested=True):
        model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
        )
        # Evaluate the model
        eval_result = model.evaluate(valid_x, valid_y, batch_size=64)
        eval_bc = eval_result[0]
        eval_accuracy = eval_result[1]
        
        # Log parameters and results
        mlflow.log_params(params)
        mlflow.log_metric("binary_crossentropy", eval_bc)
        mlflow.log_metric("accuracy", eval_accuracy)

        # Log model
        mlflow.tensorflow.log_model(model, "model", signature=infer_signature(train_x, train_y))

        return {"loss": eval_bc, "accuracy": eval_accuracy, "status": STATUS_OK, "model": model}

#@task
def objective(params, train_x, train_y, valid_x, valid_y, test_x, test_y):
    # MLflow will track the parameters and results for each run
    result = train_model(
        params,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        test_x=test_x,
        test_y=test_y,
    )
    return result


space = {
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-1)),
    "dropout": hp.uniform("dropout", 0.1, 0.5),
    "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
    "epochs": hp.choice("epochs", [10, 30, 50]),
}

@task
def mlflow_start(space, train_x, train_y, valid_x, valid_y, test_x, test_y):
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8181")
    mlflow.set_experiment("/joint_damage_prefect3")
    
    mlflow.autolog()
    
    with mlflow.start_run():
        # Conduct the hyperparameter search using Hyperopt
        trials = Trials()
        best = fmin(
            fn=lambda params: objective(params, train_x, train_y, valid_x, valid_y, test_x, test_y),  # Use a lambda to pass params
            space=space,
            algo=tpe.suggest,
            max_evals=30,
            trials=trials,
        )
    
        # Fetch the details of the best run
        best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
    
        # Log the best parameters, loss, and model
        mlflow.log_params(best)
        mlflow.log_metric("eval_bc", best_run["loss"])
        mlflow.log_metric("accuracy", best_run["accuracy"])
        mlflow.tensorflow.log_model(best_run["model"], "model", signature=infer_signature(train_x, train_y))
    
        # Print out the best parameters and corresponding loss
        print(f"Best parameters: {best}")
        print(f"Best eval binary_crossentropy: {best_run['loss']}")


# 워크플로우 정의
@flow(log_prints=True)
def audio_processing_flow():
    
    (audio_data_list, sample_rate_list, 
     Y_train, wn_Y_train, minus_Y_train, 
     wn_audio_data_list, wn_sample_rate_list, 
     minus_audio_data_list, minus_sample_rate_list) = load_dataset()
    
    mfcc_x = change_mfcc_x(audio_data_list, sample_rate_list, wn_audio_data_list, wn_sample_rate_list, minus_audio_data_list, minus_sample_rate_list)
    mfcc_y = change_mfcc_y(Y_train, wn_Y_train, minus_Y_train)

    mfcc_x = scaling_data(mfcc_x)
    train_x, train_y, valid_x, valid_y, test_x, test_y = data_split(mfcc_x, mfcc_y)

    space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-1)),
        "dropout": hp.uniform("dropout", 0.1, 0.5),
        "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
        "epochs": hp.choice("epochs", [10, 30, 50]),
    }
    
    mlflow_start(space, train_x, train_y, valid_x, valid_y, test_x, test_y)
    
# 워크플로우 실행
if __name__ == "__main__":

    """
    audio_processing_flow.serve(name="prefect-test",
                                tags=["onboarding"],
                                interval=600)
    """
    audio_processing_flow.from_source(
        source="C:/heobin/jupyter-workspace/MLOps/mlflow",
        entrypoint="prefect_test.py:audio_processing_flow"
        ).deploy(
        name="test-deployment",
        work_pool_name="test-pool",
        )
