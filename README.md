[//]: # (Image References)

[image1]: ./images/pipeline.png "ASR Pipeline"
[image2]: ./images/select_kernel.png "select aind-vui kernel"

## Project Overview

In this notebook, I have built a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline.

![ASR Pipeline][image1]


## Project 

### Amazon Web Services

This project requires GPU acceleration to run efficiently. 

1. Follow the Cloud Computing Setup instructions lesson to create an EC2 instance.

2. Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
```
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzvf dev-clean.tar.gz
wget http://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzvf test-clean.tar.gz
mv flac_to_wav.sh LibriSpeech
cd LibriSpeech
./flac_to_wav.sh
```

3. Create JSON files corresponding to the train and validation datasets.
```
cd ..
python create_desc_json.py LibriSpeech/dev-clean/ train_corpus.json
python create_desc_json.py LibriSpeech/test-clean/ valid_corpus.json
```

4. Start Jupyter:
```
jupyter notebook --ip=0.0.0.0 --no-browser
```

5. Look at the output in the window, and find the line that looks like: `http://0.0.0.0:8888/?token=3156e...` Copy and paste the **complete** URL into the address bar of a web browser (Firefox, Safari, Chrome, etc). Before navigating to the URL, replace 0.0.0.0 in the URL with the "IPv4 Public IP" address from the EC2 Dashboard.


### Local Environment Setup

You should run this project with GPU acceleration for best performance.

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/udacity/AIND-VUI-Capstone.git
cd AIND-VUI-Capstone
```

2. Create (and activate) a new environment with Python 3.6 and the `numpy` package.

	- __Linux__ or __Mac__: 
	```
	conda create --name aind-vui python=3.5 numpy
	source activate aind-vui
	```
	- __Windows__: 
	```
	conda create --name aind-vui python=3.5 numpy scipy
	activate aind-vui
	```

3. Install TensorFlow.
	- Option 1: __To install TensorFlow with GPU support__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step and only need to install the `tensorflow-gpu` package:
	```
	pip install tensorflow-gpu==1.1.0
	```
	- Option 2: __To install TensorFlow with CPU support only__,
	```
	pip install tensorflow==1.1.0
	```

4. Install a few pip packages.
```
pip install -r requirements.txt
```

5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
	```
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```
	- __Windows__: 
	```
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
	```
	- __NOTE:__ a Keras/Windows bug may give this error after the first epoch of training model 0: `‘rawunicodeescape’ codec can’t decode bytes in position 54-55: truncated \uXXXX `. 
To fix it: 
		- Find the file `keras/utils/generic_utils.py` that you are using for the capstone project. It should be in your environment under `Lib/site-packages` . This may vary, but if using miniconda, for example, it might be located at `C:/Users/username/Miniconda3/envs/aind-vui/Lib/site-packages/keras/utils`.
		- Copy `generic_utils.py` to `OLDgeneric_utils.py` just in case you need to restore it.
		- Open the `generic_utils.py` file and change this code line:</br>`marshal.dumps(func.code).decode(‘raw_unicode_escape’)`</br>to this code line:</br>`marshal.dumps(func.code).replace(b’\’,b’/’).decode(‘raw_unicode_escape’)`

6. Obtain the `libav` package.
	- __Linux__: `sudo apt-get install libav-tools`
	- __Mac__: `brew install libav`
	- __Windows__: Browse to the [Libav website](https://libav.org/download/)
		- Scroll down to "Windows Nightly and Release Builds" and click on the appropriate link for your system (32-bit or 64-bit).
		- Click `nightly-gpl`.
		- Download most recent archive file.
		- Extract the file.  Move the `usr` directory to your C: drive.
		- Go back to your terminal window from above.
	```
	rename C:\usr avconv
    set PATH=C:\avconv\bin;%PATH%
	```

7. Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
	- __Linux__ or __Mac__: 
	```
	wget http://www.openslr.org/resources/12/dev-clean.tar.gz
	tar -xzvf dev-clean.tar.gz
	wget http://www.openslr.org/resources/12/test-clean.tar.gz
	tar -xzvf test-clean.tar.gz
	mv flac_to_wav.sh LibriSpeech
	cd LibriSpeech
	./flac_to_wav.sh
	```
	- __Windows__: Download two files ([file 1](http://www.openslr.org/resources/12/dev-clean.tar.gz) and [file 2](http://www.openslr.org/resources/12/test-clean.tar.gz)) via browser and save in the `AIND-VUI-Capstone` directory.  Extract them with an application that is compatible with `tar` and `gz` such as [7-zip](http://www.7-zip.org/) or [WinZip](http://www.winzip.com/). Convert the files from your terminal window.
	```
	move flac_to_wav.sh LibriSpeech
	cd LibriSpeech
	powershell ./flac_to_wav.sh
	```

8. Create JSON files corresponding to the train and validation datasets.
```
cd ..
python create_desc_json.py LibriSpeech/dev-clean/ train_corpus.json
python create_desc_json.py LibriSpeech/test-clean/ valid_corpus.json
```

9. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `aind-vui` environment.  Open the notebook.
```
python -m ipykernel install --user --name aind-vui --display-name "aind-vui"
jupyter notebook vui_notebook.ipynb
```

10. Before running code, change the kernel to match the `aind-vui` environment by using the drop-down menu.  Then, follow the instructions in the notebook.

![select aind-vui kernel][image2]


### My final models

The final project architectures implemented are:

#### Model 0: Simple RNN
The model is trained for at least 20 epochs, the loss values are saved in `model_0.pickle` and the trained weights for the model specified in `simple_rnn_model` are stored in `model_0.h5`.

#### Model 1: RNN + TimeDistributed Dense
The model is trained for at least 20 epochs, the loss values are saved in `model_1.pickle` and the trained weights for the model specified in `rnn_model` are stored in `model_1.h5`.   	

#### Model 2: CNN + RNN + TimeDistributed Dense

The model is trained for at least 20 epochs, the loss values are saved in `model_2.pickle` and the trained weights for the model specified in `cnn_rnn_model` are stored in `model_2.h5`.   

#### Model 3: Deeper RNN + TimeDistributed Dense

The model is trained for at least 20 epochs, the loss values are saved in `model_3.pickle` and the trained weights for the model specified in `deep_rnn_model` are stored in `model_3.h5`. 

#### Model 4: Bidirectional RNN + TimeDistributed Dense

The model is trained for at least 20 epochs, the loss values are saved in `model_4.pickle` and the trained weights for the model specified in `bidirectional_rnn_model` are stored in `model_4.h5`. 

#### Final Model
The final model has the following architecture:   	|

- 1 convolutional layer with batch normalization layer
- 1 bidirectional layer with a GRU cell, dropout rate of 0.5 followed batch normalization layer
- 1 Time Distributed layer, fully-connected dense layer that produces a softmax output with a learning rate of 0.03

## Special Thanks

The `create_desc_json.py` and `flac_to_wav.sh` files are borrowed from the [ba-dls-deepspeech](https://github.com/baidu-research/ba-dls-deepspeech) repository, along with some functions used to generate spectrograms.
