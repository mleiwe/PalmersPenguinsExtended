# PalmersPenguinsExtended
ML Zoomcamp Capstone1 project, where I attempt to find the species of penguin, but make it a little harder. Guidance can be found [here](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/projects#capstone-1)

**Findings TLDR**: I found that the multi-layer perceptron classifier available in scikit-learn was the most accurate version. I then created the ability to deploy this locally using a docker image. (Instructions and Demo video can be seen below)

# Introduction
Palmer's Penguins is a dataset that is similar to the classic [Fischer's Iris' dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) which serves as a great means of testing out multi-class separation. In this extended dataset there are still three classes of penguins (Adelie, Chinstrap, and Gentoo) but there are now extra fields of information to explore (`life_stage`, `diet`, and `health_metric`). This is a fun challenge to see if you can combine continuous data types along with ordinal, and categorical in order to make a good ML classifier.

## Data
The data was originally published in [Gorman KB, Williams TD, Fraser WR (2014) Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus Pygoscelis). PLoS ONE 9(3): e90081. doi:10.1371/journal.pone.0090081](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0090081). I pulled the data from Kaggle here, however, copies of both the original and extended datasets are available in this repository too.

### Capture information (Fieldwork)
Fieldwork for on Palmer Archipelago near Anvers Island and Palmer Station, a US-supported research base (b, c). Penguin rookeries were located at Dream (Adélie and Chinstrap), Torgersen (Adélie only), and Biscoe (Adélie and Gentoo) Islands. Since I like a challenge, and it is easy to identify the penguin species in Torgersen, I will blind the ML solutions to the island the penguin was captured on.
![image](https://github.com/mleiwe/PalmersPenguinsExtended/assets/29621219/481007a4-8430-459c-828a-f910b27374a7)

### Extended Data Column Descriptions
The dataset consists of the following columns:

* `species`: Species of the penguin (Adelie, Chinstrap, Gentoo)
* `island`: Island where the penguin was found (Biscoe, Dream, Torgensen)
* `sex`: Gender of the penguin (Male, Female)
* `diet`: Primary diet of the penguin (Fish, Krill, Squid)
* `year`: Year the data was collected (2021-2025)
* `life_stage`: The life stage of the penguin (Chick, Juvenile, Adult)
* `body_mass_g`: Body mass in grams
* `bill_length_mm`: Bill length in millimeters
* `bill_depth_mm`: Bill depth in millimeters
* `flipper_length_mm`: Flipper length in millimeters
* `health_metrics`: Health status of the penguin (Healthy, Overweight, Underweight)

I attempt to predict the `species` of the penguin using the available features of a penguin. i.e. I don't use the `year` or `island` as they are not biological features. Furthermore the geographic distribution of the penguins can bias the distribution see image below and map above ![image](https://github.com/mleiwe/PalmersPenguinsExtended/assets/29621219/1f41387c-515b-4f0c-8ac2-9fb3d912a996)

# EDA
There is a version of the notebook available within this repository. However, it took a long time to run locally instead I suggest you look at the [Kaggle version](https://www.kaggle.com/mnl2022/palmerspenguineda)
The following libraries are required (they arrive pre-loaded within a Kaggle notebook)
* numpy
* pandas
* matplotlib.pyplot
* seaborn (NB Issues arise if you use versions below v0.12.2)
* scikit-learn
* XGBoost
* LightGBM
* 
# Training

# How to deploy locally
You can run this service locally using the docker file (provided you have docker installed).

## Build the machine
1. Open a terminal
2. Change the directory to where you have downloaded the app
3. Build the docker with docker build -t penguin_prediction .
4. Run the docker with `docker run -it --rm -p 9696:9696 penguin_prediction
## Query the model
1. Open a new terminal
2. Navigate to the directory where the repository was downloaded
3. Type python PredictTest.py If needed feel free to change the values for the query
Demonstration

## Local Demonstration
See video [here](https://drive.google.com/file/d/1_aqoiscdBVZZ5kLVsnTfa0E1lLpiGdSy/view?usp=sharing)
