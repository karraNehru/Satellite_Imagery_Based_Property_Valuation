# Tabular and Satellite Imagery-Based Property Valuation Using ML
## Overview
Predict property prices using structured tabular data and satellite imagery by combining XGBoost and CNN-based visual embeddings. The project follows a **leak-free, reproducible multimodal pipeline** and emphasizes **model explainability**.

## Problem Statement
The goal is to build a model that accurately values assets by integrating "curb appeal" and neighborhood characteristics (like green cover or road density) into traditional pricing models.

This project moves beyond standard data analysis by combining two different types of data (numbers and images) into a single, powerful predictive system.

## Data Overview
### Tabular Data 
|# | Column         | Dtype   |
|--|----------------|---------|
|1 | id             | int64   |
|2 | date           | object  |
|3 | price( target colmn )| int64   |
|4 | bedrooms       | int64   |
|5 | bathrooms      | float64 |
|6 | sqft_living    | int64   |
|7 | sqft_lot       | int64   |
|8 | floors         | float64 |
|9 | waterfront     | int64   |
|10| view           | int64   |
|11| condition      | int64   |
|12| grade          | int64   |
|13| sqft_above     | int64   |
|14| sqft_basement  | int64   |
|15| yr_built       | int64   |
|16| yr_renovated   | int64   |
|17| zipcode        | int64   |
|18| lat            | float64 |
|19| long           | float64 |
|20| sqft_living15  | int64   |
|21| sqft_lot15     | int64   |

- **dtypes:** float64(4), int64(16), object(1)
- **Total Train data:** 16209 rows , 21 cols ( including target )
- **Total Test data:** around 5000 rows , 20 cols
### Imagery Data
The image dataset used in this project is generated using the `data_fetcher.py` script.  
This script retrieves satellite or map images corresponding to each property entry in the tabular dataset using the latitude and longitude cordinates.

**Details:**
- **Source Tool:** Mapbox API  
- **Image Resolution:** 512 × 512 pixels  
- **Zoom Level:** 16.5  
- **image Format:** `.jpg`  
- **File Naming Convention:** Each image is saved as `<id>.jpg`, where `id` matches the property identifier in the tabular data.  

## Modeling Approaches
#### 1. Tabular Baseline
- XGBoost trained on engineered and processed tabular features
- This model serves as a strong baseline
#### 2. Early Fusion (Multimodal)
- EfficientNet-B0 used as a frozen image encoder
- Image embeddings compressed using PCA (180 components)
- Concatenated with tabular features
- Trained using XGBoost
#### 3. Late Fusion
- Separate tabular and image branches combined using an MLP
- Evaluated to test nonlinear fusion effectiveness

## Result summary and Explainability
Refer [The Report](https://github.com/karraNehru/Satellite_Imagery_Based_Property_Valuation/edit/main/README.md) for complete summary and clear Explaination

## Repository Structure

```
Data_fetcher.py/        # Image Data fetching using lat , long
preprocessing.ipynb/    # EDA, feature engineering, Preprocessing , train and validation splitting
model_training/         # training of Tabular, Early, Late fusion models , Exatracting Embeddings
23118039_final.csv/     # prediction csv of test data
23118039_report.pdf/    # project report
```

## **How To Run?**
NOTE : Replace and ensure the File Path according to your directory in the scripts and notebooks Before Running.

also ensure all your files are in same directory else provide full path.
1. Install [Requirements]()
In windows Powershell
```bash
#create virtual environment
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
In Jupyter Notebook code cell
```bash
!pip install -r requirements.txt
```

2. Run data_fetcher.py


   Running `datafetcher.py` will:
   1. Connect to the Mapbox API and fetch images based on property coordinates.
   2. Store all fetched images inside an `images/` directory.
   3. Automatically create `train/` and `test/` subfolders for dataset splitting.
Note: ensure your .env contains Map Box Api Token as MAPBOX_KEY = "pk.abc-123n-eh-ru..."
```bash
#windows powershell opened with BASE_DIR Location
python data_fetcher.py
```
or run in Jupyter Notebook code cell
```bash
%run data_fetcher.py
```
**Output Folder Structure Example:**
```bash
images/
├── train/
│ ├── 1234567890.jpg
│ ├── 1234567891.jpg
│ └── ...
└── test/
├── 1234567999.jpg
├── 1234568000.jpg
└── ...
```
3. Run preprocessing.ipynb
   In powershell
   ```bash
   jupyter notebook preprocessing.ipynb
   ```
   or open [preprocessing.ipynb]() in jupyter notebook and Run all cells
4. Run model_training.ipynb
   In powershell
   ```bash
   jupyter notebook model_training.ipynb
   ```
   or open [model_training.ipynb]() in jupyter notebook and Run all cells






