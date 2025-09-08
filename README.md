# Resource Allocation using Multimodal AI with Misinformation Mitigation in Healthcare
This project presents a comprehensive multimodal AI framework designed to predict mental health severity, particularly in the context of Alzheimer's disease, by leveraging diverse data sources:

* Neuroimaging: Structural and functional data from MRI scans
* Vocal Biomarkers: Audio features (e.g., MFCCs) capturing speech characteristics correlated with cognitive decline
* Physiological & Behavioural Signals: Including  breathing rate, heart rate and motor tapping
* Clinical Scores: Such as PHQ scores, indicating depression and mood-related symptoms

These multimodal inputs are used to train a predictive model that estimates cognitive severity, supporting early intervention and optimised care.
Beyond the clinical prediction model, this project also simulates the spread of health misinformation via social media networks, modelling how exposure to inaccurate or harmful information can impact perception and decision-making. This simulation is used to adjust severity predictions and dynamically inform resource allocation — ensuring patients most at risk (both medically and informationally) are prioritised for treatment.

The entire pipeline is visualised through a Streamlit app, including model predictions, explainability tools (SHAP, LIME), and a real-time network-based misinformation simulation.

[Link to App](https://cognitiveaitriage-upcnmprvydp5bhgfjpox8k.streamlit.app/)

## Key Features
* Multimodal Input: Combines features from audio (MFCC), image (ResNet), physiological signals, gaze, pose, and PHQ scores.
* Severity Prediction: Uses Random Forest Regression to predict PHQ severity.
* Misinformation Spread Simulation: Network-based misinformation modelling to adjust severity scores.
* Resource Allocation: Dynamically allocate limited treatment resources based on adjusted scores.
* Explainability: Visual explanations using SHAP and LIME for transparent predictions.
* Web App: Interactive UI built with Streamlit for simulations, explanations, and visualisations.

## Interpreting Results

### Physiological Markers

| **Dementia Risk** | **Breathing Rate (bpm)** | **Tapping (Hz)** | **Heart Rate (bpm)** |
|-------------------|--------------------------|------------------|----------------------|
| Low               | > 16                     | > 1              | > 60                 |
| Medium            | 12 - 16                  | 0.75 - 1         | 50 - 60              |
| High              | < 12                     | < 0.75           | < 50                 |

**Physiological Marker Controls**
- **Breathing Rate (bpm)**: Represents breaths per minute. Lower values may indicate high risk for dementia.
- **Tapping Rate (taps/sec)**: Reflects tapping speed. A slower rate may be correlated with cognitive decline.
- **Heart Rate (bpm)**: Represents beats per minute. A lower heart rate may indicate a higher dementia risk.

Using the sliders and inputs, user adjusts the physiological markers so as to simulate dementia risk

### Mapping MFCC Mean Ranges to Depression Severity

MFCC Mean Range | Interactive Speech Profile | Possible Depression Severity |
--------------- | -------------------------- | -----------------------------
\> -10          | Clear/expressive/energetic | None/Minimal                 |
\-10 - 18       | Reduced variability/energy | Mild                         |
\-18 - 28       | Flat/monotonic tone        | Moderate                     |
\-28 - 30       | Dull/low-affect/low-volume | Moderately Severe            |
\< -30          | Flat/withdrawn             | Severe                       |

The mean of the MFCC coefficients across timeframes are calculated, representing the overall spectral features of the audio so as to classify speech patterns related to depression in order to assess the severity as well as clinical priority (for follow-ups or intervention) of a user.

These thresholds are a guide and are based on existing literature that describe general trends between MFCC features and depression.  As this speech analysis is intended to be used with other data modalities such as physiological data (tapping, breathing and heart rate), imaging and text-based sentiment analysis from patient interviews, it offers a holistic approach to depression/Alzheimer's risk assessment.  However, this approach may benefit from validation or tuning with a large-scale dataset.  Additionally, as variations in speech due to language, culture or individual characteristics, model performance may not generalise well to all individuals or diverse populations.  To improve accuracy, inclusion of additional ausio samples or features, like pauses, accents, speech patterns or emotional tone may help issue a more nuanced analysis of depression through the capture of the emotional context of a user's speech.  Furthermore, cross-validating the severity thresholds by incorporating clinical professionals' feedback may also help improve data accuracy.  


### Misinformation Risk Score (MRS): Mental Health Influence

MRS Mean Range  | Interpretation                        | Possible Impact on Mental State |
--------------- | ------------------------------------- | --------------------------------
0.00 - 0.10     | Minimal exposure/resistance           | None/Minimal                    |
0.11 - 0.25     | Low exposure/weak influence           | Mild                            |
0.26 - 0.50     | Moderate exposure/internalisation     | Moderate                        |
0.51 - 0.75     | High exposure/psychological impact    | Moderately Severe               |
0.76 - 1.00     | Very high exposure/echo chamber efect | Severe                          |

### Depression Severity

### Raw Severity Score: PHQ-8 Depression Severity Score Prediction from the Model

Score Range   | Interpretation
-----------   | --------------
0 - 4         | None/Minimal
5 - 9         | Mild
10 -14        | Moderate
15 - 19       | Moderately Severe
20 - 24       | Severe

Raw depression severity is a baseline measure of depression based on model predictions using features related to depressive symptoms.

NB: The model achieves an R² of 0.71 and MAE of 2.43 on the validation set, showing strong ability to learn PHQ-8 severity from physiological and behavioural features.  However, generalisation to the test set suggests the need for more diverse training data or regularisation.

### Adjusted Severity

Patient prioritisation is based on an adjusted severity score which factors in the raw severity score (the base depression severity score based on PHQ-8 score) and adjustments of other factors such as misinformation risk and physiological risk (e.g., breathing, tapping and heart rate).  Prioritisation is done by compaaring the adjusted severity to a threshold as it is likely that this score provides a more accurate reflection of the true risk.  Those with higher adjusted severity scores are prioritised for treatment as depression is a strong risk factor for dementia, particularly when it becomes chronic or severe (indicated by the "✅ Yes" in the priority column).  Populations with high adjusted depression severity have a higher likelihood of cognitive decline as prolonged depression may impact brain function and accelerate neurodegeneration.  By prioritising early intervention to prevent the cognitive symptoms observed in the early stages of dementia (e.g., memory loss or concentration difficulties), this model can be used to help address the root causes of depression (including misinformation and physiological health), reduce dementia risk and subsequently prevent further cognitive deterioration.  Adjusting the treament capacity slider reallocates resources and the number of patients potentially treated.

The adjusted severity for each patient is calculated using the formula:

<img src="https://github.com/ihe-k/Cognitive_AI_Triage/blob/main/Project_3_Eq_1.png?raw=true" width="250" />

- Misinformation risk represents a factor that reduces the severity of depression based on misinformation spread.
- Physiological risk highlights how breathing, tapping or heart rate exacerbates or alleviates the severity of depression.
- Alpha is a weighting factor that determines the extent of influence that physiological data has on adjusted severity.

The adjusted risk is always lower than the raw score as the model simulated the risk of misinformation spread across the population.  A higher  misinformation risk will reduce the adjusted severity risk as misinformation tends to artificially inflate perceived depression severity.  When misinformation is corrected for, the severity of depressive symptoms is adjusted downward. Additionally, incorporating physiological data reflects that physiological health buffers or moderated the psychological impacts of depressive symptoms and help paint a more nuanced picture of a patient's health.  

## Project Structure
```plaintext
project/
│
├── requirements.txt
├── train_model.py
├── artifacts/
│   └── alzheimer_classifier.pkl
│   └── severity_model.pkl
│   └── validation_plot.png
└── streamlit_inference.py
```

## Installation
### 1. Clone the Repository
```plaintext
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
### 2. Install Dependencies
```plaintext
Use pip to install all required packages:
pip install streamlit librosa torch torchvision shap lime scikit-learn matplotlib networkx pandas pillow
```
## Running the App
Once dependencies are installed and your .npy data files are in place:
```plaintext
streamlit run your_script_name.py
```
Then open the local Streamlit URL (usually http://localhost:8501) in your browser.

## How It Works
### 1. Feature Extraction
* extract_mfcc_features(audio_files) – Audio MFCC extraction via librosa
* extract_resnet_features(image_files) – Image embeddings using pretrained ResNet-18
* Physiological markers are synthetically simulated (breathing, tapping, heart rate)

### 2. Model Training
* All features are concatenated and standardised
* Random Forest Regressor trained to predict PHQ scores
* Model performance validated using MAE, RMSE and R²

### 3. Misinformation Simulation
Simulates misinformation spread using a Barabási–Albert network model:

* Nodes: Patients
* States: Susceptible (S), Infected (I) and Recovered (R)
* Adjusts predicted severity scores based on misinformation prevalence

### 4. Resource Allocation
* Patients with the highest adjusted severity are prioritised
* Limited capacity is configurable via the Streamlit app

### 5. Explainability
* SHAP: Visualises individual feature contributions for predictions
* LIME: Explains local model behaviour for a selected patient

## Validation
* Validation plot (validation_plot.png) shows predicted vs actual PHQ severity on the test set.
* Automatically displayed in the Streamlit app.

## Example Outputs
* PHQ Score Prediction Graphs
* SHAP & LIME Patient Explanation Visuals
* Misinformation Network Evolution
* Risk-Based Resource Allocation Charts

## Notes
* Ensure all .npy feature files are correctly shaped and aligned.
* The train_phq_score_*.npy files are merged to generate ground truth severity labels.
* Misinformation risk is integrated directly into adjusted predictions.

## Acknowledgments
* librosa – Audio analysis
* torchvision – Pretrained image models
* SHAP\LIME – Model explainability
* Streamlit – Interactive app
* scikit-learn – ML modelling

## Contact
For questions or collaboration requests, please contact me here or open an issue.
