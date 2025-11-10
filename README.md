# PI_MESS_2025

Supplemental codes for "A Spatiotemporal Physics-Informed State-Space Model of Lake Temperature Profiles".

## Data Setup and Running the Code

1. **Download the data**

   All required datasets can be obtained from the USGS ScienceBase repository:  
   [https://www.sciencebase.gov/catalog/item/6206d3c2d34ec05caca53071](https://www.sciencebase.gov/catalog/item/6206d3c2d34ec05caca53071)

   The figure below shows the spatial distribution of lakes as well as two representative lakes (one deep and one shallow).
   ![Project Diagram](figures/F1_descriptive.png)

3. **Organize the data files**

   After downloading, organize the files within the `data` directory as follows:
   - `data/GCM` — contains **climate projection files**;
   - `data/meteo_csv_files` — contains **ALL NLDAS weather files**;  
   - Any remaining data files may be placed directly in the `data` directory.
  
4. **Train the PI-MESS model**

   The user should first create the training and testing datasets by running the create_training_dataset.py script. Then, run the `train_model.py` script. At the top of the `train_model.py` script, the toggle ''suffix'' controls which type of model is trained:
   - 'MOE' for the PI-MESS model;
   - 'PINN' for the PINN model;
   - 'NPI' for the NN model.
     
   Default is PI-MESS. This will save the chosen model inside the `models` directory within the `code` directory.

5. **Generate predictions**

   The model can be used to generate predictions over the contemporary period (up to 2022) or into the future. Climate projections (included in the data release) are available 2041-2059 and 2080-2099.
   - Run `predict_contemporary.py` for contemporary period predictions;
   - Run `forecast.py` for future predictions.
