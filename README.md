# PI_MESS_2025

## Data Setup and Running the Code

1. **Download the data**

   All required datasets can be obtained from the USGS ScienceBase repository:  
   [https://www.sciencebase.gov/catalog/item/6206d3c2d34ec05caca53071](https://www.sciencebase.gov/catalog/item/6206d3c2d34ec05caca53071)

2. **Organize the data files**

   After downloading, organize the files within the `data` directory as follows:
   - `data/GCM` — contains **climate projection files**  
   - `data/meteo_csv_files` — contains **ALL NLDAS weather files**  
   - Any remaining data files may be placed directly in the `data` directory
  
3. **Train the PI-MESS model**

   Run the `train_model.py` script. At the top of the script, the toggle ''suffix'' controls which type of model is trained: 'moe' for the PI-MESS model, 'pinn' for the PINN model, and 'npi' for the NN model. Default is PI-MESS. This will save the chosen model inside the `models` directory within the `code` directory.
