# Notes observation

## From experiments of comparing multistage and two stage MRP

**high fluctuations across seasons, low fluctuation within the season**\
setting: base_mean_range=(20000, 40000), season_drift=0.25, noise_frac=0.01 



**high fluctuations across seasons, high fluctuation within the season**\
setting: base_mean_range=(20000, 40000), season_drift=0.25, noise_frac=0.15
both multistage and MRP are using rare planned subcontracting, but more corrective one.
the expected two-stages costs are basically the same for these 2 models, regardless of the impacts of gap setting (0.01) of the solver


**low fluctuations across seasons, high fluctuation within the season**\
setting: base_mean_range=(20000, 40000), season_drift=0.25, noise_frac=0.01 

**low fluctuations across seasons, low fluctuation within the season**\
setting: base_mean_range=(20000, 40000), season_drift=0.25, noise_frac=0.01 