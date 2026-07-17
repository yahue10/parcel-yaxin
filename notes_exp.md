# Notes observation

## From experiments of comparing multistage and two stage MRP

**high fluctuations across seasons, low fluctuation within the season**\
setting: base_mean_range=(20000, 40000), season_drift=(0.20,0.25), noise_frac=0.03 / 0.05


When all scenarios for each specific period, half scenaerios high demnad, haf scenario low demand, planned subcontracing is low, not preferred by MRP model. But tree model is still using planned.

MRP generally would purchase less but using more (planned and corrective) subcontracting. this might because without the flexibility of resource rebalancing, resource purcahsed has to stay in its hub. in the scenarios with high flucatuation, subcontracting is more beneficial since subcontracting is flexible even though it is expensive.

Tree model tends to use buy a bit more resources but use a bit less subcontracting.

No general trend yet for the rebalancing levels between tree model and MRP (not necessarily who is higher, this might depend on the demand scenarios, need to look deeper to see if there is trend)



**high fluctuations across seasons, high fluctuation within the season**\
Setting: base_mean_range=(20000, 40000), season_drift=0.25, noise_frac=0.15
Both multistage and MRP are using rare planned subcontracting, but more corrective one.

the expected two-stages costs are basically the same for these 2 models, regardless of the impacts of gap setting (0.01) of the solver

seems tree model tends to use more corrective subcontracitng over planned subcontracitng comprared to MRP. while MRP use mored

**low fluctuations across seasons, high fluctuation within the season**\
setting: base_mean_range=(20000, 40000), season_drift=0.25, noise_frac=0.01 

**low fluctuations across seasons, low fluctuation within the season**\
setting: base_mean_range=(20000, 40000), season_drift=0.25, noise_frac=0.01 


