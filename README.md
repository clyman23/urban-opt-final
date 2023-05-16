# Urban Opt Final Project
 
## Bike rebalancing with e-bikes
 
In this project we want to analyze how ebikes change rebalancing operations and costs because it adds an   operation for the rebalancing authority. Here we use the "Repositioning in Bike-sharing systems" paper by  Jiaqi Liang to build on top of the data they used for a station network of 30 stations, trip information   for expected returns and rentals, and predefined rebalancing fleet.
 
 We introduce electric bikes in this model to add another layer of complexity and challenge in the          rebalancing operations as 1. electric bikes are more valuable for business because of the higher revenuew  generated per successful ebike trip, and 2. rental demand for electric bikes depends not only on the       availability of e-bikes at the station, but also on the battery being sufficiently charged.
 
 So, for these rebalancing operations, we are not only picking up and dropping off classic bikes between    stations, but also swapping ebike batteries with fully charged batteries during rebalancing. This means    some vehicle capacity is now taken up by batteries instead of bikes, limiting vehicle capacity for         rebalancing.
 
 We model and analyze this by randomly assigning a subset of bikes in the bike sharing-system (BSS)         dataset to be electric bikes and modifying vehicle capacity to account for battery swapping. We then test  sensitivities of the tradeoff between using truck space for bikes (for rebalancing) or for batteries (for  ebikes). While only 20\% of the CityBike fleet is electric, those ebikes makeup close to 40\% of all       rides, demonstrating the importance of fairly and accurately rebalancing these important bikes.
 
 
 ## Setup
 
 The `Initial_Inven.json` file provides an initial inventory of the bikes at the stations.
 
 `environment.yml` outlines the environment variables and dependencies. 
 
 `input_data.ipynb` takes in the BSS data and combines the trip information from 500 days using the simu0*.json data into a single dataframe with incremental arrival and departure times from day 0 - day 499 split into 15-minute time windows. Here, we also randomly assign 40% of the data to be for electric bikes (based on secondary research from Montreal, where 40% of bike ridership demand is for ebikes). We then use this data to calculate expected rental and returns demand for classic and electric bikes. The output files `rentals.csv` and `returns.csv` show the expected rental and returns for the baseline problem, without the split into classic and electric bikes. The files `rentals_classic.csv` and `returns_classic.csv` show the expected rental and returns for classic bikes and `rentals_electric.csv` and `returns_electric.csv` show the expected rental and returns for electric bikes after the split. 
 
`baseline_problem.py` is the model formulation for the baseline bike rebalancing problem, without the split  into classic and electric bikes.
 
`eletric_bike_problem.py` is the model formulationf for the bike rebalancing problem with the baseline bike  data split into 80% classic bikes and 20% electric bikes.
 
To run the `baseline_problem.py` file, adjust the input variables as needed, and the output filepaths. We set the parameters for the rebalancing fleet (i.e. vehicle capacity, number of vehicles, etc.) and other   initial conditions (distance between stations, initial number of bikes in each vehicle, etc.). To run the  model:
 
```
python baseline_problem.py
```
 
Similarly, adjust the parameters as needed in `electric_bike_problem.py` adjusting parameters for classic bikes and for ebikes, and run:
 
```
python electric_bike_problem.py
```
 
 

