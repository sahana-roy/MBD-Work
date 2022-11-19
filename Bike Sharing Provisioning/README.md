# **Prediction Model: Bike Sharing Provisioning**

Here, I experiment with PyCaret and Streamlit. In order to present the code, I created Reveal.js slides as well.

### **About the challenge**

<em>The administration of Washington D.C wants to make a deeper analysis of the usage of the bike-sharing service present in the city in order to build a predictor model that helps the public transport department anticipate better the provisioning of bikes in the city. For these purposes, some data is available for the years 2011 and 2012.</em>

### **Main goals**

As part of the team of the hired consultancy firm, my work was to build an interactive, insightful and complete report about the bike-sharing service in the city for the head of transportation services of the local government. As part of the requirements, there are two big points:

- The customer wants a deep analysis of the bike-sharing service. He wants to know how the citizens are using the service in order to know if some changes must be made to optimize costs or provide a better service.
- The customer also wants a predictive model able to predict the total number of bicycle users on an hourly basis. They believe this will help them in the optimization of bike provisioning and will optimize the costs incurred from the outsourcing transportation company.

## **About the Data**

- `instant`: record index
- `dteday` : date
- `season` : season (1:springer, 2:summer, 3:fall, 4:winter)
- `yr` : year (0: 2011, 1:2012)
- `mnth` : month ( 1 to 12)
- `hr` : hour (0 to 23)
- `holiday` : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
- `weekday` : day of the week
- `workingday` : if day is neither weekend nor holiday is 1, otherwise is 0.
+ `weathersit` : 
	- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
	- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
	- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
	- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- `temp` : Normalized temperature in Celsius. The values are divided to 41 (max)
- `atemp`: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
- `hum`: Normalized humidity. The values are divided to 100 (max)
- `windspeed`: Normalized wind speed. The values are divided to 67 (max)
- `casual`: count of casual users
- `registered`: count of registered users
- `cnt`: count of total rental bikes including both casual and registered
