# A-Crime-Probability-Prediction-System-for-Bay-Area
## Introduction and Motivation

Crime is a significant issue in the Bay area. As a group of students who are new to this area, we are all excited about the nice weather and the diverse culture. However, after just a few weeks of being at Berkeley, we realized that things are not as heavenly as we thought. Some of our friends got robbed on the way home near the north side of the campus, and car windows were frequently smashed open if you leave anything valuable in the car. Since we are very likely to stay in the Bay area after graduation, the more about public safety we know the better. 

After we settled on the topic, it came to us that our project could have a significant impact on all Berkeley students, residents and officials in the area. For someone who is new to the area, it could be overwhelming to get to know each area. If we can predict what contributes to high crime frequency, it might reduce their chances of getting into unfortunate scenarios. The project may have larger impact once the city officials plan their resources accordingly, and we are excited about what we can do to make this area better. 

## Data

Since our goal concerns about public safety, we would want to gather official statistics about criminal reports in the San Francisco area. DataSF(https://datasf.org/) provided an excellent source of information for our research. There are several datasets we found especially helpful in conducting our research. 
Police Department Incident Reports: This dataset includes incident reports from 2018 to present. Some particularly important features are:

- Description: 
  - Datetime: The date and time when the incident occurred.

- Incident: 
  - Category: A category mapped on to the Incident Code used in statistics and reporting. Mappings provided by the Crime Analysis Unit of the Police Department.
  - Sub-category: A subcategory mapped on to the Incident Code used in statistics and reporting. These nest inside the Category field. Mappings provided by the Crime Analysis Unit of the Police Department.
  - Incident Description: The description of the incident that corresponds with the Incident Code. These are generally self-explanatory.
- Geographical:
  -	Neighborhood: The name of the neighborhood
  - Police Supervisor District: The Police District reflecting current boundaries of the neighborhoods
  - Latitude & Longitude: The latitude and longitude of a certain neighborhood
  
In order to improve the accuracy of the prediction, we also included other demographic data from several sources like Censusreporter (https://censusreporter.org), Point2homes (https://www.point2homes.com) and Areavibes (https://www.areavibes.com) to describe the characteristics of the neighborhoods in the San Francisco area: 
- Population: The number of people in a certain neighborhood in 2018
- Male/Female Ratio: The ratio is equal to the number of males to the number of females in a certain neighborhood in 2018
- Area: The area of the certain neighborhood (Square mile) 
- Median Age: The median age of the population in a certain neighborhood in 2018
- Median Household Income: The median household income of the household in a certain neighborhood in 2018 (Dollar) 

## Analytics Model

We took several steps and modifications in the process. We tried both regression methods, which could score different areas, and classification methods, which aim to classify the danger levels. After several attemptions, we find out that classification performs better for this problem. An overview of each step will be introduced in this section, and detailed results are in the 

In order to establish a safe area recommendation system, we plan to apply all the analytic models we have learnt from the courses: regression models, CART, random forest, boosting, and collaborative filtering. For each regression model we will report the MAE, RMSE and OSR2 for the test set to detect the performance of each model. For classification model, we will apply Accuracy for the test set to show its performance. And we will offer our exclusive recommendation based on the data.

In conclusion, switching from regression to classification problems dramatically improved our prediction result. It’s easier to understand for the potential beneficiary as well. Informing someone that the danger level is high makes better sense than saying the danger level is 3.6 out of 4. 

## Impact

Our system could offer more insights while new residents making relocation decisions and renting decisions, we can help them to choose a comfortable and safe place to live. This matters for many Berkeley international graduate students, since we are very likely to find a career in this area. 
Our model can also help the public to describe the profile of the districts and understand what demographics of districts are prone to certain types of crimes. It can improve their understanding of the local community, and tackle the problem that matters the most. Moreover, the governors and city planners use our model to allocate resources according to needs across neighbourhood, and get help for reducing the gender gap, decreasing the crime rate and improving the welfare and happiness of the local residents.
Additionally, we can inform local residents about when and why crimes are most likely to happen, and provide the safety index for the residents. Such technology can help them to avoid the dangerous times and neighborhood, and reduce the possibility of encountering a crime. If it is integrated to any navigation app, it can tell the user if it is a good idea to stop and rest. 
The system can advise sights for new businesses about location selection as well. This is a commercializable opportunity to provide recommendations for new small business owners. For restaurants and retailing, foot traffic and safety in the region matters. We can help them to find the place with the largest population and the lowest crime rate, and develop it as a consulting business. 

