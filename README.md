# COM624 - Machine Learning
Repository for the COM624 Machine Learning Assessment.

All the code for the respository is held within the main.py file:

[Main.py File](https://github.com/kyleroberts27/Machine-Learning/blob/master/main.py)

# Imported Python Libraries
- yfinance
- pandas
- colorama
- numpy
- matplotlib
- sklearn
- seaborn
- operator
- prophet
- pickle
- statsmodels
- pylab
- warnings
- tensorflow
- math
- streamlit/ streamlit_options_menu


# How to Run the Project in PyCharm
This application uses the open source Python Library [Streamlit](https://streamlit.io/) which is used to rapidly build and load a interactive Webapps for Machine Learning or Data Science. Streamlit also allows for data to be loaded in real time, something which I have used on every page apart from LSTM. This is because, when the model runs to load in the data, it takes upwards of 30 seconds to load. 

To get around this, once the user loads the data in for the first time, the displayed data is cached as a resource, as well as a screenshot of a higher epoch for training the Neural Network.

As all of the code for the project is held within the main.py file linked above. All a user needs to do to load and run the application, is input the command `streamlit run main.py` into the local PyCharm terminal as shown below:

![Run the project](https://github.com/kyleroberts27/Machine-Learning/assets/115091926/e6d8db3c-6fd3-429c-9048-3e257959fba5)

Once the user presses enter on the terminal command in PyCharm, the Webapp's Homepage is opened on localhost:8501. As you can see below, this has the 4 stocks I have chosen to do my analysis on.

![Machine Learning Webapp Home Page](https://github.com/kyleroberts27/Machine-Learning/assets/115091926/5e4a2e89-f080-4b8a-9a7b-ff230e7b652b)

The user can then navigate to whatever option they decide within the Nav Bar at the top by clicking on the option they chose, circled below: 

![Machine Learning Webapp Navbar](https://github.com/kyleroberts27/Machine-Learning/assets/115091926/c0d3f37f-5a79-4cc6-b433-ff483005d6dc)

For this demonstration I have navigated to EDA Analysis Page:

![EDA Analysis Page](https://github.com/kyleroberts27/Machine-Learning/assets/115091926/47493c57-87c5-4e42-b06d-a05223390b08)

As you can see, there is a `st.selectbox` drop down, for a user to chose one of the stocks within the list (currently SBUX is selected). This the same for all the pages, apart from 'Home', 'Clusters' and 'Correlation'.


