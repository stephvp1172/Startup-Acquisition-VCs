# Predicting Startup Success for Venture Capitalists
Predicting probability of startup acquisition, and expected returns (in USD). Used an expected value framework to make recommendations on whether or not venture capitalists should invest in a particular startup.

Machine Learning II

10 May 2023

Lindsay Neff, Kathleen McQuiddy, Stephanie Palanca, Prachi Pathak, Lejla Skahic

[Link to E-Presentation](https://drive.google.com/file/d/1070ty8Vw8U3yAMsBp9JZoMiysN76bqjU/view?usp=sharing)

**Introduction**

Startups are companies in the initial stage of operation characterized as having limited funds and resources. Startups have massive growth potential due to their desire to establish themselves in the market with innovative ideas and products that aim to compete with already established competitors. Not only do these qualities make startups good opportunities for investment, particularly to venture capitalists, but they also benefit markets by introducing novelty, motivating businesses to continue offering cutting edge products and services, and creating more job opportunities, making them worthy of focus in business contexts. However, startups can also be risky investments due to the instability startups have as new entrants into the market. Hence, it is important for venture capital firms to be able to identify which startups are likely to be acquired, factors that contribute to their acquisition, and how much they should invest in a given startup to receive a sizable return on investment.

Our specific analytical goal is to predict whether a startup will end up being acquired, and determine which startups are worth investing in by calculating expected benefit based on our analyses.

**Data Collection & Wrangling**

We wanted to work with a dataset that provided sufficient information on factors that we thought could influence the acquisition of a startup. In our search, we wanted to ensure that our data met at least 3 criteria: the dataset was large enough to properly train our models, that it included information about funding, and whether or not the startup was acquired. We chose this startup dataset since it is sizable (has 43 columns and 1111 rows containing categorical and quantitative data) and includes variables related to location, funding, startup type, and acquisition. In order to make our predictions of startup success more robust, we also wanted to consider the effect of economic conditions in our analysis since success or failure of startups is often influenced by them. So, we merged our startup dataset with an economic dataset that gave us information about unemployment rates and CPI to understand labor force behavior and inflationary conditions during a given startup's lifetime.

In preparing our data for analyses, we converted columns with dates (like founding, closing, and funding dates) into a date-time format, filled in missing data with '0' for numerical columns and 'N/A' for categorical columns, and converted incorrectly encoded columns into their appropriate datatype. We then converted data in categorical columns into numerical data using encoding and created dummy variables so we could perform analyses with the data in these columns. We also scaled values that had a wide range of values (specifically columns with monetary data). Finally, we chose which features we wanted to retain for our analyses- we dropped several columns that would not provide meaningful insights into predicting startup success (like Organization Name, and SEMrush data columns), and redundant columns that included information that was already captured by other columns.

**Exploratory Data Analysis**

We began our EDA by looking at descriptive statistics like means, ranges and quartile information of our data. We found that some variables, like the funding/price variables, estimated revenue, and intellectual property variables had a large range of values, whereas other variables like number of investors and active products had smaller ranges centered around their means. This gave us an insight into how our variables were scaled, and helped us identify that scaling variables could be a potential concern in our dataset, which we addressed with the use of standard scalar.

We then created a correlation matrix to understand how our variables were related to each other. Although there was generally low correlation between variables, there were some moderate correlations. As expected, we found moderate correlations between the different funding variables, and between patents and trademarks. Interestingly, startups with more trademarks received more equity funding, but the same pattern did not follow for startups with more patents. We also found that funding increases as the number of investors increases. Finally, we found a moderate correlation between our two target variables of interest: total funding and selling price.

In our EDA we also found that startups were most often located in the states of California, New York and Massachusetts, and were most often in tech industries. Specifically, most startups focused on software, web and mobile related endeavors. The 'Mobile' category particularly stood out compared to other categories when it came to funding. Of all the startups in this dataset, 676 went on to become acquired.

**Factor Selection**

After preparing the dataset for analyses, we retained columns that included information on funding data (funding types and amounts, including equity funding), intellectual property information (patents and trademarks), date-time information on founding and closing, investors and investments, several descriptors providing information on the startups (state, apps, employees, etc.) and economic indicators (unemployment rate and CPI). We decided to measure success in terms of 'Total Funding Amount' and defined success as being acquired.

**Predictive Modeling**

We were interested in making three different predictions using our models for any given startup. First, whether or not a startup was successful (which we defined as being acquired for our problem definition), second, to predict total funding for a startup, and third, to predict the selling price of an acquired startup.

We built four classifier models– Logistic Regression, SVM, KNN, and Random Forest, and used 'Acquisition' as the target variable to predict the probability of success for each startup.

We also built two linear regression models. In the first linear regression model used to predict total funding we considered all startups and 'Total Funding' as the target variable. In the second linear regression used to predict selling price of an acquired startup, we subset the data to include acquired startups that had a selling price of greater than 0 only, and used 'Selling Price' as the target variable. We then evaluated the models using LASSO, ridge regression, and elastic net.

**Model Performance Evaluation & Selection**

We used a 70-30 training and test split and reserved 30% of our data to see how well our models performed on a new dataset. In order to evaluate each model, we used the accuracy score and mean square error (MSE) for classifiers and regressors (respectively) to assess how well each model was predicting startup success. We also considered graphs and compared the values we obtained for actual and predicted to assess overfitting/underfitting in our models.

We obtained the following accuracy scores for our classifiers:

| **Classifier** | **Accuracy Score** |
| --- | --- |
| Logistic Regression | 0.713 |
| SVM | 0.716 |
| KNN | 0.719 |
| **Random Forest** | **0.732** |

Of our classifiers, Random Forest had the highest accuracy score making it the model with the best predictive power.

To evaluate our classification models we used the eli5 package to compare and contrast the global feature weights and how the model uses them for predicting if a startup would be acquired. For the SVM model it sums up the contribution and if it's a positive number it will predict the company will be acquired. For Random Forest it runs it on a probability instead.

For SVM, the feature weights range from positive to negative with lucrative industries like biotech, hardware, and semiconductors applying the most weight to decisions. For negative weight, large sized companies were penalized in the SVM as well as a general BIAS that is the expected average score or intercept when all other variables are 0. The Random Forest feature weights also show the ranges but applies a heavier weight to number of investors, founded year, and total equity funding amount.

When looking at both of these we can see Random Forest applies more weight to quantitative variables while the SVM applies more weight to our categorical dummy variables.

| **SVM** | **Random Forest** |
| --- | --- |
| ![](RackMultipart20230817-1-qcids1_html_30631ab49364d875.png) | ![](RackMultipart20230817-1-qcids1_html_20683da4cce32cc0.png) |

Below we have an example of a specific accurate prediction and exactly how the acquisition status for a company was predicted for each model. This specific start up was founded in 2008 and did not have an estimated revenue range in the data but we did know the total equity funding amount and funding rounds, number of products, selling price, and that it was outside the top 10 industries.

The SVM calculated acquisition with the highest weights going to founded year and month, about 6 months before the end of the 2000s financial crisis. Here we see the combination of BIAS, or the intercept as we're used to calling it, with the year's unemployment rate, CPI, and number of funding rounds would have pulled the prediction to not acquire this company.

With the Random Forest our BIAS pulls the most weight along with founded year. It's interesting to see how each model weighs the BIAS in different directions, but each still comes to an overall conclusion to acquire.

| **SVM** | **Random Forest** |
| --- | --- |
| ![](RackMultipart20230817-1-qcids1_html_e62151c5630f87ba.png) | ![](RackMultipart20230817-1-qcids1_html_ff5d14b0299d9f7c.png) |

This last example is a specific start up that the models predicted incorrectly. They predicted the company to not be acquired but in reality it was acquired. This company in the hosting industry was founded in 2002 in CA, it had 3 funding rounds and 2 trademarks registered. What caused the SVM model to incorrectly predict this startup is the lack of an estimated revenue range (as we see at the bottom it's listed as NA) and also the 7 year gap between founded year and closed year. On the Random Forest side for this same startup we see a lot of similarities in what was penalized - the lack of an estimated revenue range and the founded and closed years. With cleaner, more accurate data and fewer gaps we would most likely see this predicted accurately if re-run.

| **SVM** | **Random Forest** |
| --- | --- |
| ![](RackMultipart20230817-1-qcids1_html_44803ab493858c0c.png) | ![](RackMultipart20230817-1-qcids1_html_8ffcfbf9be81efde.png) ![](RackMultipart20230817-1-qcids1_html_833047a7ec5b205c.png)
 |

We obtained the following MSEs for the regression models:

| **Regression MSE (Total Funding)** | **Regression MSE (Selling Price for Acquired)** |
| --- | --- |
| **Linear Regression** | **0.015** | Linear Regression | 2090.785 |
| LASSO | 0.923 | LASSO | 2.875 |
| Ridge Regression | 0.955 | **Ridge Regression** | **2.406** |
| Elastic Net | 0.948 | Elastic Net | 2.704 |

Of the regression models for total funding, the linear regression had the lowest MSE, and so was used to obtain the predicted funding for the expected benefit calculations.

Of the regression models for selling price for acquired startups, the ridge regression had the lowest MSE, and so was used to obtain the predicted selling price for the expected benefit calculations.

**Expected Value Framework**

When conducting our model building we recognized the importance of predicting both the probability of acquisition as well as the expected value of an investment in a particular startup. We utilized the expected value framework in order to calculate the expected benefit for investors based on the probability of a startup's acquisition, the startup's predicted selling price as well as a startup's predicted total funding. We generated the following equation to calculate the expected value of an investment:

| **Expected Value Framework** |
| --- |
| _**Expected Benefit = P(A)\*[Predicted Price - Predicted Funding] + [1-P(A)]\*[-Predicted Funding]**_Where P(A) = Probability of Acquisition (From Classifier)Predicted Price (From Regressor)Predicted Funding (From Regressor) |

First, we utilized our selected random forest classifier to obtain the predicted values and probabilities of acquisition for all of the rows of data. This gave us insight into the predicted outcomes of the companies we had information on. Next, we used our two regressors to obtain the predicted prices and the predicted total funding amounts for each of the startups. The first regressor we created was to predict the total funding for a particular startup. We calculated the predicted values for all rows based on the linear regressor we trained/tested on the majority of the data. The second regressor we created was to predict selling price (price currency) for just those companies that were acquired. We omitted those not acquired as their selling price would be 0 and therefore there would not be an expected benefit from investing. However, we encountered challenges with predicted prices as several values in our price column were 0 (missing as we encoded numerical NAs as 0). To combat this, we trained the price regressor on just the subset of those companies acquired AND those companies that had values for price, so we had no NA in our target variable. We were then able to predict the selling price for about 111 companies. Our second regressor, which predicted selling price, had a high mean square error, so we used ridge regression to prevent overfitting and improve model performance.

On this subset of about 100 companies which were acquired and had price, we plugged the column values into our expected benefit equation, to produce a new column that gave a value for expected benefit. Then, we created a new column called investment decision that states invest if the expected benefit is greater than 0 and do not invest if equal to or less than 0. While ideally we could predict expected value for all of the rows of data, since we were missing price information we were only able to utilize the framework for a small subset of the data.

**Recommendations and Considerations**

Using the models we built, we were able to identify variables that were likely to predict startup success and predict the total funding and selling price for startups. We then used these models to create an expected value framework. From our analyses of a subset of the original data we were able to derive investment decisions of 'Invest' or 'Do not invest' based on the expected benefit we calculated. VCs can use our models to determine which startups to invest in and use the expected value framework to determine profitability of their investment.

Despite finding that most suggested investments were in CA, and of the defined industries 'Web' was most recommended for investing, the limited data set we had for pricing indicates that a more complex and comprehensive model is necessary to completely understand the expected benefit of investing in a particular startup. We also need more data on the price of acquisition; because we filled in missing values with 0, we subset the data to only those acquired startups that had a selling price of greater than 0. This meant that we were not able to fully capture the relationship between price and acquisition in our model. Additionally, in this business case, we focused on recouping initial investment based on whether or not the startup was acquired, but this is a somewhat reductive approach because it does not consider the possibility of long-term success without acquisition, which could be a profitable investment. In order to make our predictions more robust, we would need to include a border scope of how we define success, specifically one that goes beyond selling price during acquisition. Despite these limitations, our findings can still provide VCs with information about how startups progress and succeed within the parameters we defined for this problem, and provide insight into how to predict where and when investments should be made for a good return on investment.

**Limitations and Further Exploration**

The aforementioned considerations in our recommendations are in part because of limitations of the dataset. In addition to missing data for price, the dataset had other limitations. For monetary features like the estimated revenue, we had ranges rather than more granular estimates that limited our ability to make more precise predictions. We also faced some issues with incomparable scale ranges for different variables. While we did attempt to correct this by using standard scalar, this did pose certain limitations when building our models. Despite training several models, and using cross-validation and hyperparameter tuning techniques, our best classifiers only had about a 70% accuracy. While there are limitations to how well models perform without risking the possibility of overfitting, a dataset with no (or fewer) missing values would have increased model accuracy as we would have a larger and complete dataset to train our models on. This is especially true for the expected value framework- ideally we could build a model that predicts price and value for all rows, but missing data prevented this. If we had access to a larger data set then we could train better models to predict price/funding and the probability of acquisition. This data was also limited to startups in the United States. While there are benefits to focusing on making predictions for certain regions, investments need not be limited to these, especially when it's possible that there may be opportunities to invest in other geographical locations and the same variables could perform differently in datasets that capture nuances of startup success in other regions. As our world becomes increasingly globalized, it may be worthwhile to consider broadening our datasets to include information from other markets and economies so our ability to predict investment potential has more far reaching implications.

**Appendix**

**Data Dictionary - Startup Data**

| **Column Name** | **Description** | **Data Type** |
| --- | --- | --- |
| **Organization Name** | Name of the company | Object |
| **Industries** | A list of the industry categories that the startup falls into | Object |
| **Headquarters Location** | Where the company headquarters are located | Object
 |
| **Founded Date** | The date which the startup was reportedly founded | Object |
| **Operating Status** | Whether the startup is operating or closed | Object |
| **Closed Date** | The date in which the startup was officially closed (if applicable) | Object |
| **Number of Articles** | The number of articles that have been written about the startup | Object |
| **Investor Type** | Describes the type of investors the organization is (Angel Investor, etc.) | Object |
| **Number of Portfolio Organizations** | Total number of portfolio organizations that have invested in the startup | Float64 |
| **Number of Investments** | Total number of investments made in the startup | Float64 |
| **Number of Lead Investments** | Total number of lead investors the startup has | Float64 |
| **Number of Exits** | Total number of investor exists | Float64 |
| **Number of Exits (IPO)** | Total number of investor exits post IPO | Float64 |
| **Industry Groups** | Subset of Industry Groups | Object |
| **Number of Founders** | Number of people who founded the startup | Float64 |
| **Number of Employees** | Range of the number of employees (1-10, 11-50, etc.) | Object |
| **Number of Funding Rounds** | Total Number of Funding Rounds the startup has gone through | Float64 |
| **Funding Status** | Describes the company's most recent funding status (Early Stage Venture, Late Stage Venture, etc.) | Object |
| **Last Funding Date** | Date of most recent funding round | Object |
| **Last Funding Amount Currency (in USD)** | Amount of most recent funding round (in USD) | Float64
 |
| **Last Funding Type** | Last funding round type (Seed, Private Equity, etc.) | Object |
| **Last Equity Funding Amount Currency (in USD)** | The amount of most recent funding round excluding debt (in USD) | Float64 |
| **Last Equity Funding Type** | The most recent funding round excluding debt | Object |
| **Total Equity Funding Amount Currency (in USD)** | Total amount of funding raised from all funding rounds excluding debt (in USD) | Float64 |
| **Total Funding Amount Currency (in USD)** | Total amount of funding raised (in USD) | Float64 |
| **Number of Lead Investors** | Total number of lead investor firms and individual investors | Float64 |
| **Number of Investors** | Total number of individual investors and investment firms | Float64 |
| **Number of Acquisitions** | Total number of acquisitions | Float64 |
| **Acquisition Status** | The acquisition status of the organization (Acquired, Was acquired, made acquisitions, etc.) | Object |
| **Acquired by** | Name of the organization that acquired the startup | Object |
| **Price Currency (in USD)** | Price the startup was acquired for | Float64 |
| **Acquisition Type** | Type of acquisition (Merger, Acquisition, etc.) | Object |
| **IPO Status** | Current Public Status of the startup | Object |
| **IPO Date** | Date that the startup went public | Object |
| **Money raised at IPO Currency (in USD)** | Amount the startup raised at IPO | Float64 |
| **Valuation at IPO** | Valuation of the startup at IPO | Float64 |
| **Valuation at IPO Currency** | Valuation of the startup at IPO in USD | Float64 |
| **Number of Events** | Total number of events that a startup appeared in/participated in | Float64 |
| **Number of Contacts** | Total number of company contacts available on CrunchBase | Object |
| **SEMrush - Monthly Visits** | Total non-unique visits to the startup's site in the last month (Desktop and Mobile Web) | Object |
| **SEMrush - Average Visits (6 Months)** | Average number of visits over the past 6 months | Object |
| **Active Tech Count** | The total active technology that a company has ranging from 0 to 224 | Float64 |
| **Number of Apps** | The number of apps that a startup has ranging from 0 to 229 | Float64 |
| **Number of Products Active** | The total active products that a startup has ranging from 0 to 125 | Float64 |
| **Patents Granted** | The number of patents that have been granted to a startup, ranging from 0 to 584 | Float64 |
| **Trademarks Registered** | The number of trademarks that a startup has registered, ranging from 0 to 97 | Float64 |
| **IT Spend Currency (in USD)** | Approximate Amount that the company spends on IT per year | Float64 |
| **Estimated Revenue Range** | A range of estimated revenue. In groups, for example, less than $1M, $100M - $500M to $500M - $1B | Object
 |
| **CPI** | The Consumer Price Index for All Urban Consumers: All Items in U.S. City Average, Percent Change from Year Ago, Monthly, Seasonally Adjusted. In terms of months | Float64 |
| **Unemployment** | (Seas) Unemployment Rate from the Current Population Survey, age 16 and older. Monthly average. | Float64 |
| **state\_code** | The state of the company's headquarters | Object |
| **category\_code** | The main industry that the company belongs to | Object |

**Exploratory Data Analysis**

**Correlation Matrix**

![](RackMultipart20230817-1-qcids1_html_b26506b5a1cb86ed.png)

**Industry Categories**

![](RackMultipart20230817-1-qcids1_html_11b55163a9c82859.png)

**Startup Locations**

![](RackMultipart20230817-1-qcids1_html_6a28314a2f0b0fc5.png)

**Estimated Revenue Range**

![](RackMultipart20230817-1-qcids1_html_7f7e6f27ade00068.png)

**Total Funding Amount by Category Code & Operating Status**

![](RackMultipart20230817-1-qcids1_html_b601e0dd87df07e4.png)

**Total Funding Amount by Category Code & Funding Status**

![](RackMultipart20230817-1-qcids1_html_2ad971ded4f43c7.png)

**Unemployment Rate Over Time**

![](RackMultipart20230817-1-qcids1_html_c50e0ccaccf6529e.png)

**CPI Over Time**

![](RackMultipart20230817-1-qcids1_html_1306c0bcc5b66778.png)

**Classification Model Evaluation: Accuracy Scores**

| **Classifier** | **Accuracy Score** |
| --- | --- |
| Logistic Regression | 0.713 |
| SVM | 0.716 |
| KNN | 0.719 |
| **Random Forest** | **0.732** |

**Regression Model Evaluation**

| **Regression MSE (Total Funding)** | **Regression MSE (Selling Price for Acquired)** |
| --- | --- |
| **Linear Regression** | **0.015** | Linear Regression | 2090.785 |
| LASSO | 0.923 | LASSO | 2.875 |
| Ridge Regression | 0.955 | **Ridge Regression** | **2.406** |
| Elastic Net | 0.948 | Elastic Net | 2.704 |

**Expected Benefit Results**

\*Of those startups acquired with price information (111 rows)

![](RackMultipart20230817-1-qcids1_html_93ab1dc9fcd40f45.png)

**Exploratory Data Analysis of Recommended Investments**

**Industry Counts**

![](RackMultipart20230817-1-qcids1_html_345dd5f71c8c128a.png)

**State Counts**

![](RackMultipart20230817-1-qcids1_html_92085b50fbb85aa4.png)

**Last Funding Type Counts**

![](RackMultipart20230817-1-qcids1_html_1e3557897c89d1b3.png)

17
