#Student ID: 200258232
#Name: Rutvik Kolhe

#The following is the code to test assumptions of linear regression model

require(lmtest)
require(car)

states <- as.data.frame(
  state.x77[,c("Murder","Population",
               "Illiteracy", "Income", "Frost")])
dim(states)
t(states[1,])
dtrain <- states[1:25,]
dtest <- states[26:50,]
murderModel <- lm (Murder ~ Population + Illiteracy 
                   + Income + Frost, data=dtrain)
summary (murderModel) 
#################################################################3

#Linearity
crPlots(murderModel)

#Normality
qqPlot(states$Murder)
hist(states$Murder)

#Error/Noise
dwtest(murderModel)

#Homoscedasticity
ncvTest(murderModel)

#Multicollinearity
vif(murderModel)

#Sensitivity to outliers
outlierTest(murderModel)

