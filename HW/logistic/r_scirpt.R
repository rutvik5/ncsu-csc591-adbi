library(readxl)
library(dummies)

#read the excel file into a dataframe
eBayAuctions<- read_xls("eBayAuctions.xls")

#cleaning data
colnames(eBayAuctions)[8] <- "Competitive"
eBayAuctions$Duration <- as.factor(eBayAuctions$Duration)

#generating dummy variables
dummy_category<- cbind(eBayAuctions, dummy(eBayAuctions$Category, sep = "_"))
dummy_currency<- cbind(eBayAuctions, dummy(eBayAuctions$currency, sep = "_"))
dummy_duration<- cbind(eBayAuctions, dummy(eBayAuctions$Duration, sep = "_"))
dummy_endDay<- cbind(eBayAuctions, dummy(eBayAuctions$endDay, sep = "_"))

#reduced the dummy variables that are similar to each other
shrinked_data = data.frame(eBayAuctions$Competitive ,eBayAuctions$sellerRating, eBayAuctions$OpenPrice, eBayAuctions$ClosePrice, dummy_category$eBayAuctions_Automotive, dummy_category$eBayAuctions_Books, dummy_category$`eBayAuctions_Business/Industrial`, dummy_category$`eBayAuctions_Music/Movie/Game`, dummy_category$eBayAuctions_EverythingElse, dummy_category$eBayAuctions_Photography, dummy_currency$eBayAuctions_GBP, dummy_currency$eBayAuctions_US, dummy_duration$eBayAuctions_10, dummy_duration$eBayAuctions_3, dummy_duration$eBayAuctions_5, dummy_endDay$eBayAuctions_Sat, dummy_endDay$eBayAuctions_Thu, dummy_endDay$eBayAuctions_Sun, dummy_endDay$eBayAuctions_Mon)

######Splitting into train and test data

#randomly picks 60% of the rows from the dataset
splt<- sort(sample(nrow(shrinked_data),nrow(shrinked_data)*0.6))

#allocates 60% of the data for training and the remaining 40% for testing
train<- shrinked_data[splt,]
test<- shrinked_data[-splt,]

#############

#calculate fit.all glm model on training data
fit.all <- glm(train$eBayAuctions.Competitive ~., data = train, family = binomial(link="logit"))
summary(fit.all)
max(abs(fit.all$coefficients))

######

#calculate fit.single glm model for category$EverythingElse
fit.single<- glm(train$eBayAuctions.Competitive~ train$dummy_category.eBayAuctions_EverythingElse, family = binomial(link='logit'))
summary(fit.single)

#####
#sort the coefficients of fit.all to find the top 4 predictors
sort(abs(fit.all$coefficients), decreasing=TRUE)

#build the reduced model
fit.reduced<- glm(train$eBayAuctions.Competitive~ train$eBayAuctions.OpenPrice+train$eBayAuctions.ClosePrice+train$dummy_category.eBayAuctions_Automotive+train$dummy_category.eBayAuctions_EverythingElse+train$dummy_currency.eBayAuctions_GBP+train$dummy_currency.eBayAuctions_US+train$dummy_duration.eBayAuctions_5+train$dummy_endDay.eBayAuctions_Mon)
summary(fit.reduced)

#run anova test to compare two models- fit.all and fit.reduced
aov<- anova(fit.reduced,fit.all ,test='Chisq')

#Dispersion test on data
residual_df<- aov[2,1]
residual_deviance<- aov[2,2]
dispersion = residual_deviance / residual_df
print(dispersion)

