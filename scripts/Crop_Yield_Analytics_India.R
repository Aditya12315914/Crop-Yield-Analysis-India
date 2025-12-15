# ==============================================================
# 🌾 CROP YIELD ANALYSIS IN INDIA
# ==============================================================
# Objective: 
# Analyze how rainfall, fertilizer, and pesticide usage 
# influence crop yield across different states and seasons.
# ==============================================================

# --- STEP 1: LOAD & CLEAN DATA ---

#setwd("E:/R/Project") 
#crop <- read.csv("crop_yield.csv")
#crop <- read.csv("E:/R/Project/crop_yield_5_years.csv")
#View(crop)
setwd(getwd())
crop <- read.csv("data/crop_yield_5_years.csv")

# Clean and prepare
crop$Season <- trimws(crop$Season)
crop$Season <- as.factor(crop$Season)
crop$State  <- as.factor(crop$State)
crop$Crop   <- as.factor(crop$Crop)

# Basic info
head(crop)
str(crop)
summary(crop)
colSums(is.na(crop))
sum(duplicated(crop))

# ==============================================================

# --- STEP 2: DESCRIPTIVE ANALYTICS ---

mean(crop$Yield)
median(crop$Yield)
sd(crop$Yield)
cor(crop[, c("Yield", "Annual_Rainfall", "Fertilizer", "Pesticide")])
aggregate(Yield ~ Season, data = crop, FUN = mean)

# ==============================================================

# --- STEP 3: VISUALIZATION  ---

# Yield Distribution
hist(log1p(crop$Yield),
     main = "Distribution of Crop Yield (Log Scale)",
     xlab = "log(Yield + 1)", col = "lightblue", border = "black", breaks = 30)

# Rainfall vs Yield
plot(log1p(crop$Annual_Rainfall), log1p(crop$Yield),
     main = "Rainfall vs Yield", xlab = "log(Rainfall + 1)",
     ylab = "log(Yield + 1)", col = "blue", pch = 19)

# Fertilizer vs Yield
plot(log1p(crop$Fertilizer), log1p(crop$Yield),
     main = "Fertilizer vs Yield", xlab = "log(Fertilizer + 1)",
     ylab = "log(Yield + 1)", col = "darkred", pch = 19)

# Pesticide vs Yield
plot(log1p(crop$Pesticide), log1p(crop$Yield),
     main = "Pesticide vs Yield", xlab = "log(Pesticide + 1)",
     ylab = "log(Yield + 1)", col = "purple", pch = 19)

# ==============================================================

# --- STEP 4: LINEAR REGRESSION MODELS ---

# 4.1 National Model
model_all <- lm(Yield ~ Annual_Rainfall + Fertilizer + Pesticide + Season, data = crop)
summary(model_all)
crop$Predicted_All <- predict(model_all)

plot(crop$Yield, crop$Predicted_All, col = "blue", pch = 19,
     main = "Actual vs Predicted Yield (National Model)",
     xlab = "Actual Yield", ylab = "Predicted Yield")
abline(0, 1, col = "red", lwd = 2)

# Sample Prediction
new_data_all <- data.frame(Annual_Rainfall = 1200, Fertilizer = 200000, 
                           Pesticide = 2500, Season = "Kharif")
predict(model_all, newdata = new_data_all)

# --------------------------------------------------------------

# 4.2 State + Crop Model (Rice in Maharashtra)
crop_maha_rice <- subset(crop, Crop == "Rice" & State == "Maharashtra")
model_maha_rice <- lm(Yield ~ Annual_Rainfall + Fertilizer + Pesticide + Season, data = crop_maha_rice)
summary(model_maha_rice)
crop_maha_rice$Predicted_Maha_Rice <- predict(model_maha_rice)

plot(crop_maha_rice$Yield, crop_maha_rice$Predicted_Maha_Rice, 
     main = "Actual vs Predicted Yield (Rice in Maharashtra)",
     xlab = "Actual Yield", ylab = "Predicted Yield",
     col = "orange", pch = 19)
abline(0, 1, col = "red", lwd = 2)

# Comparison
model_compare <- data.frame(
  Model = c("National", "Maharashtra - Rice"),
  Adjusted_R2 = c(summary(model_all)$adj.r.squared, summary(model_maha_rice)$adj.r.squared)
)
print(model_compare)

# ==============================================================

# --- STEP 5: ANOVA (Yield Difference Across Seasons) ---

anova_model <- aov(Yield ~ Season, data = crop)
summary(anova_model)
TukeyHSD(anova_model)

boxplot(log1p(Yield) ~ Season, data = crop,
        main = "Crop Yield by Season (Log Scale)",
        xlab = "Season", ylab = "log(Yield + 1)",
        col = "lightgreen", border = "black")

# ==============================================================

# --- STEP 6: K-MEANS CLUSTERING ---

cluster_data <- na.omit(crop[, c("Annual_Rainfall", "Fertilizer", "Yield")])
cluster_scaled <- scale(log1p(cluster_data))

set.seed(123)
km_model <- kmeans(cluster_scaled, centers = 3, nstart = 25)
crop$Cluster <- km_model$cluster
table(crop$Cluster)
km_model$centers

cluster_colors <- c("tomato", "seagreen3", "royalblue")
plot(log1p(crop$Annual_Rainfall), log1p(crop$Yield),
     col = adjustcolor(cluster_colors[crop$Cluster], alpha.f = 0.6),
     pch = 19, cex = 0.7,
     main = "K-Means Clustering (Log Scaled)",
     xlab = "log(Annual Rainfall + 1)",
     ylab = "log(Yield + 1)")

# ==============================================================

# --- STEP 7: KNN CLASSIFICATION (Yield Category) ---

# --- Load Libraries ---
library(class)
library(ggplot2)
library(caret)   # for confusion matrix
#install.packages("caret")
# --- Prepare Dataset ---
crop_data <- na.omit(crop[, c("Annual_Rainfall", "Fertilizer", "Pesticide", "Yield")])

# Categorize Yield into Low / Medium / High
crop_data$Yield_Category <- cut(crop_data$Yield,
                                breaks = 3,
                                labels = c("Low", "Medium", "High"))

# Normalize numeric features
scaled_features <- as.data.frame(scale(crop_data[, 1:3]))
scaled_features$Yield_Category <- crop_data$Yield_Category

# --- Split into Train/Test (80-20) ---
set.seed(123)
train_idx <- sample(1:nrow(scaled_features), 0.8 * nrow(scaled_features))
train_data <- scaled_features[train_idx, 1:3]
test_data_split  <- scaled_features[-train_idx, 1:3]
train_label <- scaled_features$Yield_Category[train_idx]
test_label  <- scaled_features$Yield_Category[-train_idx]

# --- Train & Predict using KNN ---
set.seed(123)
pred_knn <- knn(train = train_data, test = test_data_split, cl = train_label, k = 5)

# --- Evaluate Accuracy ---
conf_matrix <- table(Predicted = pred_knn, Actual = test_label)
print(conf_matrix)

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("KNN Model Accuracy:", round(accuracy * 100, 2), "%\n")

# --- Predict a New Example ---
new_field <- data.frame(
  Annual_Rainfall = 1200,
  Fertilizer = 200000,
  Pesticide = 2500
)

# Scale new input using same parameters as training data
new_field_scaled <- as.data.frame(scale(new_field,
                                        center = attr(scale(crop_data[, 1:3]), "scaled:center"),
                                        scale  = attr(scale(crop_data[, 1:3]), "scaled:scale")
))

# Predict yield category for new input
set.seed(123)
pred_new <- knn(train = train_data, test = new_field_scaled, cl = train_label, k = 5)
cat("\nPredicted Yield Category for New Field:", as.character(pred_new), "\n")

# --- Visualization: Show Dataset + Predicted Point ---
ggplot(crop_data, aes(x = log1p(Annual_Rainfall), y = log1p(Yield), color = Yield_Category)) +
  geom_point(size = 2.5, alpha = 0.6) +
  geom_point(aes(x = log1p(new_field$Annual_Rainfall), y = log1p(2.5)),  # Predicted point
             color = ifelse(pred_new == "High", "green",
                            ifelse(pred_new == "Medium", "orange", "red")),
             shape = 17, size = 6) +
  geom_text(aes(x = log1p(new_field$Annual_Rainfall), y = log1p(2.5),
                label = paste("Pred:", pred_new)), vjust = -1, size = 5, color = "black") +
  scale_color_manual(values = c("Low" = "red", "Medium" = "orange", "High" = "green")) +
  labs(title = "KNN Yield Prediction by Rainfall and Yield (Log Scale)",
       x = "log(Annual Rainfall + 1)", y = "log(Yield + 1)",
       color = "Yield Category") +
  theme_minimal(base_size = 13)

# ==============================================================

# --- STEP 8: APRIORI ALGORITHM (Association Rules) ---

library(arules)
library(arulesViz)

assoc_data <- na.omit(crop[, c("Annual_Rainfall", "Fertilizer", "Pesticide", "Yield")])
assoc_data$Rainfall   <- cut(assoc_data$Annual_Rainfall, 3, labels = c("LowRain", "MedRain", "HighRain"))
assoc_data$Fertilizer <- cut(assoc_data$Fertilizer, 3, labels = c("LowFert", "MedFert", "HighFert"))
assoc_data$Pesticide  <- cut(assoc_data$Pesticide, 3, labels = c("LowPest", "MedPest", "HighPest"))
assoc_data$Yield      <- cut(assoc_data$Yield, 3, labels = c("LowYield", "MedYield", "HighYield"))

assoc_cat <- assoc_data[, c("Rainfall", "Fertilizer", "Pesticide", "Yield")]
transactions <- as(assoc_cat, "transactions")

rules <- apriori(transactions, parameter = list(support = 0.05, confidence = 0.6))
inspect(sort(rules, by = "lift")[1:10])

# Visuals
itemFrequencyPlot(transactions, topN = 5, col = "lightblue", main = "Top 5 Frequent Items")
plot(rules, method = "graph", engine = "htmlwidget")
plot(rules, method = "grouped")





# ==============================================================
# 🌾 STEP 9: MODEL PERFORMANCE COMPARISON SUMMARY
# ==============================================================

# --- 1️⃣ Linear Regression (National & State Models) ---
R2_National <- summary(model_all)$r.squared
AdjR2_National <- summary(model_all)$adj.r.squared

R2_MahaRice <- summary(model_maha_rice)$r.squared
AdjR2_MahaRice <- summary(model_maha_rice)$adj.r.squared

# --- 2️⃣ K-Means Clustering ---
library(cluster)
sil <- silhouette(km_model$cluster, dist(cluster_scaled))
Silhouette_Score <- round(mean(sil[, 3]), 3)

# --- 3️⃣ KNN Classification ---
# (You already calculated accuracy in Step 7)
KNN_Accuracy <- round(accuracy * 100, 2)  # convert to percentage

# --- 4️⃣ Apriori Association Rules ---
# Take average Lift of top 5 rules as overall indicator
top_rules <- sort(rules, by = "lift")[1:5]
Lift_Avg <- round(mean(quality(top_rules)$lift), 3)

# --- Combine All Results in a Single Table ---
model_summary <- data.frame(
  Model = c(
    "Linear Regression (All Crops)",
    "Linear Regression (Rice in Maharashtra)",
    "K-Means Clustering",
    "KNN Classification",
    "Apriori Association Rules"
  ),
  Metric = c(
    paste("R²:", round(R2_National, 3), "| Adj R²:", round(AdjR2_National, 3)),
    paste("R²:", round(R2_MahaRice, 3), "| Adj R²:", round(AdjR2_MahaRice, 3)),
    paste("Silhouette Score:", Silhouette_Score),
    paste("Accuracy:", KNN_Accuracy, "%"),
    paste("Avg Lift:", Lift_Avg)
  )
)

# --- Print Final Model Comparison Table ---
cat("\n🌾 MODEL PERFORMANCE COMPARISON SUMMARY 🌾\n")
print(model_summary, row.names = FALSE)

# ==============================================================

# 🎯 END OF PROJECT
# ==============================================================































# ==============================================================
# 🌾 CROP YIELD ANALYSIS IN INDIA
# ==============================================================
# Objective: Analyze how rainfall, fertilizer, and pesticide usage 
# influence crop yield across different states and seasons.
# ==============================================================

# --- STEP 1: LOAD & CLEAN DATA ---

setwd("E:/R/Project") 
crop <- read.csv("E:/R/Project/crop_yield_5_years.csv")
View(crop)

# Create graph directory if not exists
if(!dir.exists("E:/R/Project/graphs")) dir.create("E:/R/Project/graphs")

# Clean categorical variables
crop$Season <- as.factor(trimws(crop$Season))
crop$State  <- as.factor(crop$State)
crop$Crop   <- as.factor(crop$Crop)

# Basic info
head(crop); str(crop); summary(crop)
colSums(is.na(crop))
sum(duplicated(crop))

# ==============================================================

# --- STEP 2: DESCRIPTIVE ANALYTICS ---

mean(crop$Yield); median(crop$Yield); sd(crop$Yield)
cor(crop[, c("Yield", "Annual_Rainfall", "Fertilizer", "Pesticide")])
aggregate(Yield ~ Season, data = crop, FUN = mean)

# ==============================================================

# --- STEP 3: VISUALIZATION (BASE R) ---

# 3.1 Yield Distribution
png("E:/R/Project/graphs/Step3_Yield_Distribution.png")
hist(log1p(crop$Yield),
     main = "Distribution of Crop Yield (Log Scale)",
     xlab = "log(Yield + 1)", col = "lightblue", border = "black", breaks = 30)
dev.off()

# 3.2 Rainfall vs Yield
png("E:/R/Project/graphs/Step3_Rainfall_vs_Yield.png")
plot(log1p(crop$Annual_Rainfall), log1p(crop$Yield),
     main = "Rainfall vs Yield", xlab = "log(Rainfall + 1)",
     ylab = "log(Yield + 1)", col = "blue", pch = 19)
dev.off()

# 3.3 Fertilizer vs Yield
png("E:/R/Project/graphs/Step3_Fertilizer_vs_Yield.png")
plot(log1p(crop$Fertilizer), log1p(crop$Yield),
     main = "Fertilizer vs Yield", xlab = "log(Fertilizer + 1)",
     ylab = "log(Yield + 1)", col = "darkred", pch = 19)
dev.off()

# 3.4 Pesticide vs Yield
png("E:/R/Project/graphs/Step3_Pesticide_vs_Yield.png")
plot(log1p(crop$Pesticide), log1p(crop$Yield),
     main = "Pesticide vs Yield", xlab = "log(Pesticide + 1)",
     ylab = "log(Yield + 1)", col = "purple", pch = 19)
dev.off()

# ==============================================================

# --- STEP 4: LINEAR REGRESSION MODELS ---

# 4.1 National Model
model_all <- lm(Yield ~ Annual_Rainfall + Fertilizer + Pesticide + Season, data = crop)
summary(model_all)
crop$Predicted_All <- predict(model_all)

png("E:/R/Project/graphs/Step4_Regression_All.png")
plot(crop$Yield, crop$Predicted_All, col = "blue", pch = 19,
     main = "Actual vs Predicted Yield (National Model)",
     xlab = "Actual Yield", ylab = "Predicted Yield")
abline(0, 1, col = "red", lwd = 2)
dev.off()

# 4.2 State + Crop Model (Rice in Maharashtra)
crop_maha_rice <- subset(crop, Crop == "Rice" & State == "Maharashtra")
model_maha_rice <- lm(Yield ~ Annual_Rainfall + Fertilizer + Pesticide + Season, data = crop_maha_rice)
summary(model_maha_rice)
crop_maha_rice$Predicted_Maha_Rice <- predict(model_maha_rice)

png("E:/R/Project/graphs/Step4_Regression_Maharashtra_Rice.png")
plot(crop_maha_rice$Yield, crop_maha_rice$Predicted_Maha_Rice,
     main = "Actual vs Predicted Yield (Rice in Maharashtra)",
     xlab = "Actual Yield", ylab = "Predicted Yield",
     col = "orange", pch = 19)
abline(0, 1, col = "red", lwd = 2)
dev.off()

# ==============================================================

# --- STEP 5: ANOVA (Yield Difference Across Seasons) ---

anova_model <- aov(Yield ~ Season, data = crop)
summary(anova_model)
TukeyHSD(anova_model)

png("E:/R/Project/graphs/Step5_ANOVA_Yield_by_Season.png")
boxplot(log1p(Yield) ~ Season, data = crop,
        main = "Crop Yield by Season (Log Scale)",
        xlab = "Season", ylab = "log(Yield + 1)",
        col = "lightgreen", border = "black")
dev.off()

# ==============================================================
#windows(width=6,height=5)
# --- STEP 6: K-MEANS CLUSTERING ---

library(cluster)
library(factoextra)

# Prepare and clean data
cluster_data <- na.omit(crop[, c("Annual_Rainfall", "Fertilizer", "Yield")])

# Log transform + scale data
cluster_scaled <- scale(log1p(cluster_data))

# Apply K-Means clustering with 10 clusters
set.seed(123)
km_model <- kmeans(cluster_scaled, centers = 6, nstart = 25)

# Add cluster assignment
crop$Cluster <- km_model$cluster

# Summary of cluster distribution
print(table(crop$Cluster))

# Optional: Check cluster centers (to interpret what each cluster represents)
print(km_model$centers)

# Visualize with clean polygons
fviz_cluster(
  km_model,
  data = cluster_scaled,
  palette = "jco",               # automatically handles many clusters (10 colors)
  ellipse.type = "convex",       # polygon boundaries
  repel = TRUE,                  # avoid overlapping text
  show.clust.cent = TRUE,        # show cluster centers
  labelsize = 10,                # larger labels for cluster numbers
  pointsize = 1.2,               # smaller points to reduce clutter
  geom = "point",                # only points (no text clutter)
  ggtheme = theme_minimal(),
  main = "K-Means Clustering of Crop Data (6 Clusters)"
)
# ==============================================================

# --- STEP 7: KNN CLASSIFICATION (Yield Category) ---

library(class)
library(caret)
library(ggplot2)

crop_data <- na.omit(crop[, c("Annual_Rainfall", "Fertilizer", "Pesticide", "Yield")])
crop_data$Yield_Category <- cut(crop_data$Yield, breaks = 3, labels = c("Low", "Medium", "High"))
scaled_features <- as.data.frame(scale(crop_data[, 1:3]))
scaled_features$Yield_Category <- crop_data$Yield_Category

set.seed(123)
train_idx <- sample(1:nrow(scaled_features), 0.8 * nrow(scaled_features))
train_data <- scaled_features[train_idx, 1:3]
test_data_split  <- scaled_features[-train_idx, 1:3]
train_label <- scaled_features$Yield_Category[train_idx]
test_label  <- scaled_features$Yield_Category[-train_idx]

# Train and Predict
set.seed(123)
pred_knn <- knn(train = train_data, test = test_data_split, cl = train_label, k = 5)

# Confusion Matrix
conf_matrix <- table(Predicted = pred_knn, Actual = test_label)
accuracy <- round(sum(diag(conf_matrix)) / sum(conf_matrix) * 100, 2)
print(conf_matrix)
cat("KNN Model Accuracy:", accuracy, "%\n")

# Visualization
png("E:/R/Project/graphs/Step7_KNN_Classification.png")
ggplot(crop_data, aes(x = log1p(Annual_Rainfall), y = log1p(Yield), color = Yield_Category)) +
  geom_point(size = 2.5, alpha = 0.6) +
  labs(title = "KNN Yield Classification (Log Scale)",
       x = "log(Annual Rainfall + 1)", y = "log(Yield + 1)",
       color = "Yield Category") +
  scale_color_manual(values = c("Low" = "red", "Medium" = "orange", "High" = "green")) +
  theme_minimal(base_size = 13)
dev.off()

# ==============================================================

# --- STEP 8: APRIORI ALGORITHM (Association Rules) ---

library(arules)
library(arulesViz)

assoc_data <- na.omit(crop[, c("Annual_Rainfall", "Fertilizer", "Pesticide", "Yield")])
assoc_data$Rainfall   <- cut(assoc_data$Annual_Rainfall, 3, labels = c("LowRain", "MedRain", "HighRain"))
assoc_data$Fertilizer <- cut(assoc_data$Fertilizer, 3, labels = c("LowFert", "MedFert", "HighFert"))
assoc_data$Pesticide  <- cut(assoc_data$Pesticide, 3, labels = c("LowPest", "MedPest", "HighPest"))
assoc_data$Yield      <- cut(assoc_data$Yield, 3, labels = c("LowYield", "MedYield", "HighYield"))

assoc_cat <- assoc_data[, c("Rainfall", "Fertilizer", "Pesticide", "Yield")]
transactions <- as(assoc_cat, "transactions")
rules <- apriori(transactions, parameter = list(support = 0.05, confidence = 0.6))
inspect(sort(rules, by = "lift")[1:10])

# Save Graphs
png("E:/R/Project/graphs/Step8_Apriori_ItemFrequency.png")
itemFrequencyPlot(transactions, topN = 5, col = "lightblue", main = "Top 5 Frequent Items")
dev.off()

png("E:/R/Project/graphs/Step8_Apriori_Grouped.png")
plot(rules, method = "grouped", control = list(k = 5))
dev.off()

# ==============================================================

# --- STEP 9: MODEL PERFORMANCE SUMMARY ---

top_rules <- sort(rules, by = "lift")[1:5]
Lift_Avg <- round(mean(quality(top_rules)$lift), 3)

model_summary <- data.frame(
  Model = c(
    "Linear Regression (All Crops)",
    "Linear Regression (Rice in Maharashtra)",
    "K-Means Clustering",
    "KNN Classification",
    "Apriori Association Rules"
  ),
  Metric = c(
    paste("R²:", round(summary(model_all)$r.squared, 3),
          "| Adj R²:", round(summary(model_all)$adj.r.squared, 3)),
    paste("R²:", round(summary(model_maha_rice)$r.squared, 3),
          "| Adj R²:", round(summary(model_maha_rice)$adj.r.squared, 3)),
    paste("Silhouette Score:", Silhouette_Score),
    paste("Accuracy:", accuracy, "%"),
    paste("Avg Lift:", Lift_Avg)
  )
)

cat("\n🌾 MODEL PERFORMANCE COMPARISON SUMMARY 🌾\n")
print(model_summary, row.names = FALSE)

# ==============================================================


