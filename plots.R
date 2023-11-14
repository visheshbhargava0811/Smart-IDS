data = read.csv(file = "results.txt", header = TRUE)
data$acc = as.numeric(data$acc)

acc <- ggplot(data = data, aes(x = algo, y = acc)) + 
  geom_bar(stat = "identity", fill = "steelblue") +
  labs( x = "Algorthim", y = "Accuracy")

acc + ylim(0.2,1)
