required_packages <- c(
  "ggplot2",
  "caret",
  "class",
  "cluster",
  "factoextra",
  "arules",
  "arulesViz"
)

install.packages(setdiff(required_packages, rownames(installed.packages())))
