library(tidyverse)
#create parameter strings
weekdays <- c("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
#create output names
output_file <- paste0(weekdays,"Analysis", ".md")
#Create a list for each weekdays with just weekday parameter
params <- lapply(weekdays, FUN = function(x){list(weekday=x)})
#put into a data frame
reports <- tibble(output_file, params)
#required library
library(rmarkdown)
#Use apply to knit
apply(reports, MARGIN=1, 
      FUN=function(x){
        render(input = "Project2.Rmd", output_file = x[[1]], params = x[[2]])
        })
