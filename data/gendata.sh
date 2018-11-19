# based on https://github.com/szilard/benchm-ml/blob/master/0-init/2-gendata.txt

for yr in 2005 2006 2007; do
  wget http://stat-computing.org/dataexpo/2009/$yr.csv.bz2
  bunzip2 $yr.csv.bz2
done

time R --vanilla --quiet << EOF

library(data.table)
set.seed(123)

d1a <- fread("2005.csv")
d1b <- fread("2006.csv")
d2 <- fread("2007.csv")
d1 <- rbind(d1a, d1b)

d1 <- d1[!is.na(DepDelay)]
d2 <- d2[!is.na(DepDelay)]

for (k in c("Month","DayofMonth","DayOfWeek")) {
  d1[[k]] <- paste0("c-",as.character(d1[[k]]))
  d2[[k]] <- paste0("c-",as.character(d2[[k]]))
}

d1[["dep_delayed_15min"]] <- ifelse(d1[["DepDelay"]]>=15,"Y","N") 
d2[["dep_delayed_15min"]] <- ifelse(d2[["DepDelay"]]>=15,"Y","N") 

cols <- c("Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", 
            "Origin", "Dest", "Distance","dep_delayed_15min")
d1 <- d1[, cols, with = FALSE]
d2 <- d2[, cols, with = FALSE]

for (n in c(1e4,1e5,1e6,1e7)) {
  write.table(d1[sample(nrow(d1),n),], file = paste0("train-",n/1e6,"m.csv"), row.names = FALSE, sep = ",")
}
idx_test <- sample(nrow(d2),1e5)
idx_valid <- sample(setdiff(1:nrow(d2),idx_test),1e5)
write.table(d2[idx_test,], file = "test.csv", row.names = FALSE, sep = ",")
write.table(d2[idx_valid,], file = "valid.csv", row.names = FALSE, sep = ",")

EOF