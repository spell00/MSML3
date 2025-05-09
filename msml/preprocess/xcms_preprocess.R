# https://bioconductor.org/packages/release/bioc/vignettes/xcms/inst/doc/xcms.html
start.time <- Sys.time()

library(xcms)
library(faahKO)
library(RColorBrewer)
library(pander)
library(pheatmap)
library(MsExperiment)
library(SummarizedExperiment)

## Get the full path to the CDF files
# cdfs <- dir(system.file("mzml", package = "faahKO"), full.names = TRUE,
#             recursive = TRUE)[c(1, 2, 5, 6, 7, 8, 11, 12)]
# Get all directory names in the current directory, except matrices
all_dirs <- list.dirs('resources/bacteries_2024', full.names = FALSE, recursive = FALSE)
# Filter out the directory you want to exclude
all_dirs <- all_dirs[!all_dirs %in% "matrices"]
all_dirs <- c('01-03-2024', '13-03-2024', '02-02-2024', '21-02-2024')

# Create empty list to store mzmls
mzmls <- list()
for (var in all_dirs) {
    print(var)
    mzmls <- do.call(c, list(mzmls, dir(paste0('resources/bacteries_2024/',var,'/mzml'), full.names = TRUE, recursive = TRUE)))
}
mzmls <- unlist(mzmls)

# splits mzmls on / and keep last. 
mzmls_names <- sapply(strsplit(mzmls, "/"), function(x) x[length(x)])

# Get batch names
batches <- sapply(strsplit(mzmls, "/"), function(x) x[length(x)-1])
# split mzmls names and keep second item
mzmls_names <- sapply(strsplit(mzmls_names, "_"), function(x) x[2])

# Concat batch and mzmls names
mzmls_names <- paste(batches, mzmls_names, sep = "_")

## Create a phenodata data.frame
pd <- data.frame(sample_name = sub(basename(mzmls), pattern = ".mzml",
                                   replacement = "", fixed = TRUE),
                 sample_group = mzmls_names,
                 stringsAsFactors = FALSE)

faahko <- readMsExperiment(spectraFiles = mzmls, sampleData = pd)
# faahko


# spectra(faahko)


## Get the base peak chromatograms. This reads data from the files.
bpis <- chromatogram(faahko, aggregationFun = "max")
# ## Define colors for the two groups
# group_colors <- paste0(brewer.pal(3, "Set1")[1:2], "60")
# names(group_colors) <- c("KO", "WT")
# 
# ## Plot all chromatograms.
# plot(bpis, col = group_colors[sampleData(faahko)$sample_group])

tc <- spectra(faahko) |>
    tic() |>
    split(f = fromFile(faahko))
# boxplot(tc, col = group_colors[sampleData(faahko)$sample_group],
#         ylab = "intensity", main = "Total ion current")

## Bin the BPC
bpis_bin <- bin(bpis, binSize = 2)

## Calculate correlation on the log2 transformed base peak intensities
cormat <- cor(log2(do.call(cbind, lapply(bpis_bin, intensity))))
colnames(cormat) <- rownames(cormat) <- bpis_bin$sample_name

## Define which phenodata columns should be highlighted in the plot
ann <- data.frame(group = bpis_bin$sample_group)
rownames(ann) <- bpis_bin$sample_name

# 2.3 Chromatic peak detection
faahko <- findChromPeaks(faahko, param = CentWaveParam(snthresh = 2))

mpp <- MergeNeighboringPeaksParam(expandRt = 4)
faahko <- refineChromPeaks(faahko, mpp)

# 2.4 Alignment
faahko <- adjustRtime(faahko, param = ObiwarpParam(binSize = 0.6))

# 2.5 Correspondance
## Perform the correspondence
pdp <- PeakDensityParam(sampleGroups = sampleData(faahko)$sample_group,
                        minFraction = 0.4, bw = 30)
faahko <- groupChromPeaks(faahko, param = pdp)

# 2.6 Gap filling
faahko <- fillChromPeaks(faahko, param = ChromPeakAreaParam())

res <- quantify(faahko, value = "into", method = "sum")

matrix <- assay(res)

# save the matrix
# paste all elements of list
fname <- paste0("resources/bacteries_2024/matrices/matrix_", paste(all_dirs, collapse='_'), ".csv")
write.csv(matrix, fname)

end.time <- Sys.time()
time.taken <- end.time - start.time
print(time.taken)
