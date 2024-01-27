# ======================================================================
# ======================================================================
# Idea: View which variables are important in the CCP dataset
# Creator: Marcos Paulo
# ======================================================================
# ======================================================================

# Load libraries
library(haven)

# Read dataset
directory <- 'C:/Users/marcola/OneDrive/Área de Trabalho/ccpcnc/'
data <- read_dta('C:/Users/marcola/OneDrive/Área de Trabalho/ccpcnc/ccpcnc_v3_small.dta')
