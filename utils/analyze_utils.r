library(meta)
# library(rjags)
# library(coda)
library(data.table)
library(ggplot2)
library(readxl)
library(dplyr)
library(ggsci)


# Ref Siegel et al. "A guide to estimating the reference range from a meta-analysis"

# * Frequentist approach

# Function for Frequentist-based Reference Range based on
# REML Takes in study means, study standard deviations, study
# sample sizes (n)
FreqFit = function(means, sds, n) {
    N = length(means)
    means.use = means
    sds.use = sds
    # fit random effects model using REML
    m.reml = metagen(means.use, sds.use/sqrt(n), comb.fixed = FALSE, 
                     comb.random = TRUE, method.tau = "REML", hakn = FALSE, 
                     prediction = TRUE)
    # * Estimate the CI for pooled mean
    lower.random = m.reml$lower.random  # Random effects model
    upper.random = m.reml$upper.random

    # * Estimate the prediction intervals for new study mean
    lower.predict = m.reml$lower.predict  # Prediction Interval
    upper.predict = m.reml$upper.predict

    # pooled variance - assumes study populations drawn
    # independently
    sigma_hat = sqrt(sum((n - 1) * sds.use^2)/sum(n - 1))
    # Estimates from RE model
    tau_hat = m.reml$tau
    mu_hat = m.reml$TE.random
    # Estimate of total variance (within and between studies)
    total.var = sigma_hat^2 + tau_hat^2
    # * Estimates for limits of reference range
    lower_limit = qnorm(0.025, mean = mu_hat, sd = sqrt(total.var))
    upper_limit = qnorm(0.975, mean = mu_hat, sd = sqrt(total.var))
    return(list(lower.random, upper.random, lower.predict, upper.predict, lower_limit, upper_limit))
}



# * Bayesian approach

# (Model Specification)
# cat(readLines("RefRangeRandomModel.txt"), sep = "\n")
# model{
# for (i in 1:length(y)){
#  theta[i] ~ dnorm(mu, 1/tau^2)
#  y[i] ~ dnorm(theta[i], n[i]/(sigma^2))
#  #dist of sigma
#  x[i] ~ dgamma((n[i]-1)/2, 1/(2*sigma^2))
#  }
# #posterior predictive interval
# new ~ dnorm(mu, 1/(sigma^2 + tau^2))
# # priors
# mu ~ dnorm(0, 0.001)
# tau ~ dunif(0,100)
# sigma ~ dunif(0,100)
# beta ~ dnorm(0, 0.001)
# }

# Function for Bayesian Reference Range Takes in study means,
# study standard deviations, study sample sizes (n) Fits
# Bayesian model using packages rjags and coda
BayesFit = function(means, sds, n, n.iter = 50000) {
    N = length(means)
    y = means
    sd = sds
    x = (n - 1) * (sd^2)
    data = list(y = y, x = x, n = n)
    Inits1 = list(.RNG.name = "base::Mersenne-Twister", .RNG.seed = 123)
    Inits2 = list(.RNG.name = "base::Mersenne-Twister", .RNG.seed = 124)
    Inits = list(Inits1, Inits2)
    jags.model = jags.model(file = "RefRangeRandomModel.txt", 
    data = data, inits = Inits, n.chain = 2, quiet = T)
    burn.in = 5000
    update(jags.model, n.iter = burn.in)
    params = c("tau", "mu", "sigma", "new")
    samps = coda.samples(jags.model, params, n.iter = n.iter, 
    .RNG.name = "base::Mersenne-Twister", .RNG.seed = 123)
    results = list(samps, summary(samps)$quantiles["new", "2.5%"], 
    summary(samps)$quantiles["new", "97.5%"])
    return(results)
}


# * Empirical approach
# Fit empirical method to data takes in study means, sds, and sample sizes (n)
EmpFit = function(means, sds, n) {
    N = length(means)
    means.use = means
    sds.use = sds
    # fit random effects model using REML
    m.reml = metagen(means.use, sds.use/sqrt(n), comb.fixed = FALSE, 
                     comb.random = TRUE, method.tau = "REML", hakn = FALSE, 
                     prediction = TRUE)
    # * Estimate the CI for pooled mean
    lower.random = m.reml$lower.random
    upper.random = m.reml$upper.random

    # * Estimate the prediction intervals for new study mean
    lower.predict = m.reml$lower.predict
    upper.predict = m.reml$upper.predict

    sigma_hat = sqrt(sum((n - 1) * sds^2)/sum(n - 1))
    mu_hat = sum(n * means)/sum(n)
    var.e.w = sum((n - 1) * (means - mu_hat)^2)/sum(n - 1)
    total.var = var.e.w + sigma_hat^2
    # * Estimates for limits of reference range
    lower_limit = qnorm(0.025, mean = mu_hat, sd = sqrt(total.var))
    upper_limit = qnorm(0.975, mean = mu_hat, sd = sqrt(total.var))

    return(list(lower.random, upper.random, lower.predict, upper.predict, lower_limit, upper_limit))
}


# * Diagnosis1: QQ Plot
QQPlot = function(means) {
    fig = ggplot(data.frame(means = means), aes(sample = means)) +
            stat_qq() +
            stat_qq_line() +
            labs(title = "QQ Plot", x = "Theoretical Quantiles", y = "Sample Quantiles") +
            theme_minimal()
    
    ggsave(filename = "qqplot.png", plot = fig, width = 8, height = 6)
}

# * Diagnosis2: Forest Plot

ForestPlot = function(sds, n, log_transform=FALSE) {
    lower.sd = sqrt(sds^2 * (n - 1) / qchisq(0.025, n - 1, lower.tail = FALSE))
    upper.sd = sqrt(sds^2 * (n - 1) / qchisq(0.025, n - 1, lower.tail = TRUE))

    s = 1:length(sds)  # study index
    sd.dat = as.data.frame(cbind(s, sds, lower.sd, upper.sd))
    names(sd.dat) = c("s", "sds", "lower.sd", "upper.sd")

    pooled.sd = sqrt(sum((n - 1) * sds^2) / sum(n - 1))

    y_label = if (log_transform) {
        "Log of Standard Deviation"
    } else {
        "Standard Deviation"
    }
    fig = ggplot(sd.dat, aes(x = sds, xmin = lower.sd, xmax = upper.sd)) +
            geom_hline(yintercept = pooled.sd, linetype = 2) +
            xlab('Study') +
            ylab(y_label) +
            facet_wrap(~s, strip.position = "left", nrow = 20) +
            geom_errorbar(aes(y = sds, ymin = lower.sd, ymax = upper.sd), width = 1, cex = 1) +
            geom_point(aes(y = sd.dat$sds), size = 2.5) +
            theme(
                plot.title = element_text(size = 16, face = "bold"),
                axis.text.y = element_blank(),
                axis.ticks.y = element_blank(),
                axis.text.x = element_text(face = "bold"),
                axis.title = element_text(size = 12, face = "bold"),
                legend.title = element_blank(),
                strip.text.y.left = element_text(hjust = 0, vjust = 1, angle = 0, face = "bold")
            ) +
            coord_flip() +
            guides(fill = guide_legend(reverse = TRUE)) +
            guides(color = guide_legend(reverse = TRUE))

    ggsave(filename = "forestplot.png", plot = fig, width = 8, height = 6)
}