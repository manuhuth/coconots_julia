using Pkg

#Pkg.add("ProfileView")

using DelimitedFiles
using Random
using Base.Threads
using BSplines
using BenchmarkTools
using XLSX
using DataFrames
using QuadGK
using Plots
using StaticArrays
using CategoricalArrays
using Distributions
using FastGaussQuadrature
using Profile
using LinearAlgebra
using Base.Threads
using ForwardDiff
using Optim
using StatsBase
using FreqTables
using LineSearches
using ProgressMeter
using ProfileView


#----------------------------------Functions-----------------------------------
function exponential_function(x)
  """
  Wrapper function for the exponential function. Used as link function.

  # Arguments
  - `float::x`: function input.
  ...
  """
  return exp.(x)
end

function logistic_function(x)
  """
  Computes value of the logistic function.

  # Arguments
  - `Float::x`: value at which the logistic function should be evaluated.
  ...
  """
  return 1 ./ (1 .+ exp.(-x))
end

#---------------------------------Distributions---------------------------------
function poisson_distribution(x, lambda)
    return exp(-lambda) * lambda^x / factorial(x)
end

function generalized_poisson_distribution(x, lambda, eta)
    if x <= 20
        return exp(-lambda-x*eta) * lambda*(lambda+x*eta)^(x-1) / factorial(x)
    else
        return exp(-lambda-x*eta) * lambda*(lambda+x*eta)^(x-1) / factorial(big(x))
    end
end

function bivariate_generalized_poisson(y, z, lambda, alpha1, alpha2, alpha3, eta)
    U = compute_U(alpha1, alpha2, alpha3)
    beta3 = compute_beta_i(lambda, U, alpha3)
    beta2 = compute_beta_i(lambda, U, alpha2)
    beta1 = compute_beta_i(lambda, U, alpha1)

    sum = 0.0
    for j in 0:min(y,z)
        sum = sum + (lambda*U*(1-alpha1-alpha3) + eta*(y-j))^(y-j-1) *
                    (lambda*U*(1-alpha1-alpha3) + eta*(z-j))^(z-j-1) *
                    (lambda*U*(alpha1+alpha3) + eta*(j))^(j-1) /
                    factorial(j) / factorial(y-j) / factorial(z-j) * exp(j*eta)
    end

    return sum * (beta1+beta3) * (lambda*U*(1-alpha1-alpha3))^2 * exp(-(beta1+beta3)-2*(lambda*U*(1-alpha1-alpha3))-y*eta-z*eta)
end





#-------------------------------helper functions--------------------------------
function compute_beta_i(lambda, U, alpha_i)
    return lambda*U*alpha_i
end

function compute_U(alpha1, alpha2, alpha3)
    return 1 / (1-alpha1-alpha2-alpha3)
end

function compute_zeta(lambda, U, alpha1, alpha3)
    return lambda * U * (1-2*alpha1-alpha3)
end


#--------------------------Likelihood relevant----------------------------------
function compute_g_r_y_z(r, y, z, lambda, alpha1, alpha2, alpha3, eta, max)
    U = compute_U(alpha1, alpha2, alpha3)
    beta3 = compute_beta_i(lambda, U, alpha3)
    beta2 = compute_beta_i(lambda, U, alpha2)
    beta1 = compute_beta_i(lambda, U, alpha1)
    zeta = compute_zeta(lambda, U, alpha1, alpha3)

    if isnothing(max)
        max = y
    end

    sum = 0.0
    for s in 0:max
        for v in 0:max
            for w in 0:max
                if ((r-s-v) >= 0) & ((z-r+v-w) >= 0) & ((y-s-v-w) >= 0)
                    sum = sum + generalized_poisson_distribution(s, beta3, eta) *
                                generalized_poisson_distribution(v, beta1, eta) *
                                generalized_poisson_distribution(w, beta1, eta) *
                                generalized_poisson_distribution(r-s-v, beta2, eta) *
                                generalized_poisson_distribution(z-r+v-w, lambda, eta) *
                                generalized_poisson_distribution(y-s-v-w, zeta, eta)
                end
            end
        end
    end

    return sum / bivariate_generalized_poisson(y, z, lambda, alpha1, alpha2, alpha3, eta)
end

function compute_g_r_y(y, r, alpha, eta, lambda)
    psi = eta * (1-alpha) / lambda
    if y < 20
        return factorial(y) / factorial(r) / factorial(y-r) * alpha *  (1-alpha) *
                (alpha + psi*r)^(r-1) * (1-alpha+psi*(y-r))^(y-r-1) /
                (1+ psi*y)^(y-1)
    else
        return factorial(big(y)) / factorial(big(r)) / factorial(big(y-r)) * alpha *  (1-alpha) *
                (alpha + psi*r)^(r-1) * (1-alpha+psi*(y-r))^(y-r-1) /
                (1+ psi*y)^(y-1)
    end
end

function compute_convolution_x_r_y(x, y, lambda, alpha, eta)
    sum = 0.0
    for r in 0:min(x, y)
        sum = sum + compute_g_r_y(y, r, alpha, eta, lambda) * generalized_poisson_distribution(x-r, lambda, eta)
    end
    return sum
end

function compute_negative_log_likelihood_GP1(lambdas, alpha, eta, data)

    sum = 0.0
    for t in 2:length(data)
        sum = sum - log(compute_convolution_x_r_y(data[t], data[t-1], lambdas[t], alpha, eta))
    end

    return sum
end

function compute_negative_log_likelihood_GP2(lambdas, alpha1, alpha2, alpha3, eta, data, max=nothing)

    sum = 0.0
    for t in 3:length(data)
        sum = sum - log(compute_convolution_x_r_y_z(data[t], data[t-1], data[t-2], lambdas[t], alpha1, alpha2, alpha3, eta, max))
    end

    return sum
end

function compute_convolution_x_r_y_z(x, y, z, lambda, alpha1, alpha2, alpha3, eta, max=nothing)
    sum = 0.0
    for r in 0:min(x, y+z)
        sum = sum + compute_g_r_y_z(r, y, z, lambda, alpha1, alpha2, alpha3, eta, max) * generalized_poisson_distribution(x-r, lambda, eta)
    end
    return sum
end



#------------------------Helper-------------------------------------------------
function compute_inverse_matrix(M)

  return inv(M)
end

function compute_hessian(f, x)
  return ForwardDiff.hessian(f, x)
end

function compute_mu_hat_gmm(data)
    sum = 0.0
    x_bar = mean(data)
    for t in 3:length(data)
        sum = sum + (data[t] - x_bar) * (data[t-1] - x_bar) * (data[t-2] - x_bar)#
    end
    return sum / length(data)
end

function compute_autocorrelation(data, order)
    x_t = data[1:(length(data)-order)]
    x_lag = data[(order+1):length(data)]

    return cor(x_t, x_lag) #/ (var(x_t) * var(x_lag))^0.5
end

function set_to_unit_interval(x)
    return max(min(x, 0.9999), 0.0001)
end

function reparameterize_alpha(parameter)
    alpha3 = parameter[1] * logistic_function(parameter[2])
    alpha1 = (parameter[1] - alpha3) / 2
    alpha2 = (1-alpha1-alpha3) * logistic_function(parameter[3])

    return alpha1, alpha2, alpha3
end

#-------------------------Optimization relevant----------------------------------
function minimize_pars_reparameterization_GP2(theta, data, covariates=nothing,
     link_function=exponential_function, max=nothing)

    lambdas = repeat([theta[5]], length(data))
    if  !isnothing(covariates)
        lambdas = link_function.(covariates * theta[5:5+(size(covariates, 2)-1)])
    end

    alpha1, alpha2, alpha3 = reparameterize_alpha(theta)

    return compute_negative_log_likelihood_GP2(lambdas, alpha1, alpha2, alpha3, theta[4], data, max)
end

function minimize_pars_reparameterization_Poisson2(theta, data, covariates=nothing,
     link_function=exponential_function, max=nothing)

    lambdas = repeat([theta[4]], length(data))
    if  !isnothing(covariates)
        lambdas = link_function.(covariates * theta[4:4+(size(covariates, 2)-1)])
    end

    alpha1, alpha2, alpha3 = reparameterize_alpha(theta)

    return compute_negative_log_likelihood_GP2(lambdas, alpha1, alpha2, alpha3, 0, data, max)
end

function minimize_pars_GP2(theta, data, covariates=nothing, link_function=exponential_function, max=nothing)

    lambdas = repeat([theta[5]], length(data))
    if  !isnothing(covariates)
        lambdas = link_function.(covariates * theta[5:5+(size(covariates, 2)-1)])
    end

    return compute_negative_log_likelihood_GP2(lambdas, theta[1], theta[2], theta[3], theta[4], data, max)
end

function minimize_pars_Poisson2(theta, data, covariates=nothing, link_function=exponential_function, max=nothing)

    lambdas = repeat([theta[4]], length(data))
    if  !isnothing(covariates)
        lambdas = link_function.(covariates * theta[4:4+(size(covariates, 2)-1)])
    end

    return compute_negative_log_likelihood_GP2(lambdas, theta[1], theta[2], theta[3], 0, data, max)
end


function minimize_pars_GP1(theta, data, covariates=nothing, link_function=exponential_function, max=nothing)

    lambdas = repeat([theta[3]], length(data))
    if  !isnothing(covariates)
        lambdas = link_function.(covariates * theta[3:3+(size(covariates, 2)-1)])
    end

    return compute_negative_log_likelihood_GP1(lambdas, theta[1], theta[2], data)
end

function minimize_pars_Poisson1(theta, data, covariates=nothing, link_function=exponential_function, max=nothing)

    lambdas = repeat([theta[2]], length(data))
    if  !isnothing(covariates)
        lambdas = link_function.(covariates * theta[2:2+(size(covariates, 2)-1)])
    end

    return compute_negative_log_likelihood_GP1(lambdas, theta[1], 0, data)
end


#------------------------Starting values -------------------------------------
function get_starting_values!(type, order, data, covariates, starting_values, n_parameter_without)
    if isnothing(starting_values)

        if (type == "GP") & (order == 2)
            starting_values = compute_starting_values_GP2_reparameterized("GP", data)
            if !isnothing(covariates)
                starting_values[n_parameter_without+1] = 0
                starting_values = vcat(starting_values, repeat([0], size(covariates,2)-1))
            end
        end

        if (type == "Poisson") & (order == 2)
        starting_values = compute_starting_values_GP2_reparameterized("Poisson", data)
            if !isnothing(covariates)
                starting_values[n_parameter_without+1] = 0
                starting_values = vcat(starting_values, repeat([0], size(covariates,2)-1))
            end
        end

        if (type == "GP") & (order == 1)
            starting_values = compute_starting_values_GP1("GP", data)
            if !isnothing(covariates)
                starting_values[n_parameter_without+1] = 0
                starting_values = vcat(starting_values, repeat([0], size(covariates,2)-1))
            end
        end

        if (type == "Poisson") & (order == 1)
        starting_values = compute_starting_values_GP1("Poisson", data)
            if !isnothing(covariates)
                starting_values[n_parameter_without+1] = 0
                starting_values = vcat(starting_values, repeat([0], size(covariates,2)-1))
            end
        end
    end
    return starting_values
end

function compute_starting_values_GP2_reparameterized(type, data)
    if type == "GP"
        eta = 1 - (mean(data) / var(data))^0.5
    elseif type == "Poisson"
        eta = 0
    end
    alpha3 = set_to_unit_interval(compute_mu_hat_gmm(data) / mean(data) / (1+ 2* eta) * (1-eta)^4)
    alpha1 = set_to_unit_interval(compute_autocorrelation(data, 1))
    alpha2 = set_to_unit_interval(compute_autocorrelation(data, 2))

    if alpha3 + alpha2 + alpha1 >= 1
        while alpha3 + alpha2 + alpha1 >= 1
            alpha3 = alpha3 * 0.8
            alpha2 = alpha2 * 0.8
            alpha1 = alpha2 * 0.8
        end
    end

    if alpha3 + 2*alpha1 >= 1
        while alpha3 + 2*alpha1 >= 1
            alpha3 = alpha3 * 0.8
            alpha1 = alpha2 * 0.8
        end
    end

    lambda = mean(data) * (1-alpha1-alpha2-alpha3) * (1-eta)

    if type == "GP"
        return [2*alpha1 + alpha3, log(alpha3 / (2*alpha1) ), log(alpha2 / (1-alpha1-alpha2-alpha3)), eta, lambda]
    elseif type == "Poisson"
        return [2*alpha1 + alpha3, log(alpha3 / (2*alpha1) ), log(alpha2 / (1-alpha1-alpha2-alpha3)), lambda]
    end
end

function compute_starting_values_GP1(type, data)

    alpha = compute_autocorrelation(data, 1)
    lambda = mean(data) * (1-alpha)

    if type == "GP"
        eta = 1 - (mean(data) / var(data))^0.5
        return [alpha, eta, lambda]
    elseif type == "Poisson"
        eta = 0
        return [alpha, lambda]
    end
end



#---------------------Bounds----------------------------------------------------
function get_bounds_GP2(type, covariates)

    lambda_lower = [0]
    lambda_upper = [Inf]
    if !isnothing(covariates)
        lambda_lower = repeat([-Inf], size(covariates, 2) )
        lambda_upper = repeat([Inf], size(covariates, 2) )
    end

    if type == "GP"
        lower = vcat( [0,-10,-10], [0],  lambda_lower)
        upper = vcat( [1, 10,10], [1],  lambda_upper)
    elseif type == "Poisson"
        lower = vcat( [0,-10,-10], lambda_lower)
        upper = vcat( [1, 10,10],  lambda_upper)
    end
    return lower, upper
end

function get_bounds_GP1(type, covariates)

    lambda_lower = [0]
    lambda_upper = [Inf]
    if !isnothing(covariates)
        lambda_lower = repeat([-Inf], size(covariates, 2) )
        lambda_upper = repeat([Inf], size(covariates, 2) )
    end

    if type == "GP"
        lower = vcat( [0], [0],  lambda_lower)
        upper = vcat( [1], [1],  lambda_upper)
    elseif type == "Poisson"
        lower = vcat( [0], lambda_lower)
        upper = vcat( [1],  lambda_upper)
    end
    return lower, upper
end

function get_bounds(order, type, covariates)
    if order == 1
        return get_bounds_GP1(type, covariates)
    elseif order == 2
        return get_bounds_GP2(type, covariates)
    end
end

#-----------------------cocoreg---------------------------------------------
function cocoReg(type, order, data, covariates=nothing,
                link_function=exponential_function, max_loop=nothing,
                starting_values=nothing, optimizer=Fminbox(LBFGS()))




    #-------------------------start dependent on type----------------------------------------------------------
    if order == 2
        if type == "GP"
            starting_values = get_starting_values!(type, order, data, covariates, starting_values, 4)
            fn = OnceDifferentiable(theta -> minimize_pars_reparameterization_GP2(theta, data,
                                                                            covariates,
                                                                            link_function,
                                                                            max_loop),
                                    starting_values,
                                    autodiff=:forward)
            f_alphas = theta -> minimize_pars_GP2(theta, data, covariates,
                                                link_function, max_loop)

        elseif type == "Poisson"
            starting_values = get_starting_values!(type, order, data, covariates, starting_values, 3)
            fn = OnceDifferentiable(theta -> minimize_pars_reparameterization_Poisson2(theta, data,
                                                                            covariates,
                                                                            link_function,
                                                                            max_loop),
                                    starting_values,
                                    autodiff=:forward)
            f_alphas = theta -> minimize_pars_Poisson2(theta, data, covariates,
                                    link_function, max_loop)
        end
    end
    #-----------------------------------------Order 1 models-------------------------
    if order == 1
        if type == "GP"
            starting_values = get_starting_values!(type, order, data, covariates, starting_values, 2)
            fn = OnceDifferentiable(theta -> minimize_pars_GP1(theta, data,
                                                                            covariates,
                                                                            link_function),
                                    starting_values,
                                    autodiff=:forward)
            f_alphas = theta -> minimize_pars_GP1(theta, data, covariates,
                                                link_function)

        elseif type == "Poisson"
            starting_values = get_starting_values!(type, order, data, covariates, starting_values, 1)
            fn = OnceDifferentiable(theta -> minimize_pars_Poisson1(theta, data,
                                                                            covariates,
                                                                            link_function),
                                    starting_values,
                                    autodiff=:forward)
            f_alphas = theta -> minimize_pars_Poisson1(theta, data, covariates,
                                    link_function)
        end
    end
    #--------------------------end dependent on type

    #write down constraints
    lower, upper = get_bounds(order, type, covariates)# -> make dependent on type and order

    #obtain fit
    fit = optimize(fn, lower, upper, starting_values,
                                        optimizer)#, Optim.Options(show_trace = true, show_every = 1,
                                        #iterations=1000, g_tol = 10^(-5), f_tol = 10^(-20)))
    parameter = Optim.minimizer(fit)

    #get alphas from reparameterized results
    if order == 2
        alpha1, alpha2, alpha3 = reparameterize_alpha(parameter)
        parameter[1:3] = [alpha1, alpha2, alpha3]
    end

    #construct output
    out = Dict("parameter" => parameter,
               "covariance_matrix" => compute_inverse_matrix(compute_hessian(f_alphas, parameter)),
               "log_likelihood" => -f_alphas(parameter)
               )

    #compute standard errors
    out["se"] = diag(out["covariance_matrix"]).^0.5
    return out
end



#------------------------test data------------------------------------------
function read_time_series(path)
    df = DataFrame(XLSX.readtable(path, 1)...)
    return df[:, 1]
end

function simulate_covariates(n, periodicity)
    return hcat(repeat([1], n) , sin.(Array(1:n) .* 2 .* pi ./ periodicity), cos.(Array(1:n) .* 2 .* pi ./ periodicity)  )
end
