include("utils/utils.jl")

##########################################################################
#
#   Global DataFrames to store results
#
##########################################################################
module DataResults

using DataFrames

export metrics
export models_hyper_parameters
export models_accuracies

# Preparation of empty DataFrame for storing results
metric_names = [:approach, :accuracy, :error_rate, :recall, :specificity, :ppv, :npv, :f_score]
metric_types = [String[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]]
metrics = DataFrame(metric_types, metric_names)

# Dictionary for model hyperparameters
models_hyper_parameters = [
    # ANN configurations
    Dict("estimator" => :ANN, "topology" => (64,), "maxEpochs" => 200, "learningRate" => 0.01, "name" => "ANN_64"),
    Dict("estimator" => :ANN, "topology" => (128,), "maxEpochs" => 500, "learningRate" => 0.005, "name" => "ANN_128"),
    Dict("estimator" => :ANN, "topology" => (64, 32), "maxEpochs" => 400, "learningRate" => 0.001, "name" => "ANN_64_32"),
    Dict("estimator" => :ANN, "topology" => (128, 64), "maxEpochs" => 200, "learningRate" => 0.01, "name" => "ANN_128_64"),
    Dict("estimator" => :ANN, "topology" => (256,), "maxEpochs" => 300, "learningRate" => 0.05, "name" => "ANN_256"),
    Dict("estimator" => :ANN, "topology" => (128, 64, 32), "maxEpochs" => 200, "learningRate" => 0.01, "name" => "ANN_128_64_32"),
    Dict("estimator" => :ANN, "topology" => (64, 64), "maxEpochs" => 250, "learningRate" => 0.005, "name" => "ANN_64_64"),
    Dict("estimator" => :ANN, "topology" => (256, 128), "maxEpochs" => 200, "learningRate" => 0.001, "name" => "ANN_256_128"),

    # SVM configurations
    Dict("estimator" => :SVM, "kernel" => "rbf", "C" => 0.5, "degree"=>2, "name" => "SVM_rbf_05"),
    Dict("estimator" => :SVM, "kernel" => "linear", "C" => 1.0, "degree"=>4, "name" => "SVM_linear_1"),
    Dict("estimator" => :SVM, "kernel" => "poly", "C"=> 0.01 , "degree" => 2, "name" => "SVM_poly_001"),
    Dict("estimator" => :SVM, "kernel" => "sigmoid", "C" => 1.0, "degree" => 3, "name" => "SVM_sigmoid_1"),
    Dict("estimator" => :SVM, "kernel" => "rbf", "C" => 0.01, "degree" => 3, "name" => "SVM_rbf_001"),
    Dict("estimator" => :SVM, "kernel" => "poly", "C"=> 5, "degree" => 3, "name" => "SVM_poly_5"),
    Dict("estimator" => :SVM, "kernel" => "linear", "C"=> 0.5, "degree" => 3, "name" => "SVM_linear_05"),
    Dict("estimator" => :SVM, "kernel" => "sigmoid", "C" => 10.0, "degree" => 5, "name" => "SVM_sigmoid_10"),

    # Decision Tree configurations
    Dict("estimator" => :DecisionTree, "max_depth" => 3, "random_state" => 42, "name" => "DT_3"),
    Dict("estimator" => :DecisionTree, "max_depth" => 5, "random_state" => 42, "name" => "DT_5"),
    Dict("estimator" => :DecisionTree, "max_depth" => 7, "random_state" => 42, "name" => "DT_7"),
    Dict("estimator" => :DecisionTree, "max_depth" => 11, "random_state" => 42, "name" => "DT_11"),
    Dict("estimator" => :DecisionTree, "max_depth" => 15, "random_state" => 42, "name" => "DT_15"),
    Dict("estimator" => :DecisionTree, "max_depth" => 21, "random_state" => 42, "name" => "DT_21"),

    # kNN configurations
    Dict("estimator" => :KNN, "k" => 3, "name" => "KNN_3"),
    Dict("estimator" => :KNN, "k" => 5, "name" => "KNN_5"),
    Dict("estimator" => :KNN, "k" => 7, "name" => "KNN_7"),
    Dict("estimator" => :KNN, "k" => 9, "name" => "KNN_9"),
    Dict("estimator" => :KNN, "k" => 11, "name" => "KNN_11"),
    Dict("estimator" => :KNN, "k" => 15, "name" => "KNN_15")
]

models_accuracies = DataFrame(Model = String[], MinMax = Float64[])
for m in models_hyper_parameters
    push!(models_accuracies, (m["name"], 0))
end

end


##########################################################################
#
#   DATA PREPARATION 
#
##########################################################################

module DataPreparation

using ..ML1Utils

using Random
using CSV
using DataFrames
using Plots
using JSON
using Printf

export data, input_data, output_data
export train_input, train_output, test_input, test_output
export kFoldIndices

# Setting random seed for reproducibility
Random.seed!(12345)

# Load the dataset
comment("Loading dataset")
data = CSV.read("datasets/air_quality_health_impact_data.csv", DataFrame)

# Check the general distribution of data in the dataset
println(describe(data))

# We check the distribution of the different parameters in search of outliers in order to decide the better normalization method.
histos = []
for col in propertynames(data[:,2:13])
    push!(histos, histogram(data[:,col], title=col, legend=false))
end
for (i,h) in enumerate(histos)
    plot(h)
    savefig("images/histogram_"*names(data)[i+1])
end

# We check how many classes have of each kind
counts = combine(groupby(data, :HealthImpactClass), nrow => :Count)
comment("Elements per class:")
println(counts)

# and plot the distribution
cnt = counts.Count
labels = ["Class $i" for i in 0:4]
tot = sum(cnt)
datapct = [@sprintf("%.1f%%", x/tot*100) for x in cnt]

θ = (cumsum(cnt) - cnt/2) .* 360/sum(cnt)
scθ = sincosd.(θ)

p = pie(labels, cnt, title="Output class distribution")
for (s, sci) in zip(datapct, scθ)
    annotate!(1.1*sci[2], 1.1*sci[1], text(s, 8, :black))
end
plot(p, dpi=1000)
savefig("images/class_dist.png")

# Segmentation of data into inputs/outputs
comment("Segmentation of data into inputs (80%) and outputs (20%)")
input_data = Matrix(data[!, 1:13]);
output_data = Int.(data[!, 15]);

@assert input_data isa Matrix
@assert output_data isa Vector{Int64}

# Spliting data into train and test using holdOut
(tr_idx, test_idx) = holdOutStratified(output_data, 0.2)

train_input = input_data[tr_idx,:]
train_output = output_data[tr_idx]
test_input = input_data[test_idx,:]
test_output = output_data[test_idx]

train_output = collect(train_output)
test_output = collect(test_output)

println("Train Input Size: ", size(train_input))
println("Train Output Size: ", size(train_output), " Categories:", sort(unique(train_output)))
println("Test Input Size: ", size(test_input))
println("Test Output Size: ", size(test_output), " Categories:", sort(unique(test_output)))

# Check if have representation of each classes in train and test
column = train_output[:, 1]
countclasses = Dict(c => count(==(c), column) for c in 0:4)
comment("Elements per class in train set:")
println(json(countclasses,4))

column = test_output[:, 1]
countclasses = Dict(c => count(==(c), column) for c in 0:4)
comment("Elements per class in test set:")
println(json(countclasses,4))


# Making the folds for approaches 3 and 4:
kFoldIndices = crossvalidation(size(train_output, 1), 5)

end

##########################################################################
#
#   1st APPROACH: MIN-MAX NORMALIZATION
#
##########################################################################

module Approach1

using ..ML1Utils
using ..DataResults
using ..DataPreparation

using DataFrames
using ScikitLearn

@sk_import svm:SVC
@sk_import ensemble:StackingClassifier

export train_input_minmax


comment("1. First Approach: min-max normalization")
train_input_minmax, test_input_minmax = normalizeData(train_input, test_input, :MinMax)

comment("Pre-normalization characteristics")
println(describe(DataFrame(train_input, names(data)[1:13])))

comment("Post-normalization characteristics")
println(describe(DataFrame(train_input_minmax, names(data)[1:13])))

comment("Model training ...")
accuracies = []
models = []
for (idx, modelHyperParameters) in enumerate(models_hyper_parameters)
    model = deepcopy(genModel(modelHyperParameters))
    fit!(model, train_input_minmax, train_output)
    acc = score(model, test_input_minmax, test_output)
    push!(accuracies, (idx, modelHyperParameters["estimator"], acc))
    push!(models, deepcopy(model))
    models_accuracies[models_accuracies.Model.==modelHyperParameters["name"], "MinMax"] .= acc
end

comment("Selecction of best models, ensemble and training")
best_models_position = bestModelPositions(accuracies)

stacking_classifier = StackingClassifier(
    estimators = [("m$(idx)_" * string(models_hyper_parameters[idx]["estimator"]), models[idx]) for idx in best_models_position],
    final_estimator = SVC(probability = true), n_jobs = -1)
fit!(stacking_classifier, train_input_minmax, train_output)

comment("Metrics of the stack ensemble")
outputs = stacking_classifier.predict(test_input_minmax)
categories = sort(unique(output_data))

x = copy(outputs)
y = vec(copy(test_output))
cm = confusionMatrix(x, y)
metrics_minmax = [
    "MinMax",
    cm["accuracy"],
    cm["error_rate"],
    cm["sensitivity"],
    cm["specificity"],
    cm["positive_predictive_value"],
    cm["negative_predictive_value"],
    cm["f_score"]
]
push!(metrics, metrics_minmax)

println("METRICS 1st APPROACH (Min-Max Normalization):")
println("---------------------")
println("Accuracy: ", cm["accuracy"])
println("Error Rate: ", cm["error_rate"])
println("Sensitivity (Recall): ", cm["sensitivity"])
println("Specificity: ", cm["specificity"])
println("Precision: ", cm["positive_predictive_value"])
println("Negative Predictive Value: ", cm["negative_predictive_value"])
println("F-Score: ", cm["f_score"])
println("----------------------")
println()

printConfusionMatrix(cm["confusion_matrix"], ["Class $i" for i in 0:4])

end

##########################################################################
#
#   2nd APPROACH: ZERO-MEAN NORMALIZATION
#
##########################################################################

module Approach2

using ..ML1Utils
using ..DataResults
using ..DataPreparation

using DataFrames
using ScikitLearn

@sk_import svm:SVC
@sk_import ensemble:StackingClassifier


comment("2. Second Approach: zero-mean normalization")
train_input_zm, test_input_zm = normalizeData(train_input, test_input, :ZeroMean)

comment("Post-normalization characteristics")
println(describe(DataFrame(train_input_zm, names(data)[1:13])))

comment("Model training ...")
accuracies = []
models = []
models_accuracies[!, :ZeroMean] .= 0.0
for (idx, modelHyperParameters) in enumerate(models_hyper_parameters)
    model = genModel(modelHyperParameters)
    fit!(model, train_input_zm, train_output)
    acc = score(model, test_input_zm, test_output)
    push!(accuracies, (idx, modelHyperParameters["estimator"], acc))
    push!(models, deepcopy(model))
    models_accuracies[models_accuracies.Model.==modelHyperParameters["name"], "ZeroMean"] .= acc
end;

comment("Selecction of best models, ensemble and training")
best_models_position = bestModelPositions(accuracies)

stacking_classifier = StackingClassifier(
    estimators = [("m$(idx)_" * string(models_hyper_parameters[idx]["estimator"]), models[idx]) for idx in best_models_position],
    final_estimator = SVC(probability = true), n_jobs = -1)

fit!(stacking_classifier, train_input_zm, train_output)

comment("Metrics of the stack ensemble")
outputs = stacking_classifier.predict(test_input_zm)
categories = sort(unique(output_data))

x = copy(outputs)
y = vec(copy(test_output))

cm = confusionMatrix(x, y)
metrics_zm = [
    "ZeroMean",
    cm["accuracy"],
    cm["error_rate"],
    cm["sensitivity"],
    cm["specificity"],
    cm["positive_predictive_value"],
    cm["negative_predictive_value"],
    cm["f_score"]
]
push!(metrics, metrics_zm)

# Display results
println("METRICS 2nd APPROACH (Zero-Mean Normalization):")
println("------------------------------------------")
println("Accuracy: ", cm["accuracy"])
println("Error Rate: ", cm["error_rate"])
println("Sensitivity (Recall): ", cm["sensitivity"])
println("Specificity: ", cm["specificity"])
println("Precision: ", cm["positive_predictive_value"])
println("Negative Predictive Value: ", cm["negative_predictive_value"])
println("F-Score: ", cm["f_score"])
println("-------------------------------------------")
println()

printConfusionMatrix(cm["confusion_matrix"], ["Class $i" for i in 0:4])

end

##########################################################################
#
#   3rd APPROACH: CROSSVALIDATION
#
##########################################################################

module approach3

using ..ML1Utils
using ..DataResults
using ..DataPreparation

# Normalization with min-max
train_input_minmax, test_input_minmax = normalizeData(train_input, test_input, :MinMax)

metrics_given = trainClassEnsemble(models_hyper_parameters,
                                    (train_input_minmax, train_output), 
                                    kFoldIndices)

# storing accuracies
models_accuracies[!, :CrossVal] .= 0.0
for i in eachindex(metrics_given["models_accuracies"])
    models_accuracies[i, :CrossVal] = metrics_given["models_accuracies"][i][3]
end

metrics_cv = [
    "CrossVal",
    metrics_given["accuracy"][1],
    metrics_given["error_rate"][1],
    metrics_given["sensitivity"][1],
    metrics_given["specificity"][1],
    metrics_given["positive_predictive_value"][1],
    metrics_given["negative_predictive_value"][1],
    metrics_given["f_score"][1]
]
push!(metrics, metrics_cv)

# Display results
println("METRICS 3rd APPROACH (Cross-Validation):")
println("------------------------------------------")
println("Accuracy: ", metrics_given["accuracy"][1])
println("Error Rate: ", metrics_given["error_rate"][1])
println("Sensitivity (Recall): ", metrics_given["sensitivity"][1])
println("Specificity: ", metrics_given["specificity"][1])
println("Precision: ", metrics_given["positive_predictive_value"][1])
println("Negative Predictive Value: ", metrics_given["negative_predictive_value"][1])
println("F-Score: ", metrics_given["f_score"][1])
println("-------------------------------------------")
println()

end

##########################################################################
#
#   4th APPROACH: DIMENSIONALITY REDUCTION
#
##########################################################################

module approach4

using ..ML1Utils
using ..DataResults
using ..DataPreparation

using ScikitLearn

@sk_import decomposition:PCA


# Normalization with min-max
train_input_minmax, test_input_minmax = normalizeData(train_input, test_input, :MinMax)

# Reduction to 3 parameters
comment("Dimensionality reduction to 3 features")
#Define the PCA object and the number of componentes that are desired
pca_3 = PCA(3)

#Ajust the matrix acording to the train data
fit!(pca_3, train_input_minmax)

#Once it is ajusted it can be used to transform the data
pca_train_3 = pca_3.transform(train_input_minmax)
pca_test_3 = pca_3.transform(test_input_minmax)

print("Train Patterns ", size(train_input_minmax), " -> ", size(pca_train_3))
print("Train Patterns ", size(test_input_minmax), " -> ", size(pca_test_3))

@assert (size(train_input_minmax)[1],3) == size(pca_train_3)
@assert (size(test_input_minmax)[1],3) == size(pca_test_3)

# one hot encoding only for graph
one_hot_train_output = oneHotEncoding(train_output)

colors = [:green, :red, :blue, :orange, :purple]
target_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]

# Ploting
drawResults(pca_train_3, one_hot_train_output; colors=colors, target_names=target_names, filename="images/dim_red_3.png")

# training
metrics_given = trainClassEnsemble(models_hyper_parameters,
                                    (pca_train_3, train_output),
                                    kFoldIndices)

# storing model accuracies
models_accuracies[!, :PCA_3] .= 0.0
for i in eachindex(metrics_given["models_accuracies"])
    models_accuracies[i, :PCA_3] = metrics_given["models_accuracies"][i][3]
end

metrics_pca3 = [
    "PCA_3",
    metrics_given["accuracy"][1],
    metrics_given["error_rate"][1],
    metrics_given["sensitivity"][1],
    metrics_given["specificity"][1],
    metrics_given["positive_predictive_value"][1],
    metrics_given["negative_predictive_value"][1],
    metrics_given["f_score"][1]
]
push!(metrics, metrics_pca3)

# Display results
println("METRICS 4th APPROACH (PCA 3 dimensions):")
println("------------------------------------------")
println("Accuracy: ", metrics_given["accuracy"][1])
println("Error Rate: ", metrics_given["error_rate"][1])
println("Sensitivity (Recall): ", metrics_given["sensitivity"][1])
println("Specificity: ", metrics_given["specificity"][1])
println("Precision: ", metrics_given["positive_predictive_value"][1])
println("Negative Predictive Value: ", metrics_given["negative_predictive_value"][1])
println("F-Score: ", metrics_given["f_score"][1])
println("-------------------------------------------")
println()

# Reduction to 95%
comment("Dimensionality reduction to 95% features")

## Apply a reduction conserving the 95% of the representation of the data
pca_95 = PCA(0.95)
fit!(pca_95, train_input_minmax)

#Once it is ajusted it can be used to transform the data
pca_train_95 = pca_95.transform(train_input_minmax)
pca_test_95 = pca_95.transform(test_input_minmax)

print("Train Patterns ", size(train_input_minmax), " -> ", size(pca_train_95))
print("Train Patterns ", size(test_input_minmax), " -> ", size(pca_test_95))

one_hot_train_output = oneHotEncoding(train_output)
colors = [:green, :red, :blue, :orange, :purple]
target_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]

# Call the function
drawResults(pca_train_95, one_hot_train_output; colors=colors, target_names=target_names, filename="images/dim_red_95.png")

# training
metrics_given = trainClassEnsemble(models_hyper_parameters,
                                    (pca_train_95, train_output),
                                    kFoldIndices)

# Storing accuracies
models_accuracies[!, :PCA_095] .= 0.0
for i in eachindex(metrics_given["models_accuracies"])
    models_accuracies[i, :PCA_095] = metrics_given["models_accuracies"][i][3]
end

metrics_pca95 = [
    "PCA_095",
    metrics_given["accuracy"][1],
    metrics_given["error_rate"][1],
    metrics_given["sensitivity"][1],
    metrics_given["specificity"][1],
    metrics_given["positive_predictive_value"][1],
    metrics_given["negative_predictive_value"][1],
    metrics_given["f_score"][1]
]
push!(metrics, metrics_pca95)

# Display results
println("METRICS 4th APPROACH (PCA 0.95 dimensions):")
println("------------------------------------------")
println("Accuracy: ", metrics_given["accuracy"][1])
println("Error Rate: ", metrics_given["error_rate"][1])
println("Sensitivity (Recall): ", metrics_given["sensitivity"][1])
println("Specificity: ", metrics_given["specificity"][1])
println("Precision: ", metrics_given["positive_predictive_value"][1])
println("Negative Predictive Value: ", metrics_given["negative_predictive_value"][1])
println("F-Score: ", metrics_given["f_score"][1])
println("-------------------------------------------")
println()

end

##########################################################################
#
#   RESULTS
#
##########################################################################

module finalResults

using ..ML1Utils
using ..DataResults
using ..DataPreparation

using Plots
using StatsPlots
using DataFrames


comment("Metrics for each approach")
println(metrics)

comment("Accuracies of each model")
println(models_accuracies)

# Figures: Barplot of metrics by approach
df = copy(metrics)

ctg = repeat(["MinMax", "ZeroMean", "CrossVal", "PCA_3", "PCA_095"], inner=7)
nam = repeat(names(df)[2:8], outer=5)
mm = collect(df[1,2:8])
zm = collect(df[2,2:8])
cv = collect(df[3,2:8])
pca3 = collect(df[4,2:8])
pca95 = collect(df[5,2:8])

p = groupedbar(nam, [mm zm cv pca3 pca95], group = ctg, 
                xlabel = "Metrics", ylabel = "Scores",
                title = "Metrics by Approach")
plot(p, dpi=1000)
savefig("images/metrics_per_approach.png")

# Figures: line plot of comparison of model accuracies
df = copy(models_accuracies)

p = plot(
    1:nrow(df), 
    Matrix(df[:, 2:end]),
    label = permutedims(names(df)[2:end]),
    title = "Comparison of Model Accuracies",
    xlabel = "Models",
    ylabel = "Accuracy",
    legend = :topright,
    size = (800, 600),
    lw = 2,
    marker = :circle,
    markersize = 4
)
xticks!(1:nrow(df), df.Model, rotation = 45)
ylims!(0.7, 1.0)
plot(p, dpi=1000)
savefig("images/models_accuracies.png")

end

