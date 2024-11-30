import Pkg;
Pkg.add("Flux")

using Random
using Random:seed!
using Statistics
using Flux
using ScikitLearn
using CSV, DataFrames
using Printf

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier
@sk_import neural_network: MLPClassifier
@sk_import ensemble:StackingClassifier

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    # First we are going to set a line as defensive to check values
    @assert(all([in(value, classes) for value in feature]))
    
    # Second defensive statement, check the number of classes
    numClasses = length(classes)
    @assert(numClasses>1)
    
    if (numClasses==2)
        # Case with only two classes
        oneHot = reshape(feature.==classes[1], :, 1)
    else
        #Case with more than two clases
        oneHot =  BitArray{2}(undef, length(feature), numClasses)
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass])
        end
    end
    return oneHot
end

# Function to transorm in one-hot-encoding based on the values of the array passed
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

function holdOut(N::Int, P::Real)
    # We need to be sure that P is between 0 and 1:
    @assert ((P>=0.) & (P<=1.))
    
    idx = randperm(N)
    cut = round(Int64, length(idx)*P)
    test_idx = idx[1:cut]
    train_idx = idx[cut+1:end]
    return (train_idx, test_idx)
end

# Obtain values MEAN and STANDARD DESVIATION of an array to use in normalization:
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims=1), std(dataset, dims=1)
end

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real, 2})
	return minimum(dataset, dims = 1), maximum(dataset, dims = 1)
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real, 2},
	normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
	minValues = normalizationParameters[1]
	maxValues = normalizationParameters[2]
	dataset .-= minValues
	dataset ./= (maxValues .- minValues)
	# eliminate any atribute that do not add information
	dataset[:, vec(minValues .== maxValues)] .= 0
	return dataset
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real, 2})
	normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
end;

function normalizeMinMax(dataset::AbstractArray{<:Real, 2},
	normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
	normalizeMinMax!(copy(dataset), normalizationParameters)
end;

function normalizeMinMax(dataset::AbstractArray{<:Real, 2})
	normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset))
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real, 2},
	normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
	avgValues = normalizationParameters[1]
	stdValues = normalizationParameters[2]
	dataset .-= avgValues
	dataset ./= stdValues
	# Remove any atribute that do not have information
	dataset[:, vec(stdValues .== 0)] .= 0
	return dataset
end

function normalizeZeroMean(dataset::AbstractArray{<:Real, 2},
	normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
	normalizeZeroMean!(copy(dataset), normalizationParameters)
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real, 2})
	normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset))
end;

function normalizeData(train_inputs::AbstractArray{<:Real, 2},
    test_inputs::AbstractArray{<:Real, 2},
    normalizationType::Symbol)

    @assert normalizationType in [:ZeroMean, :MinMax]
    
    if normalizationType == :MinMax
        parameters = calculateMinMaxNormalizationParameters(train_inputs)
        # normalize the train using the previous parameters
        new_train_inputs = normalizeMinMax(train_inputs, parameters)
        # normalize the test using the  train parameters
        new_test_inputs = normalizeMinMax(test_inputs, parameters)
    elseif normalizationType == :ZeroMean
        parameters = calculateZeroMeanNormalizationParameters(train_inputs)
        # normalize the train using the previous parameters
        new_train_inputs = normalizeZeroMean(train_inputs, parameters)
        # normalize the test using the  train parameters
        new_test_inputs = normalizeZeroMean(test_inputs, parameters)
    end

    return (new_train_inputs, new_test_inputs)
end;

function crossvalidation(N::Int64, k::Int64)
	sorted_vector = collect(1:k)
    repeated_vector = repeat(sorted_vector, Int(ceil(N / k)))
    repeated_vector = repeated_vector[1:N]
    return shuffle!(repeated_vector)
end

function crossvalidation(targets::AbstractArray{Bool, 2}, k::Int64)
	@assert size(targets, 2) < 2 "Targets must be a 2D array with more than one column."
	N = size(targets, 1)
    num_classes = size(targets,2)
	indexes = zeros(Int, N)
    
    for class in 1:num_classes
        # Number of instances for the current class
        class_indexes = findall(targets[:, class])
		class_count = length(class_indexes,)
        # Generate cross-validation indexes for the current class
        stratified_indexes = crossvalidation(class_count, k)
        
        # Assign stratified_indexes to the corresponding positions in indexes
        indexes[class_indexes] .= stratified_indexes
    end
    return indexes
end

# Function to transform a feature into one-hot encoding in base of the classes passed by parameter
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    # First we are going to set a line as defensive to check values
    @assert(all([in(value, classes) for value in feature]))
    
    # Second defensive statement, check the number of classes
    numClasses = length(classes)
    @assert(numClasses>1)
    
    if (numClasses==2)
        # Case with only two classes
        oneHot = reshape(feature.==classes[1], :, 1)
    else
        #Case with more than two clases
        oneHot =  BitArray{2}(undef, length(feature), numClasses)
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass])
        end
    end
    return oneHot
end

oneHotEncoding(feature::AbstractArray{<:Any, 1}) = oneHotEncoding(feature, unique(feature));

oneHotEncoding(feature::AbstractArray{Bool, 1}) = reshape(feature, :, 1);


function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # el operador .>= compara cada elemento del output con threshold y devuelve true o false
    boolean_outputs = outputs .>= threshold
    return confusionMatrix(boolean_outputs, targets)
end

function printConfusionMatrix(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1})
    accuracy, error_rate, sensitivity, specificity, ppv, npv, f_score, cm = confusionMatrix(outputs, targets)

    # Display results
    println("Confusion Matrix:")
    println(cm)
    println("Accuracy: ", accuracy)
    println("Error Rate: ", error_rate)
    println("Sensitivity (Recall): ", sensitivity)
    println("Specificity: ", specificity)
    println("Precision: ", ppv)
    println("Negative Predictive Value: ", npv)
    println("F-Score: ", f_score)
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    accuracy, error_rate, sensitivity, specificity, ppv, npv, f_score, cm = confusionMatrix(outputs, targets, threshold)

    # Display results
    println("Confusion Matrix:")
    println(cm)
    println("Accuracy: ", accuracy)
    println("Error Rate: ", error_rate)
    println("Sensitivity (Recall): ", sensitivity)
    println("Specificity: ", specificity)
    println("Precision: ", ppv)
    println("Negative Predictive Value: ", npv)
    println("F-Score: ", f_score)
end

function oneVSall(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2})
      # Number of instances and classes
    #numInstances, numClasses = size(targets)
    numClasses = size(targets,2)
    @assert(numClasses>2)
    
    # Initialize outputs matrix
    outputs = Array{Float32}(undef, size(inputs,1), numClasses)

    # Loop through each class for training
    for numClass in 1:numClasses
        #supposed fit function
        model = fit(inputs, targets[:,[numClass]])
        
        # Store the model's outputs in the outputs matrix
        outputs[:, numClass] .= model(inputs)[:]
    end

    # Optional: Apply softmax function to outputs
    outputs = softmax(outputs')'  # Transpose, apply softmax, then transpose back

    # Get the maximum value for each row (pattern) to determine predicted classes
    vmax = maximum(outputs, dims=2)
    outputs = (outputs .== vmax)  # Create a Boolean matrix for predicted classes

    return outputs
end

# MULTICLASS CONFUSION MATRIX

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Ensure outputs and targets have the same number of columns
    @assert size(outputs, 2) == size(targets, 2) "Number of columns must match between outputs and targets."
    
    # Check for binary classification case (only 1 column)
    if size(outputs, 2) == 1
        # Call the previous binary confusionMatrix function
        return confusionMatrix1(outputs, targets)
    end
    
    # For multiclass case, initialize metrics vectors for each class
    numClasses = size(outputs, 2)
    recall = zeros(numClasses)
    specificity = zeros(numClasses)
    precision = zeros(numClasses)
    NPV = zeros(numClasses)
    F1 = zeros(numClasses)

    # Iterate over each class and calculate metrics for each
    for classIdx in 1:numClasses
        # Extract the binary vectors for the current class
        classOutputs = collect(outputs[:, classIdx])
        classTargets = collect(targets[:, classIdx])

        # Call the binary confusionMatrix function for the current class
        metrics = confusionMatrix1(classOutputs, classTargets)
        

        # Unpack the metrics and assign them to the corresponding class index
        recall[classIdx], specificity[classIdx], precision[classIdx], NPV[classIdx], F1[classIdx] = metrics
    end
    # Compute the confusion matrix (numClasses x numClasses)
    confusionMatrix = zeros(Int, numClasses, numClasses)
    for i in 1:numClasses
        for j in 1:numClasses
            confusionMatrix[i, j] = sum(outputs[:, i] .& targets[:, j])
        end
    end
    # Aggregate metrics based on macro or weighted strategy
    if weighted
        classCounts = sum(targets, dims=1)[:]
        totalPatterns = sum(classCounts)
        weights = classCounts / totalPatterns
        recall = sum(recall .* weights)
        specificity = sum(specificity .* weights)
        precision = sum(precision .* weights)
        NPV = sum(NPV .* weights)
        F1 = sum(F1 .* weights)
    else
        # Macro: average the metrics across classes
        recall = mean(recall)
        specificity = mean(specificity)
        precision = mean(precision)
        NPV = mean(NPV)
        F1 = mean(F1)
    end
    # Calculate accuracy
    accuracyValue = mean(outputs.==targets)
    errorRate = 1 - accuracyValue

    #return confusionMatrix, recall, specificity, precision, NPV, F1, accuracyValue, errorRate
    return accuracyValue, errorRate, recall, specificity, precision, NPV, F1, confusionMatrix
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Convert real-valued outputs to boolean values using classifyOutputs
    booleanOutputs = classifyOutputs(outputs)
    # Call the previous confusionMatrix function that handles boolean arrays
    return confusionMatrix(booleanOutputs, targets; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any}, targets::AbstractArray{<:Any}; weighted::Bool=true)
    # Ensure that outputs and targets are of the same length
    @assert length(outputs) == length(targets) "Outputs and targets must be of the same length."
    
    # Check that all output classes are included in the desired output classes
    @assert all([in(output, unique(targets)) for output in outputs]) "All output classes must be present in the target classes."
    
    # Get unique classes from outputs and targets
    unique_outputs = unique(outputs)
    unique_targets = unique(targets)

    # One-hot encode the outputs and targets
    encoded_outputs = oneHotEncoding(outputs, unique_outputs)
    encoded_targets = oneHotEncoding(targets, unique_targets)

    # Call the previous confusionMatrix function with encoded matrices
    return confusionMatrix(encoded_outputs, encoded_targets; weighted=weighted)
end

function confusionMatrix1(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # Confusion matrix components
    TP = sum(outputs .& targets)      # True Positives
    TN = sum(.!outputs .& .!targets)  # True Negatives
    FN = sum(outputs .& .!targets)    # False Negatives
    FP = sum(.!outputs .& targets)    # False Positives
    
    # Confusion Matrix
    cm = [TN FP; FN TP]
    
    # Accuracy
    accuracy = (TN + TP) / (TN + TP + FN + FP)
    
    # Error rate
    error_rate = (FP + FN) / (TN + TP + FN + FP)
    
    # Sensitivity (Recall)
    sensitivity = TP / (FN + TP)
    
    # Specificity 
    specificity = TN / (FP + TN)
    
    # Positive Predictive Value (Precision)
    ppv = TP / (TP + FP)
    
    # Negative Predictive Value
    npv = TN / (TN + FN)
    
    # F-Score (Harmonic mean of Precision and Recall)
    #Operador ternario (? :) si la condicion es verdadera devuelve 0 si no hace el calculo
    f_score = (sensitivity == 0 && ppv == 0) ? 0 : 2 / ((1/ppv) + (1/sensitivity))
    
    return accuracy, error_rate, sensitivity, specificity, ppv, npv, f_score, cm
end

function genModel(modelsHyperParameters:: Dict{String})
    estimator = modelsHyperParameters["estimator"]
    if estimator == :SVM
        return SVC(kernel=modelsHyperParameters["kernel"],
        degree = modelsHyperParameters["degree"],
        C = modelsHyperParameters["C"])
    elseif estimator == :DecisionTree
        return DecisionTreeClassifier(max_depth = modelsHyperParameters["max_depth"],
        random_state = modelsHyperParameters["random_state"])

    elseif estimator == :KNN
        return KNeighborsClassifier(n_neighbors = modelsHyperParameters["k"])
        
    elseif estimator == :ANN
        return MLPClassifier(hidden_layer_sizes = modelsHyperParameters["topology"],
        max_iter = modelsHyperParameters["maxEpochs"],
        learning_rate_init = modelsHyperParameters["learningRate"])
    else
        throw(ArgumentError("Unknown estimator: $estimator"))
    end
end

function best_model_positions(models_data::Vector{Any})

    best_models = Dict{Symbol, Tuple{Int, Float64}}()

    for model_data in models_data
        index = model_data[1]
        estimator = model_data[2]
        accuracy = model_data[3]

        if !haskey(best_models, estimator) || accuracy > best_models[estimator][2]
            best_models[estimator] = (index, accuracy)
        end
    end

    return [v[1] for (k, v) in best_models]
end


function print_confusion_matrix(confusion_matrix::AbstractMatrix{Int}, class_labels::Vector{String}=nothing)
    num_classes = size(confusion_matrix, 1)

    # Print header if class_labels are provided
    if !isnothing(class_labels)
        println("CONFUSION MATRIX:")
        print("          ")
        for label in class_labels
            print(@sprintf("%10s", label))
        end
        println()
    else
        println("Confusion Matrix:")
    end
    
    # Print each row of the confusion matrix with formatted spacing
    for i in 1:num_classes
        print(@sprintf("%10s", class_labels[i]))  # Print class label or row index
        for j in 1:num_classes
            print(@sprintf("%10d", confusion_matrix[i, j]))
        end
        println()
    end
end