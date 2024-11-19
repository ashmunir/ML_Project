### UNIT 2 ###
using Statistics;
using DelimitedFiles;
using Flux;
using Flux.Losses;
using Random;
 
function oneHotEncoding(feature::AbstractArray{<:Any,1},      
    classes::AbstractArray{<:Any,1})
    # First we are going to set a line as defensive to check values
    @assert(all([in(value, classes) for value in feature]));
 
    # Second defensive statement, check the number of classes
    numClasses = length(classes);
    @assert(numClasses>1)
 
    if (numClasses==2)
        # Case with only two classes
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        #Case with more than two clases
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;
 
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));
 
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);
 
function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # eliminate any atribute that do not add information
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;
 
function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset));
end;
 
function normalizeMinMax( dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
normalizeMinMax!(copy(dataset), normalizationParameters);
end;
 
function normalizeMinMax( dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));
end;
 
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end;
 
function classifyOutputs(outputs::AbstractArray{<:Real,2};
    threshold::Real=0.5)
    numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Look for the maximum value using the findmax funtion
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Set up then boolean matrix to everything false while max values aretrue.
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Defensive check if all patterns are in a single class
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;
 
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    mean(outputs.==targets);
end;
 
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;
 
function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
    threshold::Real=0.5)
accuracy(outputs.>=threshold, targets);
end;
 
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
    threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;
 
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;                                                
 
# UNIT 3
function holdOut(N::Int, P::Real)
    #Generating random array
    index = randperm(N)
 
    #Split value
    split = N-round(Int, N * P)
 
    #Splitting sets from the array
    set1 = index[1:split]
    set2 = index[split+1:N]
    tuple = (set1,set2)
    return tuple
end;
function holdOut(N::Int, Pval::Real, Ptest::Real)
    #Generating random array
    index = randperm(N)
 
    #Split values
    split_train = N - (round(Int, N * Pval) + round(Int, N * Ptest))
    split_val = split_train + round(Int, N * Pval)
 
    #Splitting sets from the array
    set1 = index[1:split_train]
    set2 = index[split_train+1:split_val]
    set3 = index[split_val+1:N]
    tuple = (set1,set2,set3)
    return tuple
end;

# UNIT 4
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    TP = sum(outputs .& targets)
    TN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    error_rate = (FP + FN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN) 
    specificity = TN / (TN + FP)
    ppv = TP / (TP + FP)  # Positive Predictive Value
    npv = TN / (TN + FN)  # Negative Predictive Value
    f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity)

    return ([TP FP; FN TN],accuracy,error_rate,sensitivity,specificity,ppv,npv,f1_score)
end
function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    boolean_outputs = outputs .>= threshold
    return confusionMatrix(boolean_outputs,targets)
end
#UNIT 5
#### 4.1 Confusion Matrix functions
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    TP = sum(outputs .& targets)
    TN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)

    return [TP FP; FN TN]
end
#### 2 oneHotEncoding
function oneHotEncoding(feature::AbstractArray{<:Any,1},      
    classes::AbstractArray{<:Any,1})
# First we are going to set a line as defensive to check values
@assert(all([in(value, classes) for value in feature]));

# Second defensive statement, check the number of classes
numClasses = length(classes);
@assert(numClasses>1)

if (numClasses==2)
    # Case with only two classes
    oneHot = reshape(feature.==classes[1], :, 1);
else
    #Case with more than two clases
    oneHot =  BitArray{2}(undef, length(feature), numClasses);
    for numClass = 1:numClasses
        oneHot[:,numClass] .= (feature.==classes[numClass]);
    end;
end;
return oneHot;
end;

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);

#### 2 classifyOutputs
function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
    threshold::Real=0.5) 
    numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Look for the maximum value using the findmax funtion
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Set up then boolean matrix to everything false while max values aretrue.
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Defensive check if all patterns are in a single class
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;

#### 2 accuracy
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    mean(outputs.==targets);
end;
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)));
    # The number of columns will never be 2, because an output variable with 2 classes will be encoded as a 1-column matrix
    # and a variable with 3 classes will be encoded as a 3-column matrix.
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

using Statistics #For mean function

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    num_classes = size(outputs, 2)
    if num_classes != size(targets, 2)
        error("The number of columns in outputs and targets must be equal.")
    end
    
    # If only one class, call the previous confusionMatrix for binary classification
    if num_classes == 1
        return confusionMatrix(vec(outputs), vec(targets))
    elseif num_classes == 2
        error("Two-column matrices are not valid because they represent binary classification, which should be handled by a binary confusion matrix.")
    end
    
    # Initialize vectors for storing metrics for each class
    sensitivities = zeros(Float64, num_classes)
    specificities = zeros(Float64, num_classes)
    PPVs = zeros(Float64, num_classes)
    NPVs = zeros(Float64, num_classes)
    F1_scores = zeros(Float64, num_classes)
    
    # Iterate over each class and calculate metrics
    for class_idx in 1:num_classes
        outputs_class = outputs[:, class_idx]
        targets_class = targets[:, class_idx]
        
        # Call the binary confusion matrix function to get TP, TN, FP, FN
        cm = confusionMatrix(outputs_class, targets_class)
        TP, FP = cm[1, :]
        FN, TN = cm[2, :]

        # Calculate metrics for the current class
        sensitivity = TP + FN == 0 ? 0.0 : TP / (TP + FN)
        specificity = TN + FP == 0 ? 0.0 : TN / (TN + FP)
        PPV = TP + FP == 0 ? 0.0 : TP / (TP + FP)           # Positive Predictive Value
        NPV = TN + FN == 0 ? 0.0 : TN / (TN + FN)           # Negative Predictive Value
        F1 = PPV + sensitivity == 0 ? 0.0 : 2 * (PPV * sensitivity) / (PPV + sensitivity)
        
        # Store the metrics
        sensitivities[class_idx] = sensitivity
        specificities[class_idx] = specificity
        PPVs[class_idx] = PPV
        NPVs[class_idx] = NPV
        F1_scores[class_idx] = F1
    end
    
    # Initialize confusion matrix for the multi-class problem
    confusion_matrix = zeros(Int64, num_classes, num_classes)

    # Build the multi-class confusion matrix
    for i in 1:num_classes
        for j in 1:num_classes
            confusion_matrix[i, j] = sum(outputs[:, i] .& targets[:, j])
        end
    end

    total_samples = sum(confusion_matrix)

    # Aggregate metrics using macro or weighted average
    if weighted
        class_counts = sum(targets, dims=1)  # Number of instances per class

        # Weighted average of metrics
        sensitivity = sum((class_counts .* sensitivities) / total_samples)
        specificity = sum((class_counts .* specificities) / total_samples)
        PPV = sum((class_counts .* PPVs) / total_samples)
        NPV = sum((class_counts .* NPVs) / total_samples)
        F1 = sum((class_counts .* F1_scores) / total_samples)

    else
        sensitivity = mean(sensitivities)
        specificity = mean(specificities)
        PPV = mean(PPVs)
        NPV = mean(NPVs)
        F1 = mean(F1_scores)
    end
    
    acc = accuracy(outputs,targets)
    error_rate = 1 - acc
    
    return Dict(
        :ConfusionMatrix => confusion_matrix,
        :Accuracy => acc,
        :ErrorRate => error_rate,
        :Sensitivity => sensitivity,
        :Specificity => specificity,
        :PPV => PPV,
        :NPV => NPV,
        :F1 => F1
    )
end
function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    outputs = classifyOutputs(outputs) 
    confusionMatrix(outputs, targets, weighted=weighted)
end

function trainClassANN(topology::AbstractArray{<:Int,1},  
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = 
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = 
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)), 
    maxEpochs::Int = 1000, minLoss::Real = 0.0, learningRate::Real = 0.01,  
    maxEpochsVal::Int = 20, showText::Bool = false) 

    # Unpack datasets
    (inputs, targets) = trainingDataset;
    (val_inputs, val_targets) = validationDataset;
    (test_inputs, test_targets) = testDataset;

    # Check if inputs and targets sizes match
    @assert(size(inputs,1) == size(targets,1))

    # Define the ANN
    ann = buildClassANN(size(inputs,2), topology, size(targets,2))

    # Loss function
    loss(model, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y);

    # Training, validation, and test loss history
    trainingLosses = Float32[];
    validationLosses = Float32[];
    testLosses = Float32[];

    # Initialize epoch counter and loss
    numEpoch = 0;
    trainingLoss = loss(ann, inputs', targets');
    push!(trainingLosses, trainingLoss);

    # If validation set exists, calculate initial validation loss
    if size(val_inputs, 1) > 0
        validationLoss = loss(ann, val_inputs', val_targets');
        push!(validationLosses, validationLoss);
    end

    # If test set exists, calculate initial test loss
    if size(test_inputs, 1) > 0
        testLoss = loss(ann, test_inputs', test_targets');
        push!(testLosses, testLoss);
    end

    # Print the initial loss
    if showText
        println("Epoch ", numEpoch, ": training loss: ", trainingLoss);
        if !isempty(val_inputs)
            println(" - validation loss: ", validationLoss);
        end
        if !isempty(test_inputs)
            println(" - test loss: ", testLoss);
        end
    end

    # Define optimizer
    opt_state = Flux.setup(Adam(learningRate), ann);

    # Initialize for early stopping
    best_ann = deepcopy(ann); 
    best_val_loss = Inf;       
    epochs_without_improvement = 0;

    # Training loop
    while (numEpoch < maxEpochs) && (trainingLoss > minLoss) && (epochs_without_improvement < maxEpochsVal)

        # Train the model on the training set
        Flux.train!(loss, ann, [(inputs', targets')], opt_state);

        numEpoch += 1;

        # Calculate the losses
        trainingLoss = loss(ann, inputs', targets');
        push!(trainingLosses, trainingLoss);

        # If validation dataset is provided
        if  size(val_inputs, 1) > 0
            validationLoss = loss(ann, val_inputs', val_targets');
            push!(validationLosses, validationLoss);

            # Early stopping condition
            if validationLoss < best_val_loss
                best_val_loss = validationLoss;
                best_ann = deepcopy(ann);  
                epochs_without_improvement = 0;
            else
                epochs_without_improvement += 1;
            end
        end

        # If test dataset is provided
        if size(test_inputs, 1) > 0
            testLoss = loss(ann, test_inputs', test_targets');
            push!(testLosses, testLoss);
        end

        # Show training progress
        if showText
            println("Epoch ", numEpoch, ": training loss: ", trainingLoss);
            if !isempty(val_inputs)
                println(" - validation loss: ", validationLoss);
            end
            if !isempty(test_inputs)
                println(" - test loss: ", testLoss);
            end
        end

    end
    # If validation set was used, return the best model, else return the last one
    if  size(val_inputs, 1) > 0
        return (best_ann, trainingLosses, validationLosses, testLosses);
    else
        return (ann, trainingLosses, validationLosses, testLosses);
    end
    
end
function trainClassANN(topology::AbstractArray{<:Int,1},  
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
    maxEpochsVal::Int=20, showText::Bool=false)

    (inputs, targets) = trainingDataset;
    (val_inputs, val_targets) = validationDataset;
    (test_inputs, test_targets) = testDataset;

    trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)),
    (val_inputs, reshape(val_targets, length(val_targets), 1)),
    (test_inputs, reshape(test_targets, length(test_targets), 1)), transferFunctions,
    maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal, showText);

end

# Unit 5
using Random

function crossvalidation(N::Int, k::Int)
    # Create a sorted vector from 1 to k
    subset_indices = collect(1:k)
    # Repeat N/k times
    repeated_indices = repeat(subset_indices, 1, ceil(Int, N/k))
    # Save N first elements
    repeated_indices = repeated_indices[1:N]
    
    return shuffle!(repeated_indices)
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int)
    num_classes = size(targets,2)
    indices = Array{Int64,1}(undef, size(targets,1));
    
    # Iterate over each class 
    for class in 1:num_classes
        class_indices = findall(targets[:, class])
        num_class_elements = sum(targets[:, class])  
        
        # Ensure that there are at least k patterns for stratification
        if num_class_elements < k
            error("Class $class has fewer than $k patterns. Stratified cross-validation cannot be performed.")
        end

        # Update index vector positions
        indices[class_indices] .= crossvalidation(length(class_indices), k)
    end
    
    return indices
end

# include("utils.jl")

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    unique_classes = unique(targets)
    class_indices = Int.(zeros(size(targets, 1)))

    for class in unique_classes
        class_data_indices = (targets .== class)
        number_of_instances = sum(class_data_indices)
        @assert (number_of_instances .>= k) "there are not enough instanses cross-validation with $(k)"

        class_indices[class_data_indices] .= crossvalidation(number_of_instances, k)
    end

    return class_indices
end


function trainClassANN(topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
    kFoldIndices::Vector{Int}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, repetitionsTraining::Int=1, 
    validationRatio::Real=0.0, maxEpochsVal::Int=20)

    # Unpack datasets
    (inputs, targets) = trainingDataset
    numFolds = maximum(kFoldIndices)
    # Check if inputs and targets sizes match
    @assert(size(inputs,1) == size(targets,1))
    
    # Arrays for folds results
    all_metric_results_acc = Float32[]
    all_metric_results_f1 = Float32[]
    for fold in 1:numFolds
        # Split train and test sets
        train_inputs = inputs[kFoldIndices.!=fold, :]
        test_inputs = inputs[kFoldIndices.==fold, :]
        train_targets = targets[kFoldIndices.!=fold, :]
        test_targets = targets[kFoldIndices.==fold, :]
        
        # Training models for each fold
        fold_results_acc = Float32[]
        fold_results_f1 = Float32[]
        
        for numTraining in 1:repetitionsTraining
            # Split validation data if there is validationRatio
            if validationRatio > 0.0
                indices = holdOut(size(train_inputs,1), validationRatio * size(train_inputs, 1) / size(inputs, 1))
                train_inputs = inputs[indices[1], :]
                val_inputs = inputs[indices[2], :]
                train_targets = targets[indices[1], :]
                val_targets = targets[indices[2], :]
            end
            ann, trainingLosses, validationLosses, testLosses = trainClassANN(
                topology, (train_inputs, train_targets), validationDataset=(val_inputs, val_targets), 
                testDataset=(test_inputs, test_targets), transferFunctions=transferFunctions, maxEpochs=maxEpochs, 
                minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
            outputs = ann(test_inputs')'
            metrics  = confusionMatrix(outputs, test_targets)
            push!(fold_results_acc, metrics[:Accuracy])
            push!(fold_results_f1, metrics[:F1])
        end
        
        mean_result = mean(fold_results_acc)
        push!(all_metric_results_acc, mean_result)
        mean_result = mean(fold_results_f1)
        push!(all_metric_results_f1, mean_result)
    end
    all_metric_results_f1 = filter(x -> !isnan(x), all_metric_results_f1)
    all_metric_results_acc = filter(x -> !isnan(x), all_metric_results_acc)
    return mean(all_metric_results_acc), std(all_metric_results_acc), mean(all_metric_results_f1), std(all_metric_results_f1)
end

function trainClassANN(topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractVector{Bool}}, 
    kFoldIndices::Vector{Vector{Int}}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, repetitionsTraining::Int=1, 
    validationRatio::Real=0.0, maxEpochsVal::Int=20)

    # Convert targets into matrix
    inputs, targets = trainingDataset
    trainingTargets = reshape(trainingTargets, (length(trainingTargets), 1))

    return trainClassANN(topology, (inputs, trainingTargets), kFoldIndices;
                        transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss,
                        learningRate=learningRate, repetitionsTraining=repetitionsTraining,
                        validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)
end