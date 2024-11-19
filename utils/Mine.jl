# SOME FUNCTIONS FOR ML1 LAB

# Load IRIS dataset
#   4 columns for attributes, 1 column for output classification
#
# @return: a tuple with inputs and targets (in one-hot-encoding)
using DelimitedFiles
using Random
using Random:seed!
using Statistics
using Flux
using ScikitLearn

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

function loadIrisData()
    dataset = readdlm("../data/iris/iris.data",',')
    inputs = convert(Array{Float32,2}, dataset[:,1:4])
    targets = dataset[:,5]
    classes = unique(targets)
    numClasses = length(classes);
    oneHot = Array{Bool,2}(undef, length(targets), numClasses)
    for numClass = 1:numClasses
        oneHot[:,numClass] .= (targets.==classes[numClass])
    end
    targets = oneHot
    
    return (inputs, targets)
end;

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

# Function to transorm in one-hot-encoding based on the values of the array passed
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

# One-hot-encoding for boolean output, returning a single column.
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)

# Obtain values MIN and MAX of an array to use in normalization:
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end

# Obtain values MEAN and STANDARD DESVIATION of an array to use in normalization:
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims=1), std(dataset, dims=1)
end

# FUNCTIONS FOR NORMALIZATION USING MIN-MAX:

# MinMax normalization passing min and max values:
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1]
    maxValues = normalizationParameters[2]
    dataset .-= minValues
    dataset ./= (maxValues .- minValues)
    # eliminate any atribute that do not add information
    dataset[:, vec(minValues.==maxValues)] .= 0
    return dataset
end

# MinMax normalization of an array:
function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset))
end

# MinMax normalization creating a copy (not ovewright the original array)
function normalizeMinMax(dataset::AbstractArray{<:Real,2})
      normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset))
end

# FUNCTIONS FOR NORMALIZATION USING MEAN AND STD

# ZeroMean normalization passing mean and std values:
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
                        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    avgValues = normalizationParameters[1]
    stdValues = normalizationParameters[2]
    dataset .-= avgValues
    dataset ./= stdValues
    # Remove any atribute that do not have information
    dataset[:, vec(stdValues.==0)] .= 0
    return dataset
end

# ZeroMean normalization of an array:
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizeZeroMean!(dataset , calculateZeroMeanNormalizationParameters(dataset)) 
end

# ZeroMean normalization creating a copy (not ovewright the original array)
function normalizeZeroMean( dataset::AbstractArray{<:Real,2}) 
    normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset))
end

# FUNCTION FOR CLASSIFY OUTPUTS

function classifyOutputs(outputs::AbstractArray{<:Real,2}, threshold::Real=0.5) 
    numOutputs = size(outputs, 2)
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold
    else
        # Look for the maximum value using the findmax funtion
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2)
        # Set up then boolean matrix to everything false while max values are true.
        outputs = falses(size(outputs))
        outputs[indicesMaxEachInstance] .= true
        # Defensive check if all patterns are in a single class
        @assert(all(sum(outputs, dims=2).==1))
        return outputs
    end
end

# ACCURACY FUNCTIONS
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    mean(outputs.==targets)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)))
    # The number of columns will never be 2, because an output variable with 2 classes will be encoded as a 1-column matrix
    # and a variable with 3 classes will be encoded as a 3-column matrix.
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1])
    else
        return mean(all(targets .== outputs, dims=2))
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}, threshold::Real=0.5)
    accuracy(outputs.>=threshold, targets)
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2},
                threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)))
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1])
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets)
    end
end

# Hold-out functions
function holdOut(N::Int, P::Real)
    # We need to be sure that P is between 0 and 1:
    @assert ((P>=0.) & (P<=1.))
    
    idx = randperm(N)
    cut = round(Int64, length(idx)*P)
    test_idx = idx[1:cut]
    train_idx = idx[cut+1:end]
    return (train_idx, test_idx)
end

function holdOut(N::Int, Pval::Real, Ptest::Real) 
    # We need to be sure that Pval and Ptest are between 0 and 1:
    @assert ((Pval>=0.) & (Pval<=1.))
    @assert ((Ptest>=0.) & (Ptest<=1.))
    # and the sum of both are less than 1:
    @assert ((Pval+Ptest)<1.)
    
    (train_ini, test) = holdOut(N, Ptest)
    Pvalrecal = Pval / (1 - Ptest)

    (idx_train, idx_val) = holdOut(length(train_ini), Pvalrecal)

    train = train_ini[idx_train]
    val = train_ini[idx_val]
    return (train, val, test)
end


# Function for building an ANN

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann=Chain()
    numInputsLayer = numInputs
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer]
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]))
        numInputsLayer = numNeurons
    end
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end
    return ann
end

function trainClassANN(topology::AbstractArray{<:Int,1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000,
        minLoss::Real=0.0,
        learningRate::Real=0.01,
        maxEpochsVal::Int=20,
        showText::Bool=false) 

    # Load inputs and targets
    (x_train, y_train) = trainingDataset
    (x_val, y_val) = validationDataset
    (x_test, y_test) = testDataset

    # This function assumes that each sample is in a row
    # we are going to check the number of samples to have same inputs and targets
    # and that the output columns is the same in the three sets
    @assert(size(x_train,1)==size(y_train,1))
    if size(validationDataset[1])[1]>0
        @assert size(x_val, 1)==size(y_val, 1)
    end
    if size(validationDataset[1])[1]>0
        @assert size(y_train, 2)==size(y_val, 2)
    end
    if size(testDataset[1])[1]>0
        @assert size(x_test, 1)==size(y_test, 1)
    end
    if size(testDataset[1])[1]>0
        @assert size(y_train, 2)==size(y_test, 2)
    end
    
    # We define the ANN
    ann = buildClassANN(size(x_train,2), topology, size(y_train,2))
    
    # Setting up the loss function to reduce the error
    loss(model,x,y) = (size(y,1) == 1) ? Flux.Losses.binarycrossentropy(model(x),y) : Flux.Losses.crossentropy(model(x),y);
    
    # This vectos is going to contain the losses and precission on each training epoch
    trLosses = Float32[]
    valLosses= Float32[]
    testLosses = Float32[]
    
    # Inicialize the counter to 0
    numEpoch = 0
    numEpochsValidation = 0
    
    # Calcualte the loss without training
    valLoss = 0.0
    testLoss = 0.0
    trLoss = loss(ann, x_train', y_train')
    if size(validationDataset[1])[1]>0
        valLoss = loss(ann, x_val', y_val')
    end
    if size(testDataset[1])[1]>0
        testLoss = loss(ann, x_test', y_test')
    end
    bestValidationLoss = Inf
    bestModel = deepcopy(ann)
    
    #  Store this one for checking the evolution.
    push!(trLosses, trLoss)
    push!(valLosses, valLoss)
    push!(testLosses, testLoss)
    
    #  and give some feedback on the screen
    if showText
        println("Epoch ", numEpoch, ": loss: ", trLoss)
    end
    
    # Define the optimizer for the network
    opt_state = Flux.setup(Adam(learningRate), ann)
    
    # Start the training until it reaches one of the stop critteria
    while (numEpoch < maxEpochs) && (trLoss > minLoss) && (numEpochsValidation < maxEpochsVal)
    
        # For each epoch, we have to train and consequently transpose the pattern to have them in columns
        Flux.train!(loss, ann, [(x_train', y_train')], opt_state);
        numEpoch += 1
    
        # calculate the loss for this epoch
        trLoss = loss(ann, x_train', y_train')
    
        # store it
        push!(trLosses, trLoss)
    
        #  and give some feedback on the screen
        if showText
            println("Epoch ", numEpoch, ": loss: ", trLoss)
        end
    
        if size(validationDataset[1])[1] > 0
            # calculate the loss for this epoch
            valLoss = loss(ann, x_val', y_val')

            # store it
            push!(valLosses, valLoss)
            if (valLoss<bestValidationLoss)
                bestValidationLoss = valLoss
                bestModel = deepcopy(ann)
            else
                numEpochsValidation += 1
            end
            #  and give some feedback on the screen
            if showText
                println("Epoch ", numEpoch, ": loss: ", valLoss)
            end
        end
        if size(testDataset[1])[1] > 0
            # calculate the loss for this epoch
            testLoss = loss(ann, x_test', y_test')

            # store it
            push!(testLosses, testLoss)
            #  and give some feedback on the screen
            if showText
                println("Epoch ", numEpoch, ": loss: ", testLoss)
            end
        end
    end
    
    # return the network and the evolution of the error
    if size(validationDataset[1])[1] > 0 
        return (bestModel, trLosses, valLosses, testLosses)
    else
        return (ann, trLosses, valLosses, testLosses)
    end
end     

function trainClassANN(topology::AbstractArray{<:Int,1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000,
        minLoss::Real=0.0,
        learningRate::Real=0.01,
        maxEpochsVal::Int=20,
        showText::Bool=false)

    (x_tr, y_tr) = trainingDataset
    (x_val, y_val) = validationDataset
    (x_test, y_test) = testDataset

    trainClassANN(topology, (x_tr, reshape(y_tr, length(y_tr), 1)), 
            validationDataset=(x_val, reshape(y_val, length(y_val), 1)),
            testDataset=(x_test, reshape(y_test, length(y_test), 1)),
            maxEpochs=maxEpochs, 
            minLoss=minLoss, 
            learningRate=learningRate,
            maxEpochsVal=maxEpochsVal,
            showText=showText)
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
    accuracyValue = accuracy(outputs, targets)
    errorRate = 1 - accuracyValue

    return confusionMatrix, recall, specificity, precision, NPV, F1, accuracyValue, errorRate
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

function crossvalidation(N::Int64, k::Int64)
    folds = collect(1:k) # Vector with the k folds
    indices = repeat(folds, inner=Int(ceil(N/k)));
    shuffle!(indices);
    return indices[1:N];
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    indices = fill(0, size(targets,1))
    for col in eachcol(targets)
        indices[col] .= crossvalidation(sum(col), k)
    end
    return indices
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    return crossvalidation(oneHotEncoding(targets), k)
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},
    kFoldIndices::Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000,
    minLoss::Real=0.0,
    learningRate::Real=0.01,
    repetitionsTraining::Int=10,
    validationRatio::Real=0.0,
    maxEpochsVal::Int=20,
    showText::Bool=false)

    (inputs, targets) = trainingDataset

    # obtain the number of folds from the vector of indices
    numFolds = maximum(kFoldIndices)

    # Create one vector per metric (for ex., accuracy, f1, specificity)
    testAccuracies = Array{Float64,1}(undef, numFolds)
    testErrorRate = Array{Float64,1}(undef, numFolds)
    testSensitivity = Array{Float64,1}(undef, numFolds)
    testSpecifity = Array{Float64,1}(undef, numFolds)
    testPPV = Array{Float64,1}(undef, numFolds)
    testNPV = Array{Float64,1}(undef, numFolds)
    testFScore = Array{Float64,1}(undef, numFolds)
    testCM = Array{Float64,1}(undef, numFolds)
    annFold = Array{Any}(undef, numFolds)
    

    ann = nothing
    trLosses = nothing
    valLosses = nothing
    testLosses = nothing

    # For each fold, load inputs and targets
    for numFold in 1:numFolds
        trainingInputs = inputs[kFoldIndices.!=numFold, :]
        testInputs = inputs[kFoldIndices.==numFold, :]
        trainingTargets = targets[kFoldIndices.!=numFold, :]
        testTargets = targets[kFoldIndices.==numFold, :]


        testAccuraciesEachRepetition = Array{Float64,1}(undef, repetitionsTraining)
        testErrorRateEachRepetition = Array{Float64,1}(undef, repetitionsTraining)
        testSensitivityEachRepetition = Array{Float64,1}(undef, repetitionsTraining)
        testSpecifityEachRepetition = Array{Float64,1}(undef, repetitionsTraining)
        testPPVEachRepetition = Array{Float64,1}(undef, repetitionsTraining)
        testNPVEachRepetition = Array{Float64,1}(undef, repetitionsTraining)
        testFScoreEachRepetition = Array{Float64,1}(undef, repetitionsTraining)

        for numTraining in 1:repetitionsTraining
            if validationRatio > 0
                # to train an ANN with a validation set, it is needed to call the holdOut function.
                (trainingIndices, validationIndices) = holdOut(size(trainingInputs, 1),
                    validationRatio * size(trainingInputs, 1) / size(inputs, 1))

                #Reshape to match type on the function trainClassANN
                tr_inp = reshape(trainingInputs[trainingIndices], (:, 1))
                tr_out = reshape(trainingTargets[trainingIndices], (:, 1))
                val_inp = reshape(trainingInputs[validationIndices], (:, 1))
                val_out = reshape(trainingTargets[validationIndices], (:, 1))
                #test_inp = reshape(testInputs, length(testInputs), 1)
                #test_out = reshape(testTargets, length(testTargets), 1)

                #Remember to pass transferFunctions
                (ann, trLosses, valLosses, testLosses) = trainClassANN(topology,
                    (tr_inp, tr_out),
                    validationDataset=(val_inp, val_out),
                    testDataset=(testInputs, testTargets),
                    transferFunctions=transferFunctions,
                    maxEpochs=maxEpochs,
                    minLoss=minLoss,
                    learningRate=learningRate,
                    maxEpochsVal=maxEpochsVal,
                    showText=showText)
            else
                # Train ANN using training and test sets.
                (ann, trLosses, valLosses, testLosses) = trainClassANN(topology,
                    (trainingInputs, trainingTargets),
                    testDataset=(testInputs, testTargets),
                    transferFunctions=transferFunctions,
                    maxEpochs=maxEpochs,
                    minLoss=minLoss,
                    learningRate=learningRate,
                    maxEpochsVal=maxEpochsVal,
                    showText=showText)
            end

            # Calculate metrics with confussionMatrix() function and save them for each of the iterations
            ResAnn = ann(testInputs')

            _, testSensitivityEachRepetition[numTraining], testSpecifityEachRepetition[numTraining], testPPVEachRepetition[numTraining],
            testNPVEachRepetition[numTraining], testFScoreEachRepetition[numTraining], testAccuraciesEachRepetition[numTraining],
            testErrorRateEachRepetition[numTraining] = confusionMatrix(ResAnn, testTargets')

        end

        # do not forget to provide the result of averaging the values of these vectors for each metric
        annFold[numFold] = deepcopy(ann)
        testAccuracies[numFold] = mean(testAccuraciesEachRepetition)
        testErrorRate[numFold] = mean(testErrorRateEachRepetition)
        testSensitivity[numFold] = mean(testSensitivityEachRepetition)
        testSpecifity[numFold] = mean(testSpecifityEachRepetition)
        testPPV[numFold] = mean(testPPVEachRepetition)
        testNPV[numFold] = mean(testNPVEachRepetition)
        testFScore[numFold] = mean(testFScoreEachRepetition)

    end # kfold loop end
    #Looking for the best fold with F Score
    _, bestFold = findmax(testFScore)

    #Save the bes Ann to return it
    bestModel = deepcopy(annFold[bestFold])

    #Save the bes metrics 
    bestAccuracy = testAccuracies[bestFold][1]
    bestError = testErrorRate[bestFold][1]
    bestSensitivity = testSensitivity[bestFold][1]
    bestSpecifity = testSpecifity[bestFold][1]
    bestPPV = testPPV[bestFold][1]
    bestNPV = testNPV[bestFold][1]
    bestFScore = testFScore[bestFold][1]

    return Dict(
        "model" => bestModel,
        "accuracy" => bestAccuracy,
        "errorRate" => bestError,
        "sensitivity" => bestSensitivity,
        "specificity" => bestSpecifity,
        "PPV" => bestPPV,
        "NPV" => bestNPV,
        "FScore" => bestFScore,)
end

function trainClassANN(topology::AbstractArray{<:Int,1},
                        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
                        kFoldIndices::Array{Int64,1};
                        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
                        maxEpochs::Int=1000, 
                        minLoss::Real=0.0, 
                        learningRate::Real=0.01,
                        repetitionsTraining::Int=1, 
                        validationRatio::Real=0.0, 
                        maxEpochsVal::Int=20,
                        showText::Bool=false)
    (x_tr, y_tr) = trainingDataset
    trDataset = (x_tr, reshape(y_tr, length(y_tr), 1))
    return trainClassANN(topology, 
                            trDataset, 
                            kFoldIndices,
                            transferFunctions=transferFunctions,
                            maxEpochs=maxEpochs,
                            minLoss=minLoss, 
                            learningRate=learningRate, 
                            repetitionsTraining=repetitionsTraining,
                            validationRatio=validationRatio,
                            maxEpochsVal=maxEpochsVal,
                            showText=showText)
end

function modelCrossValidation(modelType::Symbol,
    modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2},
    targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1})
   
    # Initialize metrics
    numFolds = maximum(crossValidationIndices)
    
    metrics = Dict(
        "model" => Array{Any}(undef, numFolds),
        "accuracy" => Array{Float64,1}(undef, numFolds),
        "errorRate" => Array{Float64,1}(undef, numFolds),
        "sensitivity" => Array{Float64,1}(undef, numFolds),
        "specificity" => Array{Float64,1}(undef, numFolds),
        "PPV" => Array{Float64,1}(undef, numFolds),
        "NPV" => Array{Float64,1}(undef, numFolds),
        "FScore" => Array{Float64,1}(undef, numFolds)
    )
    #Si es tipo ANN llamamos al train de la U5 que ya nos devuelve las metricas
    if modelType == :ANN
        trainingDataset = (inputs, targets)
        metrics[1] = trainClassANN(modelHyperparameters["topology"], trainingDataset, crossValidationIndices,
                transferFunctions = modelHyperparameters["transferFunctions"], 
                maxEpochs = modelHyperparameters["maxEpochs"], 
				minLoss = modelHyperparameters["minLoss"],
				learningRate = modelHyperparameters["learningRate"],
				maxEpochsVal = modelHyperparameters["maxEpochsVal"])
    else
    # Cross-validation loop for the other modelss
        for numFold in 1:numFolds
            trainingInputs = inputs[crossValidationIndices.!=numFold, :]
            testInputs = inputs[crossValidationIndices.==numFold, :]
            trainingTargets = targets[crossValidationIndices.!=numFold, :]
            testTargets = targets[crossValidationIndices.==numFold, :]

            # Train model based on modelType
            model = nothing
            
            if modelType == :SVM
                model = SVC(kernel=modelHyperparameters["kernel"],
                degree = modelHyperparameters["degree"],
                gamma = modelHyperparameters["gamma"],
                C = modelHyperparameters["C"])

            elseif modelType == :DecisionTree
                model = DecisionTreeClassifier(max_depth = modelHyperparameters["max_depth"],
                random_state = modelHyperparameters["random_state"])

            elseif modelType == :kNN
                model = KNeighborsClassifier(modelHyperparameters["k"])

            end
            fit!(model, trainingInputs, trainingTargets)

            # Make predictions
            predictions = predict(model, testInputs)
            metrics[numFold]  = confusionMatrix(predictions, testTargets')

        end
        
    end

    bestFold = argmax(metrics["FScore"])
    metrics["model"][1] = metrics["model"][bestFold]

     #Save the best metrics 
     metrics["accuracy"][1] = metrics["accuracy"][bestFold]
     metrics["errorRate"][1] = metrics["errorRate"][bestFold]
     metrics["sensitivity"][1] = metrics["sensitivity"][bestFold]
     metrics["specificity"][1] = metrics["specificity"][bestFold]
     metrics["PPV"][1] = metrics["PPV"][bestFold]
     metrics["NPV"][1] = metrics["NPV"][bestFold]
     metrics["FScore"][1] = metrics["FScore"][bestFold]

    return metrics[1]
end

function genModel(estimator:: Symbol, modelsHyperParameters:: Dict{String})
    model = nothing

    if estimator == :SVM
        model = SVC(kernel=modelsHyperParameters["kernel"],
        degree = modelsHyperParameters["degree"],
        gamma = modelsHyperParameters["gamma"],
        C = modelsHyperParameters["C"])

    elseif estimator == :DecisionTree
        model = DecisionTreeClassifier(max_depth = modelsHyperParameters["max_depth"],
        random_state = modelsHyperParameters["random_state"])

    elseif estimator == :KNN
        model = KNeighborsClassifier(n_neighbors = modelsHyperParameters["k"])
        
    elseif estimator == :ANN
        if haskey(modelsHyperParameters, "validation_fraction") && modelsHyperParameters["validation_fraction"] > 0
            # Ponemos early stopping a true porque es obligatorio cuando hay validation fraction
            model = MLPClassifier(hidden_layer_sizes = modelsHyperParameters["topology"],
            max_iter = modelsHyperParameters["maxEpochs"],
            learning_rate_init = modelsHyperParameters["learningRate"],
            validation_fraction = modelsHyperParameters["validation_fraction"],
            early_stopping = true)
        else
            model = MLPClassifier(hidden_layer_sizes = modelsHyperParameters["topology"],
            max_iter = modelsHyperParameters["maxEpochs"],
            learning_rate_init = modelsHyperParameters["learningRate"])
        end
    end
    return model
end


function trainClassEnsemble(estimators::AbstractArray{Symbol,1},
    modelsHyperParameters::AbstractArray{Dict{String, <:Any},1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},
    kFoldIndices::Array{Int64,1})

    @assert length(estimators) == length(modelsHyperParameters)

    (inputs, targets) = trainingDataset

    # Punto 1: almacenar cada métrica en cada fold
    # Hacer un array de arrays para las metricas???
    accuracies = Float64[]
    error_rates = Float64[]
    sensitivities = Float64[]
    specificities = Float64[]
    ppvs = Float64[]
    npvs = Float64[]
    f_scores = Float64[]

    # Punto 2
    for numFold in 1:length(unique(kFoldIndices))
        # Preparar y dividir los datos con los índices de CV (4 matrices)
        trainingInputs = inputs[kFoldIndices.!=numFold, :]
        testInputs = inputs[kFoldIndices.==numFold, :]
        trainingTargets = targets[kFoldIndices.!=numFold, :]
        testTargets = targets[kFoldIndices.==numFold, :]

        # Reshape trainingTargets to ensure it’s 1D and not have warnings
        trainingTargets = reshape(trainingTargets, :)

        # Punto 3: Crear los modelos
        numModels = length(estimators)
        models = Array{Any}(undef, numModels)
        modelName = Dict(
            :ANN => "ANN",
            :SVM => "SVM",
            :DecisionTree => "DT",
            :KNN => "KNN",
        )
        for (index, estimator) in enumerate(estimators)
            model = genModel(estimator, modelsHyperParameters[index])

            # Punto 4: Entrenar el modelo
            fit!(model, trainingInputs, trainingTargets)

            models[index] = (estimator, deepcopy(model))
        end

        # Punto 6: Construir el modelo de ensamblaje (stacking)
        ensemble_model = StackingClassifier(
            estimators=[(string(modelName[name], "_", i), model) for (i, (name, model)) in enumerate(models)],
            final_estimator=SVC(probability=true),
            n_jobs=-1
        )

        # Entrenar el modelo de esamblaje
        fit!(ensemble_model, trainingInputs, trainingTargets)

        # Realizar predicciones en el conjunto de prueba

        predictions = ensemble_model.predict(testInputs)

        accuracy, error_rate, sensitivity, specificity, ppv, npv, f_score, cm = confusionMatrix1(predictions, vec(testTargets))

        # Almacenar cada métrica en los vectores correspondientes
        push!(accuracies, accuracy)
        push!(error_rates, error_rate)
        push!(sensitivities, sensitivity)
        push!(specificities, specificity)
        push!(ppvs, ppv)
        push!(npvs, npv)
        push!(f_scores, f_score)
    end

    # Calcular las medias y desviaciones estándar de las métricas
    mean_metrics = Dict(
        "mean_accuracy" => mean(accuracies), "std_accuracy" => std(accuracies),
        "mean_error_rate" => mean(error_rates), "std_error_rate" => std(error_rates),
        "mean_sensitivity" => mean(sensitivities), "std_sensitivity" => std(sensitivities),
        "mean_specificity" => mean(specificities), "std_specificity" => std(specificities),
        "mean_ppv" => mean(ppvs), "std_ppv" => std(ppvs),
        "mean_npv" => mean(npvs), "std_npv" => std(npvs),
        "mean_f_score" => mean(f_scores), "std_f_score" => std(f_scores)
    )

    return mean_metrics

end
