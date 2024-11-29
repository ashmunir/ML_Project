using Pkg;

using DelimitedFiles;
using Statistics;
using Flux;
using Flux.Losses;
using Random
using Random: seed!
seed!(0)

Pkg.add("DelimitedFiles");
Pkg.add("Flux");

# Functions
function getData(path)
	return readdlm(path, ',')
end

function splitIris(dataset::AbstractArray{Any, 2})
	inputs = dataset[:, 1:4]
	targets = dataset[:, 5]
	return (inputs, targets)
end


loss(model, x, y) = (size(y, 1) == 1) ?
					Losses.binarycrossentropy(model(x), y) :
					Losses.crossentropy(model(x), y);

# One Hot
function oneHotEncoding(feature::AbstractArray{<:Any, 1},
	classes::AbstractArray{<:Any, 1})
	# First we are going to set a line as defensive to check values
	@assert(all([in(value, classes) for value in feature]))

	# Second defensive statement, check the number of classes
	numClasses = length(classes)
	@assert(numClasses > 1)

	if (numClasses == 2)
		# Case with only two classes
		oneHot = reshape(feature .== classes[1], :, 1)
	else
		#Case with more than two clases
		oneHot = BitArray{2}(undef, length(feature), numClasses)
		for numClass ∈ 1:numClasses
			oneHot[:, numClass] .= (feature .== classes[numClass])
		end
	end
	return oneHot
end;

oneHotEncoding(feature::AbstractArray{<:Any, 1}) = oneHotEncoding(feature, unique(feature));

oneHotEncoding(feature::AbstractArray{Bool, 1}) = reshape(feature, :, 1);

# Normalization
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real, 2})
	return minimum(dataset, dims = 1), maximum(dataset, dims = 1)
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real, 2})
	return mean(dataset, dims = 1), std(dataset, dims = 1)
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
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real, 2})
	normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset))
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real, 2},
	normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
	normalizeZeroMean!(copy(dataset), normalizationParameters)
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real, 2})
	normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset))
end;

# Classify outputs 
function classifyOutputs(outputs::AbstractArray{<:Real, 2};
	threshold::Real = 0.5)
	numOutputs = size(outputs, 2)
	@assert(numOutputs != 2)
	if numOutputs == 1
		return outputs .>= threshold
	else
		# Look for the maximum value using the findmax funtion
		(_, indicesMaxEachInstance) = findmax(outputs, dims = 2)
		# Set up then boolean matrix to everything false while max values aretrue.
		outputs = falses(size(outputs))
		outputs[indicesMaxEachInstance] .= true
		# Defensive check if all patterns are in a single class
		@assert(all(sum(outputs, dims = 2) .== 1))
		return outputs
	end
end;

# accuracy
# function accuracy(outputs::AbstractArray{Bool, 1}, targets::AbstractArray{Bool, 1})
# 	mean(outputs .== targets)
# end;
function accuracy(outputs::AbstractArray{Bool, 1}, targets::AbstractArray{Bool, 1})
	mean(outputs .== targets)
end


function accuracy(outputs::AbstractArray{Bool, 2}, targets::AbstractArray{Bool, 2})
	@assert(all(size(outputs) .== size(targets)))
	# The number of columns will never be 2, because an output variable with 2 classes will be encoded as a 1-column matrix
	# and a variable with 3 classes will be encoded as a 3-column matrix.
	if (size(targets, 2) == 1)
		return accuracy(outputs[:, 1], targets[:, 1])
	else
		return mean(all(targets .== outputs, dims = 2))
	end
end;

function accuracy(outputs::AbstractArray{<:Real, 1}, targets::AbstractArray{Bool, 1};
	threshold::Real = 0.5)
	accuracy(outputs .>= threshold, targets)
end;

function accuracy(outputs::AbstractArray{<:Real, 2}, targets::AbstractArray{Bool, 2};
	threshold::Real = 0.5)
	@assert(all(size(outputs) .== size(targets)))
	if (size(targets, 2) == 1)
		return accuracy(outputs[:, 1], targets[:, 1])
	else
		return accuracy(classifyOutputs(outputs; threshold = threshold), targets)
	end
end;

# Build a ann
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int, 1}, numOutputs::Int;
	transferFunctions::AbstractArray{<:Function, 1} = fill(σ, length(topology)))
	ann = Chain()
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
end;


# Train ANN First
function getLoss(ann, inputs, targets, loss_text, numEpoch, showText)
	loss_value = loss(ann, inputs', targets')
	if showText
		println("Epoch ", numEpoch, ": ", loss_text, " loss: ", loss_value)
	end

	return loss_value
end

function trainClassANN(topology::AbstractArray{<:Int, 1},
	dataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}},
	transferFunctions::AbstractArray{<:Function, 1} = fill(σ, length(topology)),
	maxEpochs::Int = 1000, minLoss::Real = 0.0, learningRate::Real = 0.01)

	(inputs, targets) = dataset

	# This function assumes that each sample is in a row
	# we are going to check the numeber of samples to have same inputs and targets
	@assert(size(inputs, 1) == size(targets, 1))

	# We define the ANN
	ann = buildClassANN(size(inputs, 2), topology, size(targets, 2), transferFunctions = transferFunctions)

	# Setting up the loss funtion to reduce the error
	loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

	# This vectos is going to contain the losses and precission on each training epoch
	trainingLosses = Float32[]

	# Inicialize the counter to 0
	numEpoch = 0
	# Calcualte the loss without training
	trainingLoss = loss(ann, inputs', targets')
	#  Store this one for checking the evolution.
	push!(trainingLosses, trainingLoss)
	#  and give some feedback on the screen
	println("Epoch ", numEpoch, ": loss: ", trainingLoss)

	# Define the optimazer for the network
	opt_state = Flux.setup(Adam(learningRate), ann)

	# Start the training until it reaches one of the stop critteria
	while (numEpoch < maxEpochs) && (trainingLoss > minLoss)

		# For each epoch, we habve to train and consequently traspose the pattern to have then in columns
		Flux.train!(loss, ann, [(inputs', targets')], opt_state)

		numEpoch += 1
		# calculate the loss for this epoch
		trainingLoss = loss(ann, inputs', targets')
		# store it
		push!(trainingLosses, trainingLoss)
		# shown it
		println("Epoch ", numEpoch, ": loss: ", trainingLoss)

	end

	# return the network and the evolution of the error
	return (ann, trainingLosses)
end;

function trainClassANN(topology::AbstractArray{<:Int, 1},
	(inputs, targets)::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 1}};
	transferFunctions::AbstractArray{<:Function, 1} = fill(σ, length(topology)),
	maxEpochs::Int = 1000, minLoss::Real = 0.0, learningRate::Real = 0.01)

	trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)); transferFunctions = transferFunctions,
		maxEpochs = maxEpochs, minLoss = minLoss, learningRate = learningRate)
end;

function trainClassANN(topology::AbstractArray{<:Int, 1},
	trainingDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}};
	validationDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}} =
	(Array{eltype(trainingDataset[1]), 2}(undef, 0, 0), falses(0, 0)),
	testDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}} =
	(Array{eltype(trainingDataset[1]), 2}(undef, 0, 0), falses(0, 0)),
	transferFunctions::AbstractArray{<:Function, 1} = fill(σ, length(topology)),
	maxEpochs::Int = 1000, minLoss::Real = 0.0, learningRate::Real = 0.01,
	maxEpochsVal::Int = 20, showText::Bool = false)

	# Training sets
	(train_inputs, train_targets) = trainingDataset

	# This function assumes that each sample is in a row
	# we are going to check the numeber of samples to have same inputs and targets
	@assert(size(train_inputs, 1) == size(train_targets, 1))

	# We define the ANN
	ann = buildClassANN(size(train_inputs, 2), topology, size(train_targets, 2), transferFunctions = transferFunctions)

	# These vectors are going to contain the losses and precission on each training, validation and test epoch
	trainingLosses = Float32[]
	validationLosses = Float32[]
	testLosses = Float32[]

	# Inicialize the counter to 0
	numEpoch = 0
	# We define patience like the counter who will stop the training if the validationLoss doesn't improve in MaxEpochVal Epoch
	patienceCount = 0
	# As we want to get the minimun loss for validation, we'll define our best initial value like the biggest value available (Inf)
	bestValidationLoss = Inf
	# We also want to store our best model making a deepcopy of our model (not only reference and also recursive)
	bestModel = deepcopy(ann)

	trainingLoss = getLoss(ann, train_inputs, train_targets, "Training", numEpoch, showText)

	#  Store this one for checking the evolution.
	push!(trainingLosses, trainingLoss)

	# If the validation set is not empty, we capture the loss 
	if !isempty(validationDataset[1])
		@assert(size(validationDataset[1], 1) == size(validationDataset[2], 1))

		valLoss = getLoss(ann, validationDataset[1], validationDataset[2], "Validation", numEpoch, showText)
		push!(validationLosses, valLoss)
		bestValidationLoss = valLoss
	end

	# Now the same with the test set
	if !isempty(testDataset[1])
		@assert(size(testDataset[1], 1) == size(testDataset[2], 1))

		testLoss = getLoss(ann, testDataset[1], testDataset[2], "Test", numEpoch, showText)
		push!(testLosses, testLoss)
	end

	# Define the optimazer for the network
	opt_state = Flux.setup(Adam(learningRate), ann)

	# Start the training until it reaches one of the stop critteria
	while (numEpoch < maxEpochs) && (trainingLoss > minLoss) && (patienceCount < maxEpochsVal)

		# For each epoch, we habve to train and consequently traspose the pattern to have then in columns
		Flux.train!(loss, ann, [(train_inputs', train_targets')], opt_state)

		numEpoch += 1

		trainingLoss = getLoss(ann, train_inputs, train_targets, "Training", numEpoch, showText)
		# store it
		push!(trainingLosses, trainingLoss)

		# Calculate the test set first
		if !isempty(testDataset[1])
			testLoss = getLoss(ann, testDataset[1], testDataset[2], "Test", numEpoch, showText)
			push!(testLosses, testLoss)
		end

		# If validation
		if !isempty(validationDataset[1])
			valLoss = getLoss(ann, validationDataset[1], validationDataset[2], "Validation", numEpoch, showText)

			push!(validationLosses, valLoss)
			# If there is an improve, we change the best model
			if valLoss < bestValidationLoss
				bestValidationLoss = valLoss
				bestModel = deepcopy(ann)
				# restart patience
				patienceCount = 0
			else
				patienceCount += 1
			end
		end
	end
	if size(validationDataset[1], 1) > 0
		return (bestModel, trainingLosses, validationLosses, testLosses)
	else
		return (ann, trainingLosses, validationLosses, testLosses)
	end
end;

function trainClassANN(topology::AbstractArray{<:Int, 1},
	trainingDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 1}};
	validationDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 1}} =
	(Array{eltype(trainingDataset[1]), 2}(undef, 0, 0), falses(0)),
	testDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 1}} =
	(Array{eltype(trainingDataset[1]), 2}(undef, 0, 0), falses(0)),
	transferFunctions::AbstractArray{<:Function, 1} = fill(σ, length(topology)),
	maxEpochs::Int = 1000, minLoss::Real = 0.0, learningRate::Real = 0.01,
	maxEpochsVal::Int = 20, showText::Bool = false)

	# We'll make a reshape of the targets before call the previous function
	reshapedTrainingTargets = reshape(trainingDataset[2], :, 1)
	reshapedValidationTargets = size(validationDataset[2], 1) > 0 ? reshape(validationDataset[2], :, 1) : falses(0, 0)
	reshapedTestTargets = size(testDataset[2], 1) > 0 ? reshape(testDataset[2], :, 1) : falses(0, 0)

	trainClassANN(topology,
		(trainingDataset[1], reshapedTrainingTargets),
		validationDataset = (validationDataset[1], reshapedValidationTargets),
		testDataset = (testDataset[1], reshapedTestTargets),
		transferFunctions = transferFunctions,
		maxEpochs = maxEpochs, minLoss = minLoss, learningRate = learningRate, maxEpochsVal = maxEpochsVal, showText = showText)


	# If there is validationDataset, return bestmodel, else return ann. And histories
	if !isempty(validationDataset[1])
		return (bestModel, trainingLosses, validationLosses, testLosses)
	else
		return (ann, trainingLosses, validationLosses, testLosses)
	end
end;

# Hold out 
function holdOut(N::Int, P::Real)
	@assert ((P >= 0.0) & (P <= 1.0))
	indexes = randperm(N)
	ind = floor(Int, N * (1 - P))
	return (indexes[1:ind], indexes[ind+1:N])
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
	@assert ((Pval + Ptest) < 1.0)
	(train, test) = holdOut(N, Ptest)
	newN = size(train, 1)
	newPval = (N * Pval) / newN
	(train_pos, val) = holdOut(newN, newPval)
	return (train[train_pos], train[val], test)
end;

# confusion matrix

function confusionMatrix(outputs::AbstractArray{Bool, 1}, targets::AbstractArray{Bool, 1})
	@assert length(outputs) == length(targets) "Outputs and targets must have the same length"

	# Initialize the confusion matrix
	tp = sum(outputs .& targets)
	tn = sum(.!outputs .& .!targets)
	fp = sum(outputs .& .!targets)
	fn = sum(.!outputs .& targets)

	confusion_matrix = [tn fp; fn tp]

	# Calculate metrics
	accuracy = (tp + tn) / (tp + tn + fp + fn)
	error_rate = (fp + fn) / (tp + tn + fp + fn)

	sensitivity = (tp + fn) != 0 ? tp / (tp + fn) : 1
	specificity = (tn + fp) != 0 ? tn / (tn + fp) : 1
	positive_predictive_value = (tp + fp) != 0 ? tp / (tp + fp) : 1
	negative_predictive_value = (tn + fn) != 0 ? tn / (tn + fn) : 1


	# sensitivity
	if tp + fn == 0
		sensitivity = 1
		positive_predictive_value = 1
	end

	# specificity
	if tn + fp == 0
		specificity = 1
		negative_predictive_value = 1
	end

	# Calculate F-score
	if sensitivity == 0 && positive_predictive_value == 0
		f_score = 0
	else
		f_score = 2 * (positive_predictive_value * sensitivity) / (positive_predictive_value + sensitivity)
	end

	return Dict(
		"accuracy" => accuracy,
		"error_rate" => error_rate,
		"sensitivity" => sensitivity,
		"specificity" => specificity,
		"positive_predictive_value" => positive_predictive_value,
		"negative_predictive_value" => negative_predictive_value,
		"f_score" => f_score,
		"confusion_matrix" => confusion_matrix,
	)

end;

function confusionMatrix(outputs::AbstractArray{<:Real, 1}, targets::AbstractArray{Bool, 1}; threshold::Real = 0.5)
	outputs = outputs .> threshold
	confusionMatrix(outputs, targets)
end;

function confusionMatrix(outputs::AbstractArray{Bool, 2}, targets::AbstractArray{Bool, 2}; weighted::Bool = true)
	#TODO: here is the confusion
	# Check that outputs and targets have the same number of columns
	@assert size(outputs, 2) == size(targets, 2) && size(outputs, 2) != 2 && size(targets, 2) != 2

	num_classes = size(targets, 2)

	# Handle binary case - single column
	if num_classes == 1
		# Call the existing binary confusionMatrix
		return confusionMatrix(outputs[:, 1], targets[:, 1])
	end

	# Reserve memory for the metrics
	sensitivity = zeros(Float64, num_classes)
	specificity = zeros(Float64, num_classes)
	ppv = zeros(Float64, num_classes)
	npv = zeros(Float64, num_classes)
	f1_score = zeros(Float64, num_classes)
	confusion_matrix = zeros(Int, num_classes, num_classes)

	# Calculate metrics for each class
	for class in 1:num_classes
		output_class = outputs[:, class]
		target_class = targets[:, class]

		# Check if there are patterns in that class
		if sum(target_class) > 0  # Only compute if there are true instances in the target
			metrics = confusionMatrix(output_class, target_class)  # Call the existing binary function
			sensitivity[class] = metrics["sensitivity"]
			specificity[class] = metrics["specificity"]
			ppv[class] = metrics["positive_predictive_value"]
			npv[class] = metrics["negative_predictive_value"]
			f1_score[class] = metrics["f_score"]
		end
	end


	# Fill the confusion matrix
	for i in 1:num_classes
		for j in 1:num_classes
			confusion_matrix[i, j] = sum(targets[:, i] .& outputs[:, j])
		end
	end

	# Calculate macro or weighted averages
	if weighted
		total_count = size(outputs, 1)
		counts = sum(targets, dims = 1)

		avg_sensitivity = sum(sensitivity .* counts[1, :]') / total_count
		avg_specificity = sum(specificity .* counts[1, :]') / total_count
		avg_ppv = sum(ppv .* counts[1, :]') / total_count
		avg_npv = sum(npv .* counts[1, :]') / total_count
		avg_f1_score = sum(f1_score .* counts[1, :]') / total_count
	else
		avg_sensitivity = mean(sensitivity[sensitivity.>0])
		avg_specificity = mean(specificity[specificity.>0])
		avg_ppv = mean(ppv[ppv.>0])
		avg_npv = mean(npv[npv.>0])
		avg_f1_score = mean(f1_score[f1_score.>0])
	end

	# Calculate overall accuracy and error rate using the previous function
	accu = accuracy(Bool.(outputs), Bool.(targets))

	error_rate = 1 - accu

	return Dict(
		"nombre" => "primero",
		"accuracy" => accu,
		"error_rate" => error_rate,
		"sensitivity" => avg_sensitivity,
		"specificity" => avg_specificity,
		"positive_predictive_value" => avg_ppv,
		"negative_predictive_value" => avg_npv,
		"f_score" => avg_f1_score,
		"confusion_matrix" => confusion_matrix,
	)
end;

function confusionMatrix(outputs::AbstractArray{<:Real, 2}, targets::AbstractArray{Bool, 2}; weighted::Bool = true)
	# function classifyOutputs for conversion
	bool_outputs = classifyOutputs(outputs)

	# Call the previous confusionMatrix function
	return confusionMatrix(bool_outputs, targets; weighted = weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any, 1}, targets::AbstractArray{<:Any, 1}; weighted::Bool = true)
	@assert(all([in(output, unique(targets)) for output in outputs]))

	# Get the unique classes across both outputs and targets 
	# vcat concatenates the arrays vertically
	# unique returns the unique elements in the array
	class_vector = unique(vcat(outputs, targets))

	# One-hot encode both outputs and targets
	one_hot_outputs = oneHotEncoding(outputs, class_vector)
	one_hot_targets = oneHotEncoding(targets, class_vector)

	return confusionMatrix(one_hot_outputs, one_hot_targets; weighted = weighted)
end;

function printMetrics(metrics::Dict)
	confusion_matrix = metrics["confusion_matrix"]
	tn, fp = confusion_matrix[1, 1], confusion_matrix[1, 2]
	fn, tp = confusion_matrix[2, 1], confusion_matrix[2, 2]

	println("                         Prediction")
	println("                  +----------+----------+")
	println("                  | Negative | Positive |")
	println("       +----------+----------+----------+")
	println("       | Negative |    $tn    |    $fp      |")
	println("  Real +----------+----------+----------+")
	println("       | Positive |    $fn    |    $tp      |")
	println("       +----------+----------+----------+")

	println("Accuracy: ", metrics["accuracy"])
	println("Error rate: ", metrics["error_rate"])
	println("Sensitivity/Recall: ", metrics["sensitivity"])
	println("Specificity: ", metrics["specificity"])
	println("Precision/Positive predictive value: ", metrics["positive_predictive_value"])
	println("Negative predictive value: ", metrics["negative_predictive_value"])
	println("F-Score: ", metrics["f_score"])
end;

function printConfusionMatrix(outputs::AbstractArray{Bool, 1}, targets::AbstractArray{Bool, 1})
	metrics = confusionMatrix(outputs, targets)
	printMetrics(metrics)
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real, 1}, targets::AbstractArray{Bool, 1}; threshold::Real = 0.5)
	confusion_matrix = confusionMatrix(outputs, targets; threshold = threshold)
	printMetrics(confusion_matrix)
end;

# Cross validation
function crossvalidation(N::Int64, k::Int64)
	sorted_vector = collect(1:k)
	repeated_vector = repeat(sorted_vector, Int(ceil(N / k)))
	repeated_vector = repeated_vector[1:N]
	return shuffle!(repeated_vector)
end

function crossvalidation(targets::AbstractArray{Bool, 2}, k::Int64)
	@assert size(targets, 2) < 2 "Targets must be a 2D array with more than one column."
	N = size(targets, 1)
	indexes = zeros(Int, N)

	for class in 1:size(targets, 2)
		# Number of instances for the current class
		class_indexes = findall(targets[:, class])
		class_count = length(class_indexes)
		# Generate cross-validation indexes for the current class
		stratified_indexes = crossvalidation(class_count, k)

		# Assign stratified_indexes to the corresponding positions in indexes
		indexes[class_indexes] .= stratified_indexes
	end
	return indexes
end

function crossvalidation(targets::AbstractArray{<:Any, 1}, k::Int64)
	labels = unique(targets)
	# One-hot encode both outputs and targets
	one_hot_targets = oneHotEncoding(targets, labels)
	return crossvalidation(one_hot_targets, k)
end

function crossvalidationWithoutOneHot(targets::AbstractArray{<:Any, 1}, k::Int64)
	labels = unique(targets)
	indexes = collect(1:size(targets, 1))
	for i in 1:length(labels)
		total = sum(targets .== labels[i])
		cv_indexes = crossvalidation(total, k)
		indexes[targets.==labels[i]] .= cv_indexes
	end
	return indexes
end

# Train ANN U5

function trainClassANN(topology::AbstractArray{<:Int, 1},
	trainingDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}},
	kFoldIndices::Array{Int64, 1};
	transferFunctions::AbstractArray{<:Function, 1} = fill(σ, length(topology)),
	maxEpochs::Int = 1000, minLoss::Real = 0.0, learningRate::Real = 0.01, repetitionsTraining::Int = 1,
	validationRatio::Real = 0.0, maxEpochsVal::Int = 20)

	# Initial asserts
	@assert length(topology) > 1 "The topology must have at least 2 elements (inputs and outputs)."
	@assert size(trainingDataset[1], 1) == size(trainingDataset[2], 1) "The number of patterns and labels must match."
	@assert length(kFoldIndices) == size(trainingDataset[1], 1) "The length of kFoldIndices must match the number of samples in `inputs`."
	@assert 0.0 <= validationRatio <= 1.0 "The validationRatio must be in the range [0, 1]."
	@assert repetitionsTraining > 0 "The repetitionsTraining parameter must be a positive integer."
	@assert length(transferFunctions) == length(topology) "The length of transferFunctions must match the length of topology."
	@assert learningRate > 0 "The learningRate must be positive."
	@assert maxEpochs > 0 "The maxEpochs parameter must be a positive integer."
	@assert minLoss >= 0 "The minLoss parameter must be a non-negative value."


	# obtain the number of folds from the vector of indices
	numFolds = length(unique(kFoldIndices))

	# Create one vector per metric(for ex., accuracy,f1, specificity)
	testAccuracies = Array{Float64, 1}(undef, numFolds)
	testErrorRate = Array{Float64, 1}(undef, numFolds)
	testSensitivity = Array{Float64, 1}(undef, numFolds)
	testSpecificity = Array{Float64, 1}(undef, numFolds)
	testPPV = Array{Float64, 1}(undef, numFolds)
	testNPV = Array{Float64, 1}(undef, numFolds)
	testF1Score = Array{Float64, 1}(undef, numFolds)
	foldAnns = Array{Any}(undef, numFolds)

	# Separate tagets and inputs
	(inputs, targets) = trainingDataset

	# For each fold, load inputs and targets
	for numFold in 1:numFolds
		trainingInputs = inputs[kFoldIndices.!=numFold, :]
		testInputs = inputs[kFoldIndices.==numFold, :]
		trainingTargets = targets[kFoldIndices.!=numFold, :]
		testTargets = targets[kFoldIndices.==numFold, :]

		# Execute training loop using the functions createdinprevious
		# AAN are not deterministic, so we must repeat each fold several times and
		# additional vectors should be createdto save the values for each iteration
		testAccuraciesEachRepetition = Array{Float64, 1}(undef, repetitionsTraining)
		testErrorRateEachRepetition = Array{Float64, 1}(undef, repetitionsTraining)
		testSensitivityEachRepetition = Array{Float64, 1}(undef, repetitionsTraining)
		testSpecificityEachRepetition = Array{Float64, 1}(undef, repetitionsTraining)
		testPPVEachRepetition = Array{Float64, 1}(undef, repetitionsTraining)
		testNPVEachRepetition = Array{Float64, 1}(undef, repetitionsTraining)
		testF1ScoreEachRepetition = Array{Float64, 1}(undef, repetitionsTraining)
		ann = nothing

		for numTraining in 1:repetitionsTraining
			if validationRatio > 0
				# to train an ANN with a validation set, it is needed to call the holdOutfunction.
				(trainingIndices, validationIndices) = holdOut(
					size(trainingInputs, 1),
					validationRatio * size(trainingInputs, 1) / size(inputs, 1))
				(ann, trainingLosses, validationLosses, testLosses) = trainClassANN(
					topology,
					(trainingInputs[trainingIndices, :], trainingTargets[trainingIndices, :]),
					validationDataset = (
						trainingInputs[validationIndices, :],
						trainingTargets[validationIndices, :]),
					testDataset = (testInputs, testTargets),
					transferFunctions = transferFunctions,
					maxEpochs = maxEpochs, minLoss = minLoss, learningRate = learningRate,
					maxEpochsVal = maxEpochsVal)
			else
				# Train ANN using training andtest sets.
				(ann, trainingLosses, validationLosses, testLosses) = trainClassANN(
					topology,
					(trainingInputs, trainingTargets),
					testDataset = (testInputs, testTargets),
					transferFunctions = transferFunctions,
					maxEpochs = maxEpochs, minLoss = minLoss, learningRate = learningRate,
					maxEpochsVal = maxEpochsVal)
			end
			# Calculate metrics with confussionMatrix() function and save themfor each of the iterations
			# 4.2
			outputs = ann(testInputs')
			metrics = confusionMatrix(outputs', testTargets)
			testAccuraciesEachRepetition[numTraining] = metrics["accuracy"]
			testErrorRateEachRepetition[numTraining] = metrics["error_rate"]
			testSensitivityEachRepetition[numTraining] = metrics["sensitivity"]
			testSpecificityEachRepetition[numTraining] = metrics["specificity"]
			testPPVEachRepetition[numTraining] = metrics["positive_predictive_value"]
			testNPVEachRepetition[numTraining] = metrics["negative_predictive_value"]
			testF1ScoreEachRepetition[numTraining] = metrics["f_score"]
		end
		# do not forget to provide the result of averaging the values of these vectors for each metric
		foldAnns[numFold] = deepcopy(ann)
		testAccuracies[numFold] = mean(testAccuraciesEachRepetition)
		testErrorRate[numFold] = mean(testErrorRateEachRepetition)
		testSensitivity[numFold] = mean(testSensitivityEachRepetition)
		testSpecificity[numFold] = mean(testSpecificityEachRepetition)
		testPPV[numFold] = mean(testPPVEachRepetition)
		testNPV[numFold] = mean(testNPVEachRepetition)
		testF1Score[numFold] = mean(testF1ScoreEachRepetition)
	end # Kfold loop ends

	# What is the best fold? We will see the best F1 f_score
	_, bestFoldF1Score = findmax(testF1Score)
	# Save the best ANN model
	bestANN = deepcopy(foldAnns[bestFoldF1Score])
	# Save the best metrics
	bestAccuracy = testAccuracies[bestFoldF1Score][1]
	bestF1Score = testF1Score[bestFoldF1Score][1]
	bestErrorRate = testErrorRate[bestFoldF1Score][1]
	bestSensitivity = testSensitivity[bestFoldF1Score][1]
	bestSpecificity = testSpecificity[bestFoldF1Score][1]
	bestPPV = testPPV[bestFoldF1Score][1]
	bestNPV = testNPV[bestFoldF1Score][1]

	return Dict(
		"model" => bestANN,
		"accuracy" => bestAccuracy,
		"error_rate" => bestErrorRate,
		"sensitivity" => bestSensitivity,
		"specificity" => bestSpecificity,
		"positive_predictive_value" => bestPPV,
		"negative_predictive_value" => bestNPV,
		"f_score" => bestF1Score,
	)
end;

function trainClassANN(topology::AbstractArray{<:Int, 1},
	trainingDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 1}},
	kFoldIndices::Array{Int64, 1};
	transferFunctions::AbstractArray{<:Function, 1} = fill(σ, length(topology)),
	maxEpochs::Int = 1000, minLoss::Real = 0.0, learningRate::Real = 0.01, repetitionsTraining::Int = 1,
	validationRatio::Real = 0.0, maxEpochsVal::Int = 20)

	reshapedTrainingTargets = reshape(trainingDataset[2], :, 1)

	return trainClassANN(topology,
		(trainingDataset[1], reshapedTrainingTargets), kFoldIndices,
		transferFunctions = transferFunctions,
		maxEpochs = maxEpochs, minLoss = minLoss, learningRate = learningRate,
		repetitionsTraining = repetitionsTraining, validationRatio = validationRatio,
		maxEpochsVal = maxEpochsVal)
end;

function generateModel(estimator::Symbol, modelHyperParameters::Dict{String})
	model = nothing
	# 5. In case a validation set is needed, e.g. wi, split the training set into two parts. To do this, use the holdOut function. 
	if estimator == :ANN
		if haskey(modelHyperParameters, "validation_fraction") && (modelHyperParameters["validation_fraction"] > 0.0)
			model = MLPClassifier(hidden_layer_sizes = modelHyperParameters["topology"],
				max_iter = modelHyperParameters["maxEpochs"],
				learning_rate_init = modelHyperParameters["learningRate"],
				validation_fraction = modelHyperParameters["validation_fraction"],
				early_stopping = true)
		else
			model = MLPClassifier(hidden_layer_sizes = modelHyperParameters["topology"],
				max_iter = modelHyperParameters["maxEpochs"],
				learning_rate_init = modelHyperParameters["learningRate"])
		end
	elseif estimator == :SVM
		model = SVC(kernel = modelHyperParameters["kernel"],
			C = modelHyperParameters["C"], gamma = modelHyperParameters["gamma"])
	elseif estimator == :DecisionTree
		model = DecisionTreeClassifier(max_depth = modelHyperParameters["max_depth"],
			random_state = modelHyperParameters["random_state"])
	elseif estimator == :kNN
		model = KNeighborsClassifier(n_neighbors = modelHyperParameters["k"])
	else
		error("Unsupported model type: $estimator")
	end
	return model
end;

function trainClassEnsemble(estimators::AbstractArray{Symbol, 1},
	modelsHyperParameters::AbstractArray{Dict{String, Any}, 1},
	trainingDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}},
	kFoldIndices::Array{Int64, 1})

	# Assertions for defensive programming
	@assert length(estimators) == length(modelsHyperParameters) "Each estimator should have corresponding hyperparameters."
	@assert size(trainingDataset[1], 1) == size(trainingDataset[2], 1) "Number of input samples must match number of target labels."
	@assert length(kFoldIndices) == size(trainingDataset[1], 1) "Length of kFoldIndices should match the number of input samples."
	@assert all(estimator in [:ANN, :SVM, :DecisionTree, :kNN] for estimator in estimators) "Unsupported model type in estimators array."

	# Split dataset
	(inputs, targets) = trainingDataset

	# 1. Create a vector with k elements, which will contain the test results of the cross-validation process with the selected metric.
	k = maximum(kFoldIndices)
	testResults = Array{Dict}(undef, k)
	classifiers = Array{Any}(undef, k)

	numModels = length(estimators)
	# Initialice models like an array to store each model
	models = Array{Any}(undef, numModels)
	modelLabels = Dict(
		:ANN => "ANN",
		:SVM => "SVM",
		:DecisionTree => "DT",
		:kNN => "KNN",
	)


	# 2. Make a loop with k iterations (k folds) where within each iteration from the matrices of desired inputs and outputs, by means of the vector of indices resulting from the previous function, 4 matrices are created: desired inputs and outputs for training and test.
	for i in 1:k
		# Select training and validation data based on fold indices
		train_inputs = inputs[kFoldIndices.!=i, :]
		test_inputs = inputs[kFoldIndices.==i, :]
		train_targets = targets[kFoldIndices.!=i, :]
		test_targets = targets[kFoldIndices.==i, :]

		train_targets = reshape(train_targets, :)

		# 3. Within this another loop, add a call to generate the models, which can be any of ANN, SVM, DecisionTree or kNN.
		for (index, estimator) in enumerate(estimators)
			# Generate the model with the selected estimator and hyperparameters
			model = generateModel(estimator, modelsHyperParameters[index])
			# 4. Train those models by using the corresponding training set, i. e., the remaining K subsets non used for testing.
			fit!(model, train_inputs, train_targets)
			# Store each model in the array

			models[index] = (estimator, deepcopy(model))
		end
		# 6. Build the ensemble following one of the strategies described above (any of them) and calculate the test.
		stacking_classifier = StackingClassifier(
			estimators = [("m$(idx)_" * modelLabels[name], model) for (idx, (name, model)) in enumerate(models)],
			final_estimator = SVC(probability = true), n_jobs = -1)
		fit!(stacking_classifier, train_inputs, train_targets)
		# Deep copy
		classifiers[i] = deepcopy(stacking_classifier)
		outputs = stacking_classifier.predict(test_inputs)
		metrics = confusionMatrix(outputs, vec(test_targets))
		testResults[i] = metrics
	end
	# 7. Finally, provide the result of averaging the values of these vectors for each metric together with their standard deviations.
	accuracy = [tr["accuracy"] for tr in testResults]
	error_rate = [tr["error_rate"] for tr in testResults]
	sensitivity = [tr["sensitivity"] for tr in testResults]
	specificity = [tr["specificity"] for tr in testResults]
	positive_predictive_value = [tr["positive_predictive_value"] for tr in testResults]
	negative_predictive_value = [tr["negative_predictive_value"] for tr in testResults]
	f_score = [tr["f_score"] for tr in testResults]

	return Dict(
		"accuracy" => (mean(accuracy), std(accuracy)),
		"error_rate" => (mean(error_rate), std(error_rate)),
		"sensitivity" => (mean(sensitivity), std(sensitivity)),
		"specificity" => (mean(specificity), std(specificity)),
		"positive_predictive_value" => (mean(positive_predictive_value), std(positive_predictive_value)),
		"negative_predictive_value" => (mean(negative_predictive_value), std(negative_predictive_value)),
		"f_score" => (mean(f_score), std(f_score)),
		# "confusion_matrix" => (mean(testResults[:]["confusion_matrix"]), std(testResults[:]["confusion_matrix"])),
	)
end;

function trainClassEnsemble(baseEstimator::Symbol,
	modelsHyperParameters::Dict{String, Any},
	trainingDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}},
	kFoldIndices::Array{Int64, 1},
	NumEstimators::Int = 100)

	# Create an array of NumEstimators symbols of tyoe baseEstimator, the symbols could be :ANN, :SVM, :kNN or :DecisionTree
	estimators = [baseEstimator for i in 1:NumEstimators]
	modHyperParameters = [modelsHyperParameters for i in 1:NumEstimators]
	trainClassEnsemble(estimators,
		modHyperParameters,
		trainingDataset,
		kFoldIndices)
end

function modelCrossValidation(modelType::Symbol,
	modelHyperparameters::Dict,
	inputs::AbstractArray{<:Real, 2},
	targets::AbstractArray{<:Any, 1},
	crossValidationIndices::Array{Int64, 1})

	# Number of folds for cross-validation
	numFolds = maximum(crossValidationIndices)

	# Initialize an array to store metrics for each fold
	metrics = Vector{Dict{String, Any}}()

	# Perform cross-validation
	for fold in 1:numFolds
		# Select training and validation data based on fold indices
		train_inputs = inputs[crossValidationIndices.!=fold, :]
		val_inputs = inputs[crossValidationIndices.==fold, :]
		train_targets = targets[crossValidationIndices.!=fold]
		val_targets = targets[crossValidationIndices.==fold]

		# Model training and prediction
		predictions = []  # Placeholder for storing predictions
		if modelType == :ANN
			one_hot_targets = collect(oneHotEncoding(train_targets))
			val_targets = collect(oneHotEncoding(val_targets))

			# Define the transfer function for each layer in the topology
			topology = modelHyperparameters["topology"]
			transferFunctions = fill(tanh, length(topology))

			# Train ANN and obtain metrics
			train_result = trainClassANN(topology,
				(train_inputs, one_hot_targets);
				transferFunctions = transferFunctions, maxEpochs = modelHyperparameters["maxEpochs"], 
				minLoss = modelHyperparameters["minLoss"],
				learningRate = modelHyperparameters["learningRate"],
				maxEpochsVal = modelHyperparameters["maxEpochsVal"],
			)

			# Extract the model and make predictions on validation data
			(model,_,_,_) = train_result
			predictions = model(val_inputs')'
			predictions = [ (i == argmax(prediction) ? 1 : 0) for (i, prediction) in enumerate(predictions) ]
		elseif modelType == :SVM
			model = SVC(kernel = modelHyperparameters["kernel"], C = modelHyperparameters["C"])
			fit!(model, train_inputs, train_targets)
			predictions = predict(model, val_inputs)

		elseif modelType == :DecisionTree
			model = DecisionTreeClassifier(max_depth = modelHyperparameters["max_depth"],
				random_state = modelHyperparameters["random_state"])
			fit!(model, train_inputs, train_targets)
			predictions = predict(model, val_inputs)

		elseif modelType == :kNN
			model = KNeighborsClassifier(n_neighbors = modelHyperparameters["k"])
			fit!(model, train_inputs, train_targets)
			predictions = predict(model, val_inputs)

		else
			error("Unsupported model type: $modelType")
		end

		# Calculate metrics using the confusion matrix for each fold
		metrics_fold = confusionMatrix(predictions, val_targets)

		# Append the fold metrics dictionary to the array
		push!(metrics, metrics_fold)
	end

	return metrics
end
