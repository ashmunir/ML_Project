module ML1Utils

using Pkg: Pkg;
Pkg.add("Flux")

using Random
#using Random: seed!
using Statistics
using Flux
using ScikitLearn
#using CSV
#using DataFrames
using Printf
using Plots
using StatsPlots
using JSON

@sk_import svm:SVC
@sk_import tree:DecisionTreeClassifier
@sk_import neighbors:KNeighborsClassifier
@sk_import neural_network:MLPClassifier
@sk_import ensemble:StackingClassifier
@sk_import decomposition:PCA

export comment
export holdOutStratified
export normalizeData
export genModel
export bestModelPositions
export confusionMatrix
export printConfusionMatrix
export crossvalidation
export trainClassEnsemble
export oneHotEncoding
export drawResults

function comment(text::String)
    width = 100
    println()
    println("/" * repeat("-",width) * "/")
    println("/ " * text * repeat(" ",width-length(text)-1) * "/")
    println("/" * repeat("-",width) * "/")
end

function holdOutStratified(classes::Vector{Int}, P::Real)
	# Verificar que P está entre 0 y 1
	@assert ((P >= 0.0) & (P <= 1.0))

	# Diccionario para almacenar índices separados por clase
	class_indices = Dict(c => findall(==(c), classes) for c in unique(classes))

	# Inicializar vectores para los índices de train y test
	train_idx = Int[]
	test_idx = Int[]

	# Repartir los elementos estratificadamente
	for (_, indices) in class_indices
		# Barajar los índices
		shuffled = randperm(length(indices))
		class_cut = round(Int64, length(indices) * P)

		# Obtener índices de prueba y entrenamiento para esta clase
		test_idx_class = indices[shuffled[1:class_cut]]
		train_idx_class = indices[shuffled[class_cut+1:end]]

		# Agregar a las listas generales
		append!(test_idx, test_idx_class)
		append!(train_idx, train_idx_class)
	end

	return (train_idx, test_idx)
end

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
end

function genModel(modelsHyperParameters::Dict{String})
	estimator = modelsHyperParameters["estimator"]
	if estimator == :SVM
		return SVC(kernel = modelsHyperParameters["kernel"],
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

function bestModelPositions(models_data::Vector{<:Any})

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

end

function printConfusionMatrix(confusion_matrix::AbstractMatrix{Int}, class_labels::Vector{String} = nothing)
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

function crossvalidation(N::Int64, k::Int64)
	sorted_vector = collect(1:k)
	repeated_vector = repeat(sorted_vector, Int(ceil(N / k)))
	repeated_vector = repeated_vector[1:N]
	return shuffle!(repeated_vector)
end

function crossvalidation(targets::AbstractArray{Bool, 2}, k::Int64)
	@assert size(targets, 2) < 2 "Targets must be a 2D array with more than one column."
	N = size(targets, 1)
	num_classes = size(targets, 2)
	indexes = zeros(Int, N)

	for class in 1:num_classes
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


function trainClassEnsemble(modelsHyperParameters::AbstractArray{Dict{String, Any}, 1},
	trainingDataset::Tuple{AbstractArray{<:Real}, AbstractVector{<:Any}},
	kFoldIndices::Array{Int64, 1})

	(inputs, targets) = trainingDataset

	numFolds = length(unique(kFoldIndices))

	# Punto 1: almacenar cada métrica en cada fold
	metrics = Array{Dict{String, Any}, 1}(undef, numFolds)
	fold_accuracies = Vector{Vector{Tuple{Int, Any, Float64}}}()
	mean_accuracies = Array{Float64, 1}()

	# Punto 3: Crear los modelos
	numModels = length(modelsHyperParameters)
	models = Array{Any}(undef, numModels)
	modelName = Dict(
		:ANN => "ANN",
		:SVM => "SVM",
		:DecisionTree => "DT",
		:KNN => "KNN",
	)

	# Punto 2
	for numFold in 1:numFolds
		# Preparar y dividir los datos con los índices de CV (4 matrices)
		trainingInputs = inputs[kFoldIndices.!=numFold, :]
		testInputs = inputs[kFoldIndices.==numFold, :]
		trainingTargets = targets[kFoldIndices.!=numFold, :]
		testTargets = targets[kFoldIndices.==numFold, :]

		# Reshape trainingTargets to ensure it’s 1D and not have warnings
		trainingTargets = reshape(trainingTargets, :)

		vectorcito = Vector{Tuple{Int, Any, Float64}}()

		for (index, modelHyperParameters) in enumerate(modelsHyperParameters)
			model = genModel(modelHyperParameters)

			# Punto 4: Entrenar el modelo
			fit!(model, trainingInputs, trainingTargets)

			models[index] = (modelHyperParameters["estimator"], deepcopy(model))
			acc = score(model, testInputs, testTargets)
			push!(vectorcito, (index, modelHyperParameters["estimator"], acc))
		end

		push!(fold_accuracies, vectorcito)

		mean_accuracies = averageAccuracies(fold_accuracies)

		best_model = bestModelPositions(mean_accuracies)

		# Punto 6: Construir el modelo de ensamblaje (stacking)
		ensemble_model = StackingClassifier(
			estimators = [(string(modelName[modelsHyperParameters[i]["estimator"]], "_", i), models[i][2]) for i in best_model],
			final_estimator = SVC(probability = true),
			n_jobs = -1,
		)

		# Entrenar el modelo de esamblaje
		fit!(ensemble_model, trainingInputs, trainingTargets)

		# Realizar predicciones en el conjunto de prueba

		predictions = ensemble_model.predict(testInputs)

		metrix = confusionMatrix(predictions, vec(testTargets))

		# Almacenar cada métrica en los vectores correspondientes
		# push!(metrics, deepcopy(metrix))
		metrics[numFold] = deepcopy(metrix)
	end

	accu = [tr["accuracy"] for tr in metrics]
	error_rate = [tr["error_rate"] for tr in metrics]
	sensitivity = [tr["sensitivity"] for tr in metrics]
	specificity = [tr["specificity"] for tr in metrics]
	positive_predictive_value = [tr["positive_predictive_value"] for tr in metrics]
	negative_predictive_value = [tr["negative_predictive_value"] for tr in metrics]
	f_score = [tr["f_score"] for tr in metrics]

	# Calcular las medias y desviaciones estándar de las métricas
	mean_metrics = Dict(
		"accuracy" => (mean(accu), std(accu)),
		"error_rate" => (mean(error_rate), std(error_rate)),
		"sensitivity" => (mean(sensitivity), std(sensitivity)),
		"specificity" => (mean(specificity), std(specificity)),
		"positive_predictive_value" => (mean(positive_predictive_value), std(positive_predictive_value)),
		"negative_predictive_value" => (mean(negative_predictive_value), std(negative_predictive_value)),
		"f_score" => (mean(f_score), std(f_score)),
		"models_accuracies" => mean_accuracies
	)

	return mean_metrics
end

function oneHotEncoding(feature::AbstractArray{<:Any, 1}, classes::AbstractArray{<:Any, 1})
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
end

# Function to transorm in one-hot-encoding based on the values of the array passed
oneHotEncoding(feature::AbstractArray{<:Any, 1}) = oneHotEncoding(feature, unique(feature))

# Function to transform a feature into one-hot encoding in base of the classes passed by parameter
function oneHotEncoding(feature::AbstractArray{<:Any, 1}, classes::AbstractArray{<:Any, 1})
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
end

oneHotEncoding(feature::AbstractArray{<:Any, 1}) = oneHotEncoding(feature, unique(feature));

oneHotEncoding(feature::AbstractArray{Bool, 1}) = reshape(feature, :, 1);

function drawResults(x, y; colors, target_names=nothing, filename="image.png")
    # Get the number of classes from the one-hot encoded matrix
    num_classes = size(y, 2)

    # Check that the number of colors matches the number of classes
    @assert length(colors) == num_classes "Number of colors must match the number of classes"

    # If target_names are provided, ensure they match the number of classes
    if !isnothing(target_names)
        @assert length(target_names) == num_classes "Number of target names must match the number of classes"
        labels = target_names
    else
        labels = [string("Class ", i) for i in 1:num_classes]
    end

    # Initialize the plot
    fig = plot(dpi=1000)

    # Plot each class separately
    for i in 1:num_classes
        # Logical indexing to get points of class `i`
        class_indices = y[:, i] .== 1
        scatter!(x[class_indices, 1], x[class_indices, 2], markercolor=colors[i], label=labels[i])
    end
    savefig(filename)

    return fig
end



# Obtain values MEAN and STANDARD DESVIATION of an array to use in normalization:
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real, 2})
	return mean(dataset, dims = 1), std(dataset, dims = 1)
end

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real, 2})
	return minimum(dataset, dims = 1), maximum(dataset, dims = 1)
end

function normalizeMinMax!(dataset::AbstractArray{<:Real, 2},
	normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
	minValues = normalizationParameters[1]
	maxValues = normalizationParameters[2]
	dataset .-= minValues
	dataset ./= (maxValues .- minValues)
	# eliminate any atribute that do not add information
	dataset[:, vec(minValues .== maxValues)] .= 0
	return dataset
end

function normalizeMinMax!(dataset::AbstractArray{<:Real, 2})
	normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
end

function normalizeMinMax(dataset::AbstractArray{<:Real, 2},
	normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
	normalizeMinMax!(copy(dataset), normalizationParameters)
end

function normalizeMinMax(dataset::AbstractArray{<:Real, 2})
	normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset))
end

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
end

function normalizeZeroMean(dataset::AbstractArray{<:Real, 2})
	normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset))
end



# MULTICLASS CONFUSION MATRIX
# Accuracy functions
function accuracy(outputs::AbstractArray{Bool, 1}, targets::AbstractArray{Bool, 1})
	return mean(outputs .== targets)
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
end

function accuracy(outputs::AbstractArray{<:Real, 2}, targets::AbstractArray{Bool, 2};
	threshold::Real = 0.5)
	@assert(all(size(outputs) .== size(targets)))
	if (size(targets, 2) == 1)
		return accuracy(outputs[:, 1], targets[:, 1])
	else
		return accuracy(classifyOutputs(outputs; threshold = threshold), targets)
	end
end

function confusionMatrix(outputs::AbstractArray{Bool, 2}, targets::AbstractArray{Bool, 2}; weighted::Bool = true)
	# Ensure outputs and targets have the same number of columns
	@assert size(outputs, 2) == size(targets, 2)
	@assert size(outputs, 2) != 2
	@assert size(targets, 2) != 2

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
		"accuracy" => accu,
		"error_rate" => error_rate,
		"sensitivity" => avg_sensitivity,
		"specificity" => avg_specificity,
		"positive_predictive_value" => avg_ppv,
		"negative_predictive_value" => avg_npv,
		"f_score" => avg_f1_score,
		"confusion_matrix" => confusion_matrix,
	)
end

function confusionMatrix(outputs::AbstractArray{<:Any}, targets::AbstractArray{<:Any}; weighted::Bool = true)
	# Ensure that outputs and targets are of the same length
	@assert length(outputs) == length(targets) "Outputs and targets must be of the same length."
	# Check that all output classes are included in the desired output classes
	@assert all([in(output, unique(targets)) for output in outputs]) "All output classes must be present in the target classes."

	class_vector = unique(vcat(outputs, targets))

	# One-hot encode both outputs and targets
	one_hot_outputs = oneHotEncoding(outputs, class_vector)
	one_hot_targets = oneHotEncoding(targets, class_vector)

	return confusionMatrix(one_hot_outputs, one_hot_targets; weighted = weighted)
end

function averageAccuracies(data::Vector{Vector{Tuple{Int, Any, Float64}}})
    # Verificar que todas las filas tienen el mismo número de columnas
    n_columns = length(data[1])
    @assert all(length(row) == n_columns for row in data) "Todas las filas deben tener el mismo número de columnas"

    # Inicializar un vector para los resultados
    results = Vector{Tuple{Int, Any, Float64}}()

    # Iterar por las columnas
    for col in 1:n_columns
        # Extraer el segundo atributo (estimator) y los valores de accuracy
        estimators = [row[col][2] for row in data]
        accuracies = [row[col][3] for row in data]

        # Asegurar que todos los estimators son iguales dentro de la columna
        @assert all(est == estimators[1] for est in estimators) "Los valores del segundo atributo deben ser consistentes en una columna"

        # Calcular el promedio
        avg_accuracy = mean(accuracies)

        # Pushear el índice, el segundo atributo y el promedio
        push!(results, (col, estimators[1], avg_accuracy))
    end

    return results
end



end