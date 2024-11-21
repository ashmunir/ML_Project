import Pkg;
Pkg.add("Flux")

using Random
using Random:seed!
using Statistics
using Flux
using ScikitLearn
using CSV, DataFrames

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